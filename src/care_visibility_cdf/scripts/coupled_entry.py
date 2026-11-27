"""Generate a view and a short outgoing task edge using the same swept cells."""
import time

import numpy as np
import torch

from optical_view import intersects_cell
from paired_view import TaskRouteComparison as PairedView


class TaskRouteComparison(PairedView):
    def _first_entries(self, runtime, start, views, ends, accel):
        sweep = self._edges(runtime, start, views, ends, accel)
        if sweep is None:
            return None
        grid, _ = self._geometry(runtime)
        poses, routes, unknown, occupied = sweep
        entries = []
        for view, path, route in zip(views, poses, routes):
            for index, cells in enumerate(route):
                if occupied[cells].any():
                    break
                missing = cells[unknown[cells]]
                if not len(missing):
                    continue
                if index:
                    # Extend the short edge while its evidence requirement is unchanged.
                    last = index
                    for later in range(index + 1, len(route)):
                        step = route[later]
                        if occupied[step].any() or not np.array_equal(step[unknown[step]], missing):
                            break
                        last = later
                    entries.append(dict(view=view, target=path[last], points=grid[missing]))
                break
        return entries

    def _assess(self, runtime, stopped, goal, entries, sensors):
        if not entries:
            return []
        # A different sensor can cover each cell, including on the approach.
        approaches = [[(stopped + entry['view']) / 2., entry['view']] for entry in entries]
        inputs = np.concatenate([np.c_[entry['points'], np.tile(q, (len(entry['points']), 1))]
                                 for entry, poses in zip(entries, approaches) for q in poses])
        with torch.no_grad():
            predictions = runtime.model(torch.as_tensor(
                inputs, dtype=torch.float32, device=runtime.device)).detach().cpu().numpy()
        offset = 0
        for entry, poses in zip(entries, approaches):
            points = entry['points']
            count = len(points)
            scores = predictions[offset:offset + 2 * count].reshape(2, count, 8)
            offset += 2 * count
            covered = np.zeros(count, dtype=bool)
            margin = np.full(count, -np.inf)
            best_score = -np.inf
            for index, q in enumerate(poses):
                for sensor in sensors:
                    planes, margins, _ = self._optics(runtime, q, sensor, points)
                    face_margin = (margins + .025 * np.abs(planes).sum(axis=1)).min(axis=1)
                    margin = np.maximum(margin, face_margin)
                    for j in np.flatnonzero((face_margin >= 0.) & ~covered):
                        covered[j] = intersects_cell(planes, margins[j])
                    optical_score = np.clip(face_margin / .05, -2., 1.)
                    head_score = scores[index, :, sensor]
                    quality = float(np.mean(optical_score + .25 * np.clip(head_score, -1., 1.)))
                    if quality > best_score:
                        best_score = quality
                        entry.update(sensor=sensor, score=float(np.mean(head_score)),
                                     rank=1 + int(np.sum(scores[index].mean(axis=0) > head_score.mean())))
            view, target = entry['view'], entry['target']
            progress = np.linalg.norm(goal - view) - np.linalg.norm(goal - target)
            travel = np.max(np.abs(view - stopped))
            entry.update(nominal_terminal=bool(covered.all()), margin=float(margin.min()),
                         covered_cells=int(covered.sum()),
                         quality=float(covered.mean()) + .24 * best_score
                         + .55 * progress - .22 * travel - .09 * np.log1p(count))
        return sorted(entries, key=lambda entry: entry['quality'], reverse=True)

    def _search(self, runtime, start, stopped, goal, blocked, accel, previous_view, excluded,
                attempt, identity):
        began = time.perf_counter()
        sensors = [sensor for sensor in range(8) if sensor not in excluded]
        if not sensors:
            return None
        lower, upper = runtime.q_min_np, runtime.q_max_np
        rng = np.random.default_rng(attempt)
        random = rng.normal(size=(6, 7))
        random /= np.max(np.abs(random), axis=1, keepdims=True)
        direct = goal - stopped
        direct *= .48 / max(.48, np.max(np.abs(direct)))
        directions = np.vstack([direct, .52 * random[:3], .65 * direct + .40 * random[3:]])
        ends = np.clip(stopped + directions, lower, upper)
        sweep = self._edges(runtime, start, np.tile(stopped, (len(ends), 1)), ends, accel)
        if sweep is None:
            self.node._trace_observation('task_route_preview', vbc_identity=identity,
                                         snapshot_failures=1, reason='approach_snapshot')
            return None
        poses, routes, unknown, occupied = sweep
        views = []
        for path, route in zip(poses, routes):
            reachable = None
            for pose, cells in zip(path, route):
                if (unknown[cells] | occupied[cells]).any():
                    break
                if np.max(np.abs(pose - stopped)) >= .085:
                    reachable = pose
            if reachable is not None:
                views.append(reachable)
        views = self._deduplicate(views)
        if previous_view is not None:
            views = [q for q in views if np.max(np.abs(q - previous_view)) >= .08]

        entries, failures = [], 0
        for offset, view in enumerate(views):
            toward = goal - view
            toward *= .30 / max(.30, np.max(np.abs(toward)))
            # Use geometry at the view to find the outgoing frontier. Include
            # extension of its own approach and a different joint combination.
            continuation = view - stopped
            continuation *= .30 / max(.10, np.max(np.abs(continuation)))
            lateral = .50 * toward + .29 * random[offset % len(random)]
            targets = np.clip(view + np.vstack([toward, continuation, lateral]), lower, upper)
            batch = self._first_entries(runtime, start, np.tile(view, (len(targets), 1)), targets, accel)
            if batch is None:
                failures += 1
            else:
                entries.extend(batch)

        ranked = self._assess(runtime, stopped, goal, entries, sensors)
        choice = ranked[0] if ranked else None
        if choice is not None:
            # Move the observation knot using R1 at the cells required by its
            # outgoing edge, then relinearize the free-space requirement.
            proposed = self._propose(runtime, choice['view'],
                                     [(choice['points'], choice['target'])], previous_view, excluded)
            refinements = self._deduplicate(p['view'] for p in proposed
                                            if np.max(np.abs(p['view'] - choice['view'])) <= .29)[:2]
            if refinements:
                sweep = self._edges(runtime, start, np.tile(stopped, (len(refinements), 1)),
                                    np.array(refinements), accel)
                if sweep is None:
                    failures += 1
                else:
                    _, routes, unknown, occupied = sweep
                    safe = [pose for pose, route in zip(refinements, routes)
                            if all(not (unknown[cells] | occupied[cells]).any() for cells in route)]
                    if safe:
                        updated = self._first_entries(runtime, start, np.array(safe),
                            np.tile(choice['target'], (len(safe), 1)), accel)
                        if updated is None:
                            failures += 1
                        else:
                            ranked = self._assess(runtime, stopped, goal, updated, sensors) + ranked
                            choice = max(ranked, key=lambda entry: entry['quality'])
        if choice is not None:
            choice = dict(choice, start=stopped, goal=goal)
        self.node._trace_observation('task_route_preview', vbc_identity=identity,
            direct_points=blocked.tolist(), reachable_views=len(views), actual_swept_entries=len(entries),
            optical_complete=sum(p['nominal_terminal'] for p in ranked), snapshot_failures=failures,
            selected_points=None if choice is None else choice['points'].tolist(),
            selected_view=None if choice is None else choice['view'].tolist(),
            selected_route_target=None if choice is None else choice['target'].tolist(),
            selected_optical_cells=None if choice is None else choice['covered_cells'],
            selected_sensor=None if choice is None else choice['sensor'],
            wall_ms=1000. * (time.perf_counter() - began))
        return choice
