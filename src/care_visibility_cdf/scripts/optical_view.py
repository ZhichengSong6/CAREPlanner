"""Propose sensing waypoints with R1 and screen their optics before body sweeps."""
from itertools import combinations
import time

import numpy as np
import rospy
import torch
from care_confidence_map.srv import QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform

from independent_view import TaskRouteComparison as IndependentView


PLANE_TRIPLES = np.asarray(list(combinations(range(12), 3)))
BOX_PLANES = np.vstack([np.eye(3), -np.eye(3)])
HALF_CELL = .025


def intersects_cell(planes, margins):
    # A nonempty intersection of the two bounded polyhedra has a vertex.
    if np.min(margins + HALF_CELL * np.abs(planes).sum(axis=1)) < -1e-10:
        return False
    if np.min(margins) >= 0.:
        return True
    matrix = np.vstack([planes, BOX_PLANES])
    bounds = np.r_[margins, np.full(6, HALF_CELL)]
    triples = matrix[PLANE_TRIPLES]
    nonsingular = np.abs(np.linalg.det(triples)) > 1e-10
    vertices = np.linalg.solve(triples[nonsingular], -bounds[PLANE_TRIPLES[nonsingular], None])[:, :, 0]
    return bool(np.any(np.all(vertices @ matrix.T + bounds >= -1e-9, axis=1)))


class TaskRouteComparison(IndependentView):
    def _sweeps(self, start, stopped, ends, accel, runtime, *, samples=5, query=None):
        grid, memberships = self._geometry(runtime)
        fractions = np.linspace(1. / samples, 1., samples)
        poses = stopped + (ends-stopped)[:, None, :] * fractions[None, :, None]
        v = np.clip(ends-stopped, -.25, .25)
        brake = v * np.abs(v) / (2. * accel)
        brakes = np.clip(poses + brake[:, None, :] * np.sin(np.pi*fractions)[None, :, None],
                         runtime.q_min_np, runtime.q_max_np)
        groups = memberships(np.vstack([np.linspace(start, stopped, 3),
                                         poses.reshape(-1, 7), brakes.reshape(-1, 7)]))
        n = len(ends) * len(fractions)
        existing = groups[0]
        approach = np.setdiff1d(np.union1d(groups[1], groups[2]), existing)
        steps = [np.setdiff1d(np.union1d(body, brake), existing)
                 for body, brake in zip(groups[3:3+n], groups[3+n:])]
        ids = np.unique(np.concatenate([approach, *steps]))
        unknown = np.zeros(len(grid), dtype=bool)
        occupied = unknown.copy()
        if len(ids):
            request = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in grid[ids]],
                minimum_confidence=float(self.node.seen_threshold), horizon_s=0.)
            reply = (self.query if query is None else query)(request)
            if (reply is None or not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                    or not reply.instance or not 0. <= (rospy.Time.now() - reply.snapshot_stamp).to_sec() <= .1
                    or any(len(x) != len(ids) for x in (reply.inside_map, reply.confidence_now, reply.occupancy))):
                return None
            inside = np.asarray(list(reply.inside_map), dtype=bool)
            confidence = np.asarray(reply.confidence_now, dtype=float)
            occupancy = np.asarray(reply.occupancy, dtype=float)
            occupied[ids] = ~inside | ~np.isfinite(confidence) | ~np.isfinite(occupancy) | (occupancy > .5)
            unknown[ids] = (confidence < request.minimum_confidence) & ~occupied[ids]
        if (occupied[approach] | unknown[approach]).any():
            return None
        routes = [steps[i:i+len(fractions)] for i in range(0, n, len(fractions))]
        return grid, poses, routes, unknown, occupied

    @staticmethod
    def _optics(runtime, q, sensor, cells):
        transform = fk_transform(runtime.sensor_chains[sensor], dict(zip(DEFAULT_JOINT_NAMES, q)))
        hx, hy = np.tan(np.deg2rad([runtime.nom_hfov, runtime.nom_vfov])*.5)
        planes = np.array([[1, 0, hx], [-1, 0, hx], [0, 1, hy], [0, -1, hy],
                           [0, 0, 1], [0, 0, -1]], dtype=float)
        planes /= np.linalg.norm(planes, axis=1)[:, None]
        planes = planes @ transform[:3, :3].T
        margins = (cells - transform[:3, 3]) @ planes.T
        margins += [0., 0., 0., 0., -runtime.nom_z_min, runtime.nom_z_max]
        optimistic_margin = float(np.min(margins + HALF_CELL * np.abs(planes).sum(axis=1)))
        return planes, margins, optimistic_margin

    def _propose(self, runtime, stopped, entries, previous_view, excluded):
        candidates = []
        for cells, target in entries:
            xyz = torch.as_tensor(cells, device=runtime.device, dtype=torch.float32)
            q = torch.as_tensor(stopped[None, :], device=runtime.device, dtype=torch.float32)
            _, order = runtime.rank_sensors(xyz, q)
            optics = [self._optics(runtime, stopped, sensor, cells)[2] for sensor in range(8)]
            available = [int(s) for s in order if int(s) not in excluded]
            if not available:
                continue
            heads = dict.fromkeys((available[0], max(available, key=optics.__getitem__)))
            for sensor in heads:
                _, gradient = runtime.branch_value_and_grad_numpy(cells, stopped, sensor)
                scale = float(np.max(np.abs(gradient)))
                if not np.isfinite(scale) or scale < 1e-8:
                    continue
                r1 = gradient / scale
                optical = np.zeros(7)
                for joint in np.flatnonzero(r1):
                    shifted = stopped.copy()
                    shifted[joint] += .01
                    optical[joint] = self._optics(runtime, shifted, sensor, cells)[2] - optics[sensor]
                optical /= max(1e-8, np.max(np.abs(optical)))
                for direction in (r1, .5 * (r1 + optical)):
                    for distance in (.12, .28, .48):
                        pose = np.clip(stopped + distance * direction, runtime.q_min_np, runtime.q_max_np)
                        if previous_view is not None and np.max(np.abs(pose - previous_view)) < .08:
                            continue
                        planes, margins, upper = self._optics(runtime, pose, sensor, cells)
                        terminal = upper >= 0. and all(intersects_cell(planes, m) for m in margins)
                        if terminal or upper > optics[sensor] + .003:
                            candidates.append(dict(view=pose, points=cells, sensor=sensor, margin=upper,
                                nominal_terminal=terminal, target=target if terminal else None,
                                rank=1 + [int(s) for s in order].index(sensor)))
        if candidates:
            batch = np.vstack([np.c_[c['points'], np.tile(c['view'], (len(c['points']), 1))]
                               for c in candidates])
            with torch.no_grad():
                scores = runtime.model(torch.as_tensor(
                    batch, device=runtime.device, dtype=torch.float32)).detach().cpu().numpy()
            offset = 0
            for c in candidates:
                count = len(c['points'])
                c['score'] = float(scores[offset:offset+count, c['sensor']].min())
                offset += count
        return sorted(candidates, key=lambda c: (c['nominal_terminal'], c['target'] is not None,
            c['margin'] + .03 * np.clip(c['score'], -1., 1.) - .015 * np.max(np.abs(c['view']-stopped))),
            reverse=True)

    def choose(self, blocked, reference, identity, previous_view=None, excluded=()):
        node = self.node
        runtime = node._per_sensor_runtime
        measured = getattr(node, '_activation_measured_state', None)
        if runtime is None or measured is None or measured[1] is None or not reference or not reference.points:
            return None
        start, velocity, stamp = measured
        now = rospy.Time.now().to_sec()
        if not 0. <= now - stamp <= .15:
            return None
        points = np.asarray(blocked, dtype=float).reshape(-1, 3)
        if not len(points) or not np.all(np.isfinite(points)):
            return None
        if self.last is not None and previous_view is None and now - self.last[2] < .65:
            return None
        self.last = None, start.copy(), now
        began = time.perf_counter()
        mapping = node._trajectory_mapping(reference)
        goal = np.asarray([reference.points[-1].positions[j] for j in mapping])
        limits = rospy.get_param(node.task_rejoin_limits_namespace + '/joint_acceleration_limits')
        accel = np.array([limits[name] for name in DEFAULT_JOINT_NAMES])
        if not np.isfinite(np.r_[start, velocity, goal, accel]).all() or np.any(accel <= 0.):
            return None
        lower, upper = runtime.q_min_np, runtime.q_max_np
        stopped = np.clip(start + velocity*np.abs(velocity)/(2.*accel), lower, upper)
        self.generation += 1

        # Use the actual blocked points and a direct and lateral task prefix.
        entries = [(points, None)]
        if previous_view is None:
            delta = goal-stopped
            delta *= .40 / max(.40, np.max(np.abs(delta)))
            side = np.array([.24, 0., -.26, .12, 0., 0., 0.]) * (-1.)**self.generation
            ends = np.clip(stopped+np.stack([delta, delta+side]), lower, upper)
            sweep = self._sweeps(start, stopped, ends, accel, runtime)
            if sweep is not None:
                grid, poses, routes, unknown, occupied = sweep
                for path, route in zip(poses, routes):
                    for pose, cells in zip(path, route):
                        if occupied[cells].any():
                            break
                        missing = cells[unknown[cells]]
                        if len(missing):
                            entries.append((grid[missing], pose))
                            break
            if len(points) > 1:
                entries.extend((point[None, :], None) for point in points[:2])

        proposals = self._propose(runtime, stopped, entries, previous_view, excluded)
        shortlist = []
        for proposal in proposals:
            if all(np.max(np.abs(proposal['view'] - p['view'])) >= .04 for p in shortlist):
                shortlist.append(proposal)
            if len(shortlist) == 4:
                break
        choice = None
        if shortlist:
            sweep = self._sweeps(start, stopped, np.array([p['view'] for p in shortlist]), accel, runtime)
            if sweep is not None:
                _, _, routes, unknown, occupied = sweep
                for proposal, route in zip(shortlist, routes):
                    if all(not (unknown[cells] | occupied[cells]).any() for cells in route):
                        choice = dict(proposal, start=stopped, goal=goal)
                        break
        node._trace_observation('task_route_preview', vbc_identity=identity,
            direct_points=points.tolist(), task_entries=len(entries), proposals=len(proposals),
            optical_terminals=sum(p['nominal_terminal'] for p in proposals), tested_views=len(shortlist),
            selected_points=None if choice is None else choice['points'].tolist(),
            selected_view=None if choice is None else choice['view'].tolist(),
            selected_nominal_terminal=None if choice is None else choice['nominal_terminal'],
            selected_nominal_margin_m=None if choice is None else choice['margin'],
            selected_r1_score=None if choice is None else choice['score'],
            wall_ms=1000. * (time.perf_counter() - began))
        return choice

    def waypoint(self, choice, trajectory, sweep, received):
        result = super().waypoint(choice, trajectory, sweep, received)
        result['per_sensor_hybrid'].update(nominal_terminal=choice['nominal_terminal'],
                                         proposal_kind='optical_view' if choice['nominal_terminal']
                                         else 'exploration_step')
        return result
