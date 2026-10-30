#!/usr/bin/env python3
"""Bounded candidate transport. This module has no safety/commit authority.

An offer is acknowledged before deliver is sent. Thus a late first offer cannot
start certification after the sender has timed out. The receiver's first-offer
lease never restarts. Cancellation must be acknowledged before fresh work is
released. All methods run under the owner's state lock; ROS writes run outside.
"""
import copy
import hashlib
import io
import json
import math
import threading
import time
import uuid

from egocentric_arm_planner.msg import CandidateTransfer


def payload_sha256(raw, prefix_context=""):
    normalized = copy.deepcopy(raw)
    normalized.header.seq = 0  # ROS publisher-local; never an identity.
    stream = io.BytesIO()
    normalized.serialize(stream)
    data = stream.getvalue()
    if prefix_context:
        data += b'\x00repair_prefix_context\x00' + prefix_context.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def transfer_message(action, session, sequence=0, boot="", raw=None,
                     digest="", remaining=0.0, prefix_context=""):
    msg = CandidateTransfer(action=action, session=session, sequence=sequence,
                            receiver_boot_id=boot, payload_sha256=digest,
                            remaining_budget_s=remaining,
                            planned_prefix_context=prefix_context)
    if raw is not None:
        msg.trajectory = copy.deepcopy(raw)
    return msg


class WallWorker:
    """ROS time may pause; transport deadlines use a monotonic wall clock."""
    def __init__(self, callback, on_error, period=0.05):
        self.stop = threading.Event()
        def run():
            while not self.stop.wait(period):
                try:
                    callback()
                except Exception as exc:
                    on_error(exc)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


class CandidateSender:
    def __init__(self, receipt_timeout=0.50, pipeline_timeout=0.75):
        if not (math.isfinite(receipt_timeout) and receipt_timeout > 0 and
                math.isfinite(pipeline_timeout) and
                pipeline_timeout >= receipt_timeout):
            raise ValueError("Invalid candidate admission timeouts")
        self.receipt_timeout = receipt_timeout
        self.pipeline_timeout = pipeline_timeout
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.receiver_boot_id = ""
        self.active = None
        self.waiting = None  # Same latest-only bound as the existing pipeline.
        self.waiting_prefix_context = ""
        self.waiting_boot_wall = None
        self.fault = False
        self.replan_count = 0  # One fresh request per authenticated terminal identity.
        self.receipt_count = self.resend_count = self.cancel_count = 0
        self.timeout_count = self.completed_count = self.stale_status_count = 0
        self.last_stage = "idle"
        self.last_ping_wall = -math.inf

    def ping(self, nonce):
        return transfer_message("ping", nonce)

    def pong(self, boot, now):
        if self.receiver_boot_id and boot != self.receiver_boot_id:
            # A restarted receiver lost its duplicate/commit ledger. Fail closed;
            # never replay old work into that new process.
            self.fault = True
            self.last_stage = "receiver_restarted_hold"
            self.waiting = None
            return []
        self.receiver_boot_id = boot
        if self.active is None and self.waiting is not None and not self.fault:
            if self.waiting_boot_wall is not None and now - self.waiting_boot_wall >= self.receipt_timeout:
                self.fault = True
                self.waiting = None
                self.last_stage = "receiver_readiness_timeout_hold"
                return []
            raw = self.waiting
            self.waiting = None
            self.waiting_boot_wall = None
            return self._begin(raw, now, self.waiting_prefix_context)
        return []

    def forward(self, raw, now, prefix_context=""):
        if self.fault:
            return []
        if self.active is not None or not self.receiver_boot_id:
            self.waiting = copy.deepcopy(raw)
            self.waiting_prefix_context = prefix_context
            if not self.receiver_boot_id and self.waiting_boot_wall is None:
                self.waiting_boot_wall = now
            return []
        return self._begin(raw, now, prefix_context)

    def _begin(self, raw, now, prefix_context=""):
        self.sequence += 1
        self.active = dict(sequence=self.sequence, raw=copy.deepcopy(raw),
                           digest=payload_sha256(raw, prefix_context), prefix_context=prefix_context, first_wall=now,
                           deadline=now + self.pipeline_timeout,
                           receipt_deadline=now + self.receipt_timeout,
                           stage="offering", received=False, audit_started=False, resent=False,
                           cancel_wall=None, cancel_resent=False,
                           recover=False)
        self.last_stage = "offering"
        return [("transfer", self._message("offer", now))]

    def _message(self, action, now):
        a = self.active
        return transfer_message(action, self.session, a["sequence"],
                                self.receiver_boot_id, a["raw"], a["digest"],
                                max(0.0, a["deadline"] - now), a["prefix_context"])

    def cancel(self, now, recover=False):
        a = self.active
        if a is None or a["stage"] in ("cancelling", "fault"):
            return []
        a["stage"] = "cancelling"
        a["cancel_wall"] = now
        a["recover"] = recover
        self.cancel_count += 1
        self.last_stage = "cancelling"
        return [("transfer", self._message("cancel", now))]

    def invalidate(self, now):
        self.waiting = None
        self.waiting_boot_wall = None
        return self.cancel(now, recover=False)

    def status(self, fields, now):
        a = self.active
        if (a is None or fields.get("session") != self.session or
                fields.get("sequence") != str(a["sequence"]) or
                fields.get("payload_sha256") != a["digest"] or
                fields.get("receiver_boot_id") != self.receiver_boot_id or
                fields.get("raw_candidate_stamp_ns") !=
                str(a["raw"].header.stamp.to_nsec())):
            self.stale_status_count += 1
            return [], False
        stage = fields.get("admission_stage", "")
        if stage == "offered":
            if a["stage"] == "offering" and now < a["receipt_deadline"]:
                a["stage"] = "delivering"
                self.last_stage = "delivering"
                return [("transfer", self._message("deliver", now))], False
            return [], False
        if stage in ("received", "gcdf", "vbc"):
            if stage in ("gcdf", "vbc"):
                a["audit_started"] = True
            if not a["received"]:
                a["received"] = True
                self.receipt_count += 1
            if a["stage"] not in ("cancelling", "fault"):
                a["stage"] = stage
                self.last_stage = stage
            return [], False
        if stage not in ("terminal", "cancelled", "expired", "rejected"):
            return [], False
        # A terminal receipt follows actual committed publication under the
        # receiver's commit mutex. If commit won the race, never label it cancelled.
        committed = fields.get("committed") == "1"
        recover = (a["recover"] or stage in ("expired", "rejected")) and not committed
        # This is a terminal disposition of the exact received proposal, not
        # an audit failure. Let the planner's finite proposal ledger own its
        # continuation instead of relying on the transport's generic retry.
        try:
            restore = json.loads(a.get("prefix_context", "")).get("geometric_restore")
        except (ValueError, TypeError, AttributeError):
            restore = None
        preaudit = (not self.fault and isinstance(restore, dict) and stage == "expired" and
                    fields.get("reason") == "deadline_expired" and
                    fields.get("audit_started") == "0" and not committed and
                    fields.get("committed") == "0" and
                    fields.get("result") == "timeout" and a["received"] and not a.get("audit_started", False) and
                    bool(a["prefix_context"]))
        try:
            prefix = json.loads(a.get('prefix_context', ''))
        except (ValueError, TypeError):
            prefix = None
        ground_rejected = (not self.fault and stage == 'rejected' and
            fields.get('reason') in ('ground_knot_margin', 'ground_curve_margin',
                                    'ground_interval_unresolved', 'ground_check_budget', 'ground_wall_budget', 'ground_fk_budget', 'ground_state_stale') and
            fields.get('audit_started') == '0' and fields.get('committed') == '0' and
            fields.get('result') == 'rejected' and a['received'] and not a.get('audit_started', False) and
            ((not a.get('prefix_context')) or (isinstance(prefix, dict) and type(prefix.get('schema')) is int and prefix.get('schema') == 1 and
            isinstance(prefix.get('session'), str) and len(prefix['session']) == 32 and
            type(prefix.get('revision')) is int and prefix['revision'] > 0)))
        resource = fields.get('reason') in ('ground_interval_unresolved', 'ground_check_budget', 'ground_wall_budget', 'ground_fk_budget', 'ground_state_stale')
        metric_rejected = (not self.fault and stage == 'rejected' and
            fields.get('reason', '').startswith('metric_') and
            fields.get('audit_started') == '0' and fields.get('committed') == '0' and
            fields.get('result') == 'rejected' and a['received'] and not a.get('audit_started', False) and
            ((not a.get('prefix_context')) or (isinstance(prefix, dict) and type(prefix.get('schema')) is int and prefix.get('schema') == 1 and
            isinstance(prefix.get('session'), str) and len(prefix['session']) == 32 and
            type(prefix.get('revision')) is int and prefix['revision'] > 0)))
        metric_resource = fields.get('reason') != 'metric_cube_distance_within_margin'
        action = (('preaudit_metric_resource' if metric_resource else 'preaudit_metric_rejected') if metric_rejected else
                  ('preaudit_ground_resource' if resource else 'preaudit_ground_rejected') if ground_rejected else 'preaudit_expired')
        disposition = self._message(action, now) if preaudit or ground_rejected or metric_rejected else None
        if disposition is not None:
            disposition.remaining_budget_s = 0.0
        self.completed_count += 1
        self.last_stage = stage
        self.active = None
        actions = []
        if not self.fault and self.waiting is not None:
            raw = self.waiting
            self.waiting = None
            actions += self._begin(raw, now, self.waiting_prefix_context)
        elif disposition is not None:
            actions.append(("preaudit_terminal", disposition))
        elif recover and not self.fault:
            # Clearing active above consumes this exact terminal identity.
            # Duplicate or late receipts cannot request another plan; a new
            # candidate may independently fail and must retain recovery.
            self.replan_count += 1
            actions.append(("replan", None))
        return actions, True

    def tick(self, now):
        actions = []
        if self.waiting_boot_wall is not None and now - self.waiting_boot_wall >= self.receipt_timeout:
            self.fault = True
            self.waiting = None
            self.last_stage = "receiver_readiness_timeout_hold"
        if not self.receiver_boot_id and now - self.last_ping_wall >= 0.25:
            self.last_ping_wall = now
            actions.append(("transfer", self.ping(self.session)))
        a = self.active
        if a is None or self.fault:
            return actions
        if a["stage"] == "cancelling":
            elapsed = now - a["cancel_wall"]
            if elapsed >= self.receipt_timeout:
                a["stage"] = "fault"
                self.fault = True
                self.waiting = None
                self.last_stage = "cancel_unconfirmed_hold"
            elif elapsed >= self.receipt_timeout / 2 and not a["cancel_resent"]:
                a["cancel_resent"] = True
                actions.append(("transfer", self._message("cancel", now)))
            return actions
        if (now >= a["deadline"] or
                (not a["received"] and now >= a["receipt_deadline"])):
            self.timeout_count += 1
            return actions + self.cancel(now, recover=True)
        if (not a["received"] and not a["resent"] and
                now - a["first_wall"] >= self.receipt_timeout / 2):
            a["resent"] = True
            self.resend_count += 1
            action = "offer" if a["stage"] == "offering" else "deliver"
            actions.append(("transfer", self._message(action, now)))
        return actions
