"""Horizon scope only; classification is not a robot release certificate."""


def trajectory_workload(knots, closed_loop=False):
    effective_knots = 21 if closed_loop else knots
    if effective_knots not in (2, 20, 21, 50):
        raise ValueError('unsupported qualification horizon')
    mainline = effective_knots == 21
    role = ('mainline_closed_loop' if closed_loop else
            'mainline_horizon_fixture' if mainline else
            'stress_only' if effective_knots == 50 else 'supplemental_diagnostic')
    return dict(role=role, trajectory_states=effective_knots,
                intervals=effective_knots-1, dt_s=.05,
                horizon_duration_s=(effective_knots-1)*.05,
                matches_mainline_horizon=mainline,
                failure_counts_as_mainline_failure=mainline)
