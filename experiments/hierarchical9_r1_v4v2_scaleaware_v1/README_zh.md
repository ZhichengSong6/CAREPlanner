# R1 legacy vs V2-50K scale-aware full fresh holdout

This is a full 1963-case frozen-start evaluation, not a pilot.

Everything is identical to the completed R1 vs V2-50K comparison except one model-specific runtime parameter:
- R1: legacy runtime probe unchanged, projection_epsilon_f=0.03.
- V2-50K: projection_epsilon_f is set to root_tolerance_f=0.002.

Why: V2 is trained on continuous signed-distance labels (including V4 tubes at 0, ±0.005, ±0.01, ±0.02), while the inherited 0.03 stopping epsilon came from the old R1 score scale. All projection damping, iteration budgets, root refinement, ascent, joint limits, analytic FOV certification, starts, and model weights are unchanged.

The complete frozen fresh holdout is run directly with no smoke/pilot.
