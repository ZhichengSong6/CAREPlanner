# Yiming collision CDF checkpoint

Place the trained collision CDF checkpoint here as:

    model_dict.pt

Default absolute workspace path:

    ~/Project/CAREPlanner/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt

The runtime loader accepts:
- Yiming original `{iteration: state_dict}` checkpoints,
- a raw PyTorch `state_dict`,
- dictionaries containing `state_dict`, `model_state_dict`, or `model`.

If your trained file has another filename, either rename it to
`model_dict.pt` or override the launch argument `checkpoint:=/absolute/path/file.pt`.

Do not commit large trained checkpoints unless you intentionally want them in Git.
