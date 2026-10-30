# Legacy body-sample backend

These are compatibility assets for explicit `geometry_backend:=samples`,
historical A/B replay, and rollback. The primitive mainline does not load them.

Canonical source assets:

- `config/body_samples.yaml`: historical sphere-sample geometry.
- `scripts/generate_body_samples.py`: sphere-sample generator.
- `scripts/body_sample_marker_publisher.py`: sphere-only visualization.
- `src/body_sample_model.cpp`: optional legacy loader and original C++ API.

The old source paths are relative symbolic links to these files. Edit the
canonical file, never replace a compatibility link with a second independent
copy. Both paths initially retain the exact pre-migration bytes, including
existing user changes. The generator's `--output` is explicit: regeneration
of the canonical YAML is a separate action, not part of cleanup.

The public headers, library name `libbody_sample_model.so`, exported ABI entry
`care_load_body_samples_v1`, and ROS executable names are unchanged. Headers
and the lazy compatibility adapter remain outside this directory because
current shared interfaces use them. Do not move the binary away from the
evaluator: the loader intentionally searches only for its same-directory peer.

Installed layout differs from source: YAML lives in
`share/care_confidence_map/legacy/config`, with the old `config/` alias;
the two executable scripts are installed as real files in `lib/care_confidence_map`
under their original names. Source-only `legacy/src` is not shipped as runtime data.

`body_sample_model_node`, `TrajectoryBodySample`, and primitive grid/probe points
are not all legacy-only despite their names. Shared primitive/compatibility
nodes, frame/point types, and topic names are deliberately retained.
No checkpoint, dataset, or output logs belong in this directory.
