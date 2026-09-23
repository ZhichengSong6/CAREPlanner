# V4-A legacy-bank boundary augmentation

Stage A only. The original NPZ and bank cache are read-only.

Default pilot:
- 256 original train x + 64 original val x
- fresh-holdout x excluded
- up to 4 q0 anchors per supported (x,sensor)
- each q0 is refined with analytic FOV geometry to g_s=0
- regular anchors generate 7 tube samples at 0, +/-0.005, +/-0.01, +/-0.02 rad along the unit analytic normal
- no SLSQP nearest-boundary solve per tube sample
- no training in this job

The tube value target is the signed normal offset only when the anchor is regular, the active FOV face remains unchanged, the sign is correct, and the sample remains inside joint limits. It is local distance supervision, not a global nearest-distance certificate.
