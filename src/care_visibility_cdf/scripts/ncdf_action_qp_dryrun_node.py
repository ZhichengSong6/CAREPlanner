#!/usr/bin/env python3
"""Stage-II 7-DoF action-space NCDF dry run for CAREPlanner.

This node is deliberately NON-ACTUATING. It mirrors the current short-step
velocity backend, reconstructs the nominal 7-D velocity action, and computes a
one-step NCDF visibility action correction.

State / action
--------------
    q in R^7
    u = qdot in R^7

There are no mobile-base