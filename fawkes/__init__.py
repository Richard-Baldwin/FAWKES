"""FAWKES - Field-Adaptive Wheel-control & Kinetics Execution System.

A universal reinforcement-learning framework for RoboCup SSL robot movement,
built for every team, tailored first for Phoenix Server and the TurtleRabbit
fleet. See README.md (the plan of record) for the architecture and milestones;
this package implements it.

Layers (README section 3):
  trace/    L0  FTF-1, the universal trace format + converters + manifest
  sysid/    L1  wheel-response ensemble + the DR families anchored to real data
  envs/     L2  vectorised domain-randomised environments, cascade-in-the-loop
  policies/ L3  CEM over interpretable gains, RMA two-phase adaptation
  export/   L4  firmware registry profile, residual actor, Phoenix bundle
  evaluate/ L5  gates, blind-transfer suite, reports
  viz/      L5  trajectory pictures (stdlib SVG)
"""

__version__ = "0.1.0"
