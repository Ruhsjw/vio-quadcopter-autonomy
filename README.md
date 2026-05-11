# Autonomous VIO-Based Quadcopter Navigation

A GPS-denied quadrotor autonomy stack integrating visual-inertial odometry, collision-aware motion planning, minimum-jerk trajectory generation, SE(3) control, and local replanning in obstacle-dense simulation environments.

## Overview

This project combines onboard state estimation with planning and control. Instead of using ground-truth pose feedback, the quadrotor estimates its state from IMU and stereo feature observations, then uses that estimate to plan and track collision-free trajectories.

```text
Stereo Features + IMU
        ↓
Visual-Inertial Odometry
        ↓
State Estimation
        ↓
3D Occupancy Grid + A* Planning
        ↓
Minimum-Jerk Trajectory Generation
        ↓
SE(3) Geometric Control
        ↓
Quadrotor Simulation
```

## Repository Structure

```text
vio-quadcopter-autonomy/
├── core_autonomy/          # Baseline GPS-denied autonomy stack
│   ├── code/               # Planner, trajectory generator, controller, VIO interface
│   ├── util/               # Test maps and evaluation scripts
│   └── data_out/           # Example result logs and trajectory plots
├── local_replanning/       # Limited-range local replanning extension
│   ├── code/               # Replanning-enabled planner and controller integration
│   ├── util/               # Local-map utilities and benchmark maps
│   └── data_out/           # Example replanning outputs
├── flightsim/              # Quadrotor simulator and sensor utilities
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Key Features

- Visual-inertial odometry integration for GPS-denied state estimation
- 3D occupancy-grid collision checking with obstacle inflation
- A* graph search for collision-aware path planning
- Minimum-jerk trajectory generation with adaptive time allocation
- SE(3) geometric quadrotor control
- Local replanning with limited sensor range
- Simulation validation across maze, over-under, and window-style environments

## Engineering Challenges

- Balancing trajectory aggressiveness against VIO estimator stability
- Reducing collision risk through occupancy inflation and trajectory validation
- Handling estimator noise during closed-loop trajectory tracking
- Avoiding sharp waypoint transitions that cause tracking error or motor saturation
- Updating local plans without producing unstable zig-zag motion

## Example Results

Example output logs and trajectory plots are included in:

```text
core_autonomy/data_out/
local_replanning/data_out/
```

Recommended visuals for the GitHub README:

1. 3D trajectory through an obstacle map
2. Desired vs. estimated/actual trajectory plot
3. Local replanning visualization
4. Short GIF of the quadrotor navigating through obstacles

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

## Run

Baseline autonomy stack:

```bash
python -m core_autonomy.util.test core_autonomy.code
```

Local replanning stack:

```bash
python -m local_replanning.util.test local_replanning.code
```

## Tech Stack

Python, NumPy, SciPy, Matplotlib, OpenCV, Visual-Inertial Odometry, A* Search, Occupancy Grid Mapping, Trajectory Generation, SE(3) Control, Quadrotor Simulation.
