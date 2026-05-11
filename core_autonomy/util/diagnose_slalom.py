import argparse
import contextlib
import importlib
import io
import os
from pathlib import Path

import numpy as np

from flightsim.world import World
from core_autonomy.util.test import test_mission


def _format_metric(value):
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=3, suppress_small=True)
    return str(value)


def _path_length(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _trajectory_summary(traj):
    max_speed = 0.0
    max_acc = 0.0
    max_jerk = 0.0
    total_time = float(getattr(traj, "T_total", 0.0))
    if total_time > 0.0:
        for t in np.linspace(0.0, total_time, max(2, int(np.ceil(total_time / 0.05)) + 1)):
            flat = traj.update(float(t))
            max_speed = max(max_speed, float(np.linalg.norm(flat["x_dot"])))
            max_acc = max(max_acc, float(np.linalg.norm(flat["x_ddot"])))
            max_jerk = max(max_jerk, float(np.linalg.norm(flat["x_dddot"])))

    modes = []
    for attr, name in (
        ("_compact_winding_path", "compact"),
        ("_vertical_path", "vertical"),
        ("_z_reversal_path", "z_reversal"),
        ("_short_vertical_path", "short_vertical"),
        ("_long_backtracking_path", "long_backtracking"),
        ("_high_curvature_mode", "high_curvature"),
        ("_zero_velocity_on_reversal", "zero_on_reversal"),
    ):
        if bool(getattr(traj, attr, False)):
            modes.append(name)

    return {
        "mode": ",".join(modes) if modes else "default",
        "margin_plan": getattr(traj, "margin_plan", None),
        "margin_safe": getattr(traj, "margin_safe", None),
        "dense_points": len(getattr(traj, "path", [])),
        "waypoints": len(getattr(traj, "points", [])),
        "dense_length": _path_length(getattr(traj, "path", np.zeros((0, 3)))),
        "waypoint_length": _path_length(getattr(traj, "points", np.zeros((0, 3)))),
        "trajectory_time": total_time,
        "max_des_speed": max_speed,
        "max_des_acc": max_acc,
        "max_des_jerk": max_jerk,
    }


def _candidate_rows(traj, limit=12):
    rows = list(getattr(traj, "debug_candidates", []))
    rows.sort(key=lambda item: item.get("cost", item.get("time", 0.0)))
    return rows[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Run a single diagnostic slalom-style world with optional VIO noise scaling."
    )
    parser.add_argument(
        "--map",
        default=str(Path(__file__).with_name("diag_slalom.json")),
        help="Path to a world json file. Defaults to proj3/util/diag_slalom.json.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier for VIO IMU and stereo measurement noise. Autograder uses 1.0.",
    )
    parser.add_argument(
        "--target",
        default="proj3.code",
        help="Module containing world_traj.py and se3_control.py.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show simulator and VIO debug output.",
    )
    parser.add_argument(
        "--path-summary",
        action="store_true",
        help="Print the selected trajectory mode and path statistics before simulation.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only build the trajectory and print path statistics; do not simulate.",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Print the best validated waypoint candidates from trajectory construction.",
    )
    args = parser.parse_args()

    os.environ["VIO_NOISE_SCALE"] = str(args.noise_scale)

    world = World.from_file(args.map)
    traj_cls = importlib.import_module(args.target + ".world_traj").WorldTraj
    se3_control_cls = importlib.import_module(args.target + ".se3_control").SE3Control

    if args.path_summary or args.summary_only:
        preview_traj = traj_cls(world, world.world["start"], world.world["goal"])
        print("path summary:")
        for key, value in _trajectory_summary(preview_traj).items():
            if isinstance(value, float):
                print("  {}: {:.3f}".format(key, value))
            else:
                print("  {}: {}".format(key, value))
        if args.candidates:
            print("validated candidates:")
            for row in _candidate_rows(preview_traj):
                print(
                    "  pts={points:3d} len={length:6.2f} T={time:6.2f} "
                    "cost={cost:6.2f} stop={stop_and_go} safe={margin_safe:.2f} "
                    "scale={time_scale:.2f}".format(**row)
                )
        if args.summary_only:
            return

    if args.verbose:
        results, metrics = test_mission(
            traj_cls,
            se3_control_cls,
            world,
            world.world["start"],
            world.world["goal"],
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            results, metrics = test_mission(
                traj_cls,
                se3_control_cls,
                world,
                world.world["start"],
                world.world["goal"],
            )

    print("diagnostic map: {}".format(args.map))
    print("target module: {}".format(args.target))
    print("VIO_NOISE_SCALE: {}".format(args.noise_scale))
    print("samples: {}".format(len(results["time"])))
    for key in (
        "stopped_at_goal",
        "no_collision",
        "flight_time",
        "flight_distance",
        "planning_time",
        "collision_point",
        "sim_exit",
    ):
        print("{}: {}".format(key, _format_metric(metrics[key])))


if __name__ == "__main__":
    main()
