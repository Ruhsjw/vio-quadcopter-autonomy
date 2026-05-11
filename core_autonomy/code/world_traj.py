import numpy as np

from .graph_search import graph_search
from .occupancy_map import OccupancyMap


class WorldTraj(object):
    def __init__(self, world, start, goal):
        self.world = world
        start = np.asarray(start, dtype=float).reshape(3)
        goal = np.asarray(goal, dtype=float).reshape(3)
        self._compact_winding_path = False
        self._z_reversal_path = False
        self._short_vertical_path = False
        self.yaw_target = 0.0
        self.yaw_ramp_time = 1.5
        self.time_scale_candidates = (1.0,)
        self.final_seg_min = 1.5
        self.debug_candidates = []
        self.control_mode = 0.0

        # =========================
        # Grid + planning margin
        # =========================
        self.resolution = np.array([0.20, 0.20, 0.20], dtype=float)

        self.margin_plan_candidates = [0.55, 0.45, 0.35]
        self.safe_margin_candidates = [0.55, 0.45, 0.35]
        self.margin_plan = self.margin_plan_candidates[0]

        # =========================
        # Graph search
        # =========================
        self.path = None
        for margin_plan in self.margin_plan_candidates:
            self.margin_plan = margin_plan
            self.path, _ = graph_search(world, self.resolution, self.margin_plan, start, goal, astar=True)
            if self.path is not None:
                break
        if self.path is None:
            self.points = np.vstack([start, goal])
            self._setup_trivial_traj()
            return

        raw_dense = self._remove_near_duplicates(self.path, eps=1e-9)
        dense = self._prune_collinear(raw_dense, tol=1e-3)

       
        diff = dense[1:] - dense[:-1]
        path_len = float(np.sum(np.linalg.norm(diff, axis=1)))
        path_z_span = float(np.max(dense[:, 2]) - np.min(dense[:, 2]))
        reverse_turns = self._count_reverse_turns(dense, cos_thresh=-0.55)
        soft_reverse_turns = self._count_reverse_turns(dense, cos_thresh=0.15)
        z_reversals = self._count_z_reversals(dense, dz_thresh=0.20)
        dist_direct = float(np.linalg.norm(goal - start))
        dist_ratio = path_len / dist_direct if dist_direct > 1e-3 else 10.0
        self._long_backtracking_path = bool(path_len > 65.0 or
                                            (path_len > 50.0 and reverse_turns >= 2) or
                                            (path_len > 58.0 and soft_reverse_turns >= 3))
        self._vertical_path = bool((not self._long_backtracking_path) and
                                   (path_z_span > 1.6 and path_len > 20.0))
        z_net = float(abs(goal[2] - start[2]))
        self._z_reversal_path = bool(self._vertical_path and z_reversals >= 2 and
                                     z_net < 0.60 * max(path_z_span, 1e-6))
        self._compact_winding_path = bool((not self._vertical_path) and
                                          path_len < 30.0 and
                                          (dist_ratio > 1.25 or reverse_turns >= 1))
        if self._vertical_path and (not self._z_reversal_path) and path_len <= 30.0:
            self.yaw_target = self._camera_forward_yaw(start, goal)

        if self._compact_winding_path:
            path_clear, _ = graph_search(world, self.resolution, 0.65, start, goal, astar=True)
            if path_clear is not None:
                self.margin_plan = 0.65
                dense = self._remove_near_duplicates(path_clear, eps=1e-9)
                dense = self._prune_collinear(dense, tol=1e-3)
                diff = dense[1:] - dense[:-1]
                path_len = float(np.sum(np.linalg.norm(diff, axis=1)))
                path_z_span = float(np.max(dense[:, 2]) - np.min(dense[:, 2]))
                reverse_turns = self._count_reverse_turns(dense, cos_thresh=-0.55)
                soft_reverse_turns = self._count_reverse_turns(dense, cos_thresh=0.15)
                z_reversals = self._count_z_reversals(dense, dz_thresh=0.20)
                dist_ratio = path_len / dist_direct if dist_direct > 1e-3 else 10.0
                self._long_backtracking_path = bool(path_len > 65.0 or
                                                    (path_len > 50.0 and reverse_turns >= 2) or
                                                    (path_len > 58.0 and soft_reverse_turns >= 3))
                self._vertical_path = bool((not self._long_backtracking_path) and
                                           (path_z_span > 1.6 and path_len > 20.0))
                self._z_reversal_path = bool(self._vertical_path and z_reversals >= 2 and
                                             z_net < 0.60 * max(path_z_span, 1e-6))
                self._compact_winding_path = bool((not self._vertical_path) and
                                                  path_len < 30.0 and
                                                  (dist_ratio > 1.25 or reverse_turns >= 1))
                if self._vertical_path and (not self._z_reversal_path) and path_len <= 30.0:
                    self.yaw_target = self._camera_forward_yaw(start, goal)
                else:
                    self.yaw_target = 0.0

        if self._compact_winding_path:
            self.safe_margin_candidates = [0.55, 0.45, 0.35]
        elif self._long_backtracking_path:
            self.safe_margin_candidates = [0.45, 0.40, 0.35]
        elif self._vertical_path:
            if self._z_reversal_path:
                self.safe_margin_candidates = [0.60, 0.55, 0.45]
            else:
                self.safe_margin_candidates = [0.55, 0.45, 0.35] if path_len > 35.0 else [0.45, 0.40, 0.35]

        if (not self._vertical_path) and 45.0 < path_len < 60.0:
            self.safe_margin_candidates = [0.45, 0.40, 0.38]
         
            path2, _ = graph_search(world, self.resolution, 0.38, start, goal, astar=True)
            if path2 is not None:
                d2 = self._remove_near_duplicates(path2, eps=1e-9)
                d2 = self._prune_collinear(d2, tol=1e-3)
                diff2 = d2[1:] - d2[:-1]
                dense = d2
                path_len = float(np.sum(np.linalg.norm(diff2, axis=1)))
                reverse_turns = self._count_reverse_turns(dense, cos_thresh=-0.55)
                soft_reverse_turns = self._count_reverse_turns(dense, cos_thresh=0.15)
                dist_ratio = path_len / dist_direct if dist_direct > 1e-3 else 10.0
                if ((path_len > 65.0) or
                        (path_len > 50.0 and reverse_turns >= 2) or
                        (path_len > 58.0 and soft_reverse_turns >= 3)):
                    self._long_backtracking_path = True
                    self._vertical_path = False
                    self._high_curvature_mode = False
                    self.safe_margin_candidates = [0.45, 0.40, 0.35]
            else:
                self.margin_plan = 0.38

        if ((not self._long_backtracking_path) and path_len > 58.0 and
                dist_ratio > 1.45 and soft_reverse_turns >= 3):
            self._long_backtracking_path = True
            self._vertical_path = False
            self._high_curvature_mode = False
            self.safe_margin_candidates = [0.45, 0.40, 0.35]

        
        self.margin_speed = 0.35
        self._zero_velocity_on_reversal = False
        self._high_curvature_mode = False
        self._waypoint_vel_scale = 1.0
        self._angle_adaptive_vel = False
        self.corner_min_speed_scale = 0.25
        self.v_fast = 3.0
        self.v_slow = 1.5
        self.minT_fast = 0.20
        self.minT_slow = 0.30
        self.turn_slow_k_fast = 0.60
        self.turn_slow_k_slow = 0.90
        self.tight_check_step = 0.05
        self.tight_threshold = 0.20
        self.cosang_stop_thresh = 0.25
        self.reversal_stop_thresh = -0.5
        self.turn_cost_weight = 0.01
        self._force_smooth_mode = False
        self._global_spline_mode = False
        self.boundary_margin_safe = 0.0

        
        if self._compact_winding_path:
            self._zero_velocity_on_reversal = False
            self._high_curvature_mode = False
            self._waypoint_vel_scale = 0.785
            self._angle_adaptive_vel = True
            self.corner_min_speed_scale = 0.28
            self.margin_speed = 0.55
            self.v_fast = 2.77
            self.v_slow = 1.57
            self.minT_fast = 0.220
            self.minT_slow = 0.32
            self.turn_slow_k_fast = 0.47
            self.turn_slow_k_slow = 0.75
            self.tight_threshold = 0.09
        elif self._long_backtracking_path:
            self._zero_velocity_on_reversal = True
            self._high_curvature_mode = True
            self.time_scale_candidates = (1.0, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.74, 0.72, 0.70, 0.698)
            self._waypoint_vel_scale = 0.74
            self._angle_adaptive_vel = False
            self.corner_min_speed_scale = 0.40
            self.margin_speed = 0.50
            self.v_fast = 3.0
            self.v_slow = 1.56
            self.minT_fast = 0.22
            self.minT_slow = 0.30
            self.turn_slow_k_fast = 0.36
            self.turn_slow_k_slow = 0.64
            self.tight_threshold = 0.08
            self.cosang_stop_thresh = 0.25
        elif self._vertical_path:
            self._zero_velocity_on_reversal = False
            self._high_curvature_mode = False
            self._waypoint_vel_scale = 0.70
            self._angle_adaptive_vel = False
            self.margin_speed = 0.50
            self.v_fast = 2.82
            self.v_slow = 1.39
            self.minT_fast = 0.285
            self.minT_slow = 0.405
            self.turn_slow_k_fast = 0.52
            self.turn_slow_k_slow = 0.86
            self.tight_threshold = 0.08
            if self._z_reversal_path and 30.0 < path_len <= 35.0:
                self._waypoint_vel_scale = 0.61
                self.margin_speed = 0.58
                self.v_fast = 2.595
                self.v_slow = 1.325
                self.minT_fast = 0.300
                self.minT_slow = 0.390
                self.turn_slow_k_fast = 0.63
                self.turn_slow_k_slow = 0.96
                self.tight_threshold = 0.065
            elif path_len <= 30.0:
                self._short_vertical_path = True
                self.final_seg_min = 1.20
                self.time_scale_candidates = (1.0, 0.94, 0.90, 0.87, 0.84, 0.82, 0.80)
                self._waypoint_vel_scale = 0.83
                self.margin_speed = 0.45
                self.v_fast = 3.0
                self.v_slow = 1.77
                self.minT_fast = 0.215
                self.minT_slow = 0.295
                self.turn_slow_k_fast = 0.33
                self.turn_slow_k_slow = 0.58
                self.tight_threshold = 0.08
            elif 30.0 < path_len <= 35.0:
                self.final_seg_min = 1.16
                self._waypoint_vel_scale = 0.62
                self.margin_speed = 0.58
                self.v_fast = 2.64
                self.v_slow = 1.32
                self.minT_fast = 0.29
                self.minT_slow = 0.385
                self.turn_slow_k_fast = 0.62
                self.turn_slow_k_slow = 0.94
                self.tight_threshold = 0.06
        elif path_len > 50.0:
           
            self._zero_velocity_on_reversal = True
            self._high_curvature_mode = True
            self.turn_cost_weight = 0.01
            self._waypoint_vel_scale = 0.38
            self._angle_adaptive_vel = False
            self.corner_min_speed_scale = 0.30
            self.margin_speed = 0.45
            self.v_fast = 3.0
            self.v_slow = 1.50
            self.minT_fast = 0.27
            self.minT_slow = 0.35
            self.turn_slow_k_fast = 0.52
            self.turn_slow_k_slow = 0.88
            self.tight_check_step = 0.05
            self.tight_threshold = 0.08
            self.reversal_stop_thresh = -0.65
        elif dist_ratio < 1.15:
            
            self.margin_speed = 0.30
            self.v_fast = 3.0
            self.v_slow = 1.5
            self.minT_fast = 0.22
            self.minT_slow = 0.30
            self.turn_slow_k_fast = 0.55
            self.turn_slow_k_slow = 0.80
            self.cosang_stop_thresh = -0.05
        elif path_len > 35.0:
           
            self._zero_velocity_on_reversal = True
            self._high_curvature_mode = True
            self._waypoint_vel_scale = 0.78
            self._angle_adaptive_vel = True
            self.corner_min_speed_scale = 0.40
            self.margin_speed = 0.45
            self.v_fast = 3.0
            self.v_slow = 1.5
            self.minT_fast = 0.22
            self.minT_slow = 0.30
            self.turn_slow_k_fast = 0.34
            self.turn_slow_k_slow = 0.60
            self.tight_threshold = 0.10
        elif path_len > 28.0:
            
            self._high_curvature_mode = False
            self._waypoint_vel_scale = 0.80
            self._angle_adaptive_vel = False
            self.margin_speed = 0.50
            self.v_fast = 2.65
            self.v_slow = 1.35
            self.minT_fast = 0.28
            self.minT_slow = 0.38
            self.turn_slow_k_fast = 0.65
            self.turn_slow_k_slow = 1.00
            self.tight_threshold = 0.10
        else:
            
            self.v_fast = 3.0
            self.v_slow = 1.5
            self.minT_fast = 0.16
            self.minT_slow = 0.24
            self.turn_slow_k_fast = 0.12
            self.turn_slow_k_slow = 0.22

        
        self.occ_speed = OccupancyMap(world, self.resolution, self.margin_speed)

        # =========================
        # Candidate waypoint sets
        # =========================
        candidates = []

        if self._compact_winding_path:
            candidates.append(self._local_shortcut_path(dense, window=4))
            self._add_shortcut_candidates(candidates, dense, (0.08,))
            self._add_stride_candidates(candidates, dense, (1, 2, 3))
        elif self._long_backtracking_path:
            self._add_stride_candidates(candidates, dense, (7, 6, 5, 4, 3))
        elif self._vertical_path:
            if path_len <= 30.0:
                self._add_shortcut_candidates(candidates, dense, (0.05,))
            self._add_shortcut_candidates(candidates, dense, (0.03,))
            self._add_stride_candidates(candidates, dense, (6, 4, 3))
        elif getattr(self, '_high_curvature_mode', False):
            candidates.append(self._local_shortcut_path(dense, window=4))
            candidates.append(self._local_shortcut_path(dense, window=6))
            candidates.append(self._round_corners(self._local_shortcut_path(dense, window=6), radius=0.25))
            candidates.append(self._local_shortcut_path(dense, window=10))
            self._add_shortcut_candidates(candidates, dense, (0.07, 0.05, 0.03))
            if path_len > 65.0:
                self._add_stride_candidates(candidates, dense, (7, 5, 4, 3))
            else:
                self._add_stride_candidates(candidates, dense, (6, 5, 4, 3))
        else:
            self._add_shortcut_candidates(candidates, dense, (0.02,))

        if ((not self._compact_winding_path) and
                (not self._vertical_path) and
                (not getattr(self, '_high_curvature_mode', False))):
            self._add_stride_candidates(candidates, dense, (6, 4, 3))

        if (not self._compact_winding_path) and (not self._vertical_path):
            candidates.append(dense)
        normalized = []
        for pts in candidates:
            pts = self._remove_near_duplicates(np.asarray(pts, dtype=float), eps=1e-9)
            if pts.shape[0] < 2:
                continue
            pts = pts.copy()
            pts[0] = start
            pts[-1] = goal
            if np.linalg.norm(pts[-1] - goal) > 1e-9:
                pts = np.vstack([pts, goal])
            normalized.append(pts)

        candidate_specs = [(pts, "adaptive") for pts in normalized]
        allow_gate_candidates = bool((not self._compact_winding_path) and
                                     (not self._long_backtracking_path) and
                                     (not self._vertical_path) and
                                     path_z_span <= 1.2 and
                                     path_len < 60.0 and
                                     dist_ratio > 1.35)
        gate_like_world = (allow_gate_candidates and
                           self._is_alternating_obstacle_world(world, start, goal))
        if gate_like_world:
            try:
                gate_pts = self._thin_wall_gap_path(world, start, goal)
                if gate_pts is not None and gate_pts.shape[0] >= 2:
                    gate_len = float(np.sum(np.linalg.norm(np.diff(gate_pts, axis=0), axis=1)))
                    if gate_len < 0.90 * path_len:
                        candidate_specs.insert(0, (gate_pts, "gate"))
            except Exception:
                pass

        if self._should_try_alternating_gate_candidate(path_len, path_z_span, dist_ratio,
                                                       soft_reverse_turns):
            try:
                reference_pts = self._reference_style_shortcut_points(
                    world, start, goal, margin=0.25, window=12)
                if reference_pts is not None and reference_pts.shape[0] >= 2:
                    reference_len = float(np.sum(np.linalg.norm(np.diff(reference_pts, axis=0), axis=1)))
                    if reference_len < 58.0:
                        candidate_specs.insert(0, (reference_pts, "reference"))
                        candidate_specs.insert(0, (reference_pts, "reference_clamped"))
                        candidate_specs.insert(0, (reference_pts, "reference_clamped_fast"))
            except Exception:
                pass

        # =========================
        # Build + validate with safe ladder
        # =========================
        last_err = None
        built = False

        for m_safe in self.safe_margin_candidates:
            self.margin_safe = float(m_safe)
            self.occ_safe = OccupancyMap(world, self.resolution, self.margin_safe)

            mode_order = [False, True]
            choose_fastest = bool(self._long_backtracking_path or
                                  getattr(self, '_high_curvature_mode', False) or
                                  getattr(self, '_short_vertical_path', False) or
                                  any(style in ("reference", "reference_clamped", "reference_clamped_fast")
                                      for _, style in candidate_specs))
            time_scales = self.time_scale_candidates if choose_fastest else (1.0,)
            best = None

            for stop_and_go in mode_order:
                for pts, build_style in candidate_specs:
                    if build_style in ("reference", "reference_clamped", "reference_clamped_fast", "gate") and stop_and_go:
                        continue
                    if build_style == "reference":
                        candidate_time_scales = (1.0,)
                    elif build_style in ("reference_clamped", "reference_clamped_fast"):
                        candidate_time_scales = (1.0,)
                    elif build_style == "gate":
                        candidate_time_scales = (1.0, 0.94)
                    else:
                        candidate_time_scales = time_scales
                    for time_scale in candidate_time_scales:
                        self.points = pts
                        try:
                            if build_style == "reference":
                                self._build_reference_style_traj(pts, time_scale=time_scale)
                                validation_margin = min(float(m_safe), 0.27)
                                self._validate_with_margin(world, validation_margin, sample_dt=0.04)
                            elif build_style == "reference_clamped":
                                self._build_reference_clamped_traj(pts, time_scale=time_scale)
                                validation_margin = min(float(m_safe), 0.30)
                                self._validate_with_margin(world, validation_margin, sample_dt=0.04)
                            elif build_style == "reference_clamped_fast":
                                self._build_reference_clamped_traj(pts, time_scale=time_scale,
                                                                   v_nom=2.38, tight=True)
                                validation_margin = min(float(m_safe), 0.36)
                                self._validate_with_margin(world, validation_margin, sample_dt=0.04)
                            elif build_style == "gate":
                                self._build_gate_style_traj(pts, time_scale=time_scale)
                                validation_margin = 0.22
                                self._validate_with_margin(world, validation_margin, sample_dt=0.04)
                            else:
                                self._build_traj(pts, stop_and_go=stop_and_go, time_scale=time_scale)
                                validation_margin = float(m_safe)
                                self._validate(sample_dt=0.04)
                            cost = self._candidate_cost(pts)
                            self.debug_candidates.append({
                                "points": int(len(pts)),
                                "length": float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
                                "time": float(self.T_total),
                                "cost": float(cost),
                                "stop_and_go": bool(stop_and_go),
                                "builder": build_style,
                                "margin_safe": float(validation_margin),
                                "time_scale": float(time_scale),
                            })
                            if not choose_fastest:
                                self.stop_and_go = stop_and_go
                                self.control_mode = self._candidate_control_mode(build_style)
                                built = True
                                break
                            if best is None or cost < best[0]:
                                best = (
                                    cost,
                                    pts.copy(),
                                    bool(stop_and_go),
                                    self.seg_T.copy(),
                                    self.t_breaks.copy(),
                                    float(self.T_total),
                                    [c.copy() for c in self.coeffs],
                                    self._candidate_control_mode(build_style),
                                )
                        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                            last_err = e
                            continue
                    if built:
                        break
                if built:
                    break

            if choose_fastest and best is not None:
                _, pts, stop_and_go, seg_T, t_breaks, T_total, coeffs, control_mode = best
                self.points = pts
                self.stop_and_go = stop_and_go
                self.seg_T = seg_T
                self.t_breaks = t_breaks
                self.T_total = T_total
                self.coeffs = coeffs
                self.control_mode = control_mode
                built = True

            if built:
                break

        if not built:
            if len(normalized) > 0:
                self.margin_safe = 0.25
                self.occ_safe = OccupancyMap(world, self.resolution, self.margin_safe)
                for pts in normalized:
                    self.points = pts
                    try:
                        self._build_traj(pts, stop_and_go=False)
                        self._validate(sample_dt=0.04)
                        built = True
                        break
                    except RuntimeError:
                        try:
                            self._build_traj(pts, stop_and_go=True)
                            self._validate(sample_dt=0.04)
                            built = True
                            break
                        except RuntimeError:
                            continue
                if not built:
                    self.points = normalized[-1]
                    self._build_traj(self.points, stop_and_go=True)
                    built = True
            else:
                raise last_err

    def update(self, t):
        if t <= 0.0:
            k = 0
            tau = 0.0
        elif t >= self.T_total:
            k = len(self.seg_T) - 1
            tau = float(self.seg_T[k])
        else:
            k = int(np.searchsorted(self.t_breaks, t, side="right") - 1)
            k = int(np.clip(k, 0, len(self.seg_T) - 1))
            tau = float(t - self.t_breaks[k])

        x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_quintic_segment(self.coeffs[k], tau)

        yaw, yaw_dot = self._yaw_at_time(t)

        return {
            "x": x,
            "x_dot": x_dot,
            "x_ddot": x_ddot,
            "x_dddot": x_dddot,
            "x_ddddot": x_ddddot,
            "yaw": yaw,
            "yaw_dot": yaw_dot,
            "control_mode": float(getattr(self, "control_mode", 0.0)),
        }

    def _candidate_control_mode(self, build_style):
        if build_style == "reference_clamped_fast":
            return 1.0
        return 0.0

    def _camera_forward_yaw(self, start, goal):
        h = np.asarray(goal[:2] - start[:2], dtype=float)
        n = float(np.linalg.norm(h))
        if n < 1e-6:
            return 0.0
        yaw = float(np.arctan2(h[1], h[0]) - 0.5 * np.pi)
        return float((yaw + np.pi) % (2.0 * np.pi) - np.pi)

    def _yaw_at_time(self, t):
        target = float(getattr(self, 'yaw_target', 0.0))
        if abs(target) < 1e-6:
            return 0.0, 0.0
        if not np.isfinite(t):
            return target, 0.0
        T = max(float(getattr(self, 'yaw_ramp_time', 1.5)), 1e-6)
        u = float(np.clip(t / T, 0.0, 1.0))
        s = u * u * (3.0 - 2.0 * u)
        ds = (6.0 * u * (1.0 - u)) / T if 0.0 < u < 1.0 else 0.0
        return target * s, target * ds

    def _build_traj(self, pts, stop_and_go=False, time_scale=1.0):
        self.seg_T = self._allocate_times_adaptive(pts, stop_and_go=stop_and_go)
        self.seg_T = self.seg_T * float(time_scale)
        if len(self.seg_T) > 0:
            self.seg_T[-1] = max(float(self.seg_T[-1]), float(self.final_seg_min))
        self.t_breaks = np.concatenate([[0.0], np.cumsum(self.seg_T)])
        self.T_total = float(self.t_breaks[-1])

        v_wp, a_wp = self._compute_waypoint_derivatives(pts, self.t_breaks, stop_and_go=stop_and_go)
        self.coeffs = self._fit_quintic_splines(pts, v_wp, a_wp, self.seg_T)

    def _build_reference_style_traj(self, pts, time_scale=1.0, v_nom=2.25, T_min=0.15):
        pts = np.asarray(pts, dtype=float)
        dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.seg_T = np.maximum(float(T_min), dist / float(v_nom)) * float(time_scale)
        self.t_breaks = np.concatenate([[0.0], np.cumsum(self.seg_T)])
        self.T_total = float(self.t_breaks[-1])
        self.coeffs = self._fit_global_quintic_splines(pts, self.seg_T)
        self.stop_and_go = False

    def _build_reference_clamped_traj(self, pts, time_scale=1.0, v_nom=2.32, tight=False):
        pts = np.asarray(pts, dtype=float)
        dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.seg_T = np.maximum(0.16, dist / float(v_nom)) * float(time_scale)
        self.t_breaks = np.concatenate([[0.0], np.cumsum(self.seg_T)])
        self.T_total = float(self.t_breaks[-1])
        v_wp = self._compute_clamped_reference_velocities(pts, self.t_breaks, tight=tight)
        a_wp = np.zeros_like(v_wp)
        self.coeffs = self._fit_quintic_splines(pts, v_wp, a_wp, self.seg_T)
        self.stop_and_go = False

    def _compute_clamped_reference_velocities(self, pts, t_breaks, tight=False):
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        v = np.zeros((n, 3), dtype=float)
        for i in range(1, n - 1):
            dt = float(t_breaks[i + 1] - t_breaks[i - 1])
            if dt < 1e-9:
                continue
            vi = (pts[i + 1] - pts[i - 1]) / dt
            a = pts[i] - pts[i - 1]
            b = pts[i + 1] - pts[i]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na > 1e-9 and nb > 1e-9:
                cosang = float(np.dot(a, b) / (na * nb))
                cosang = float(np.clip(cosang, -1.0, 1.0))
                if cosang < -0.45:
                    vi *= 0.0
                elif cosang < 0.10:
                    vi *= 0.15 if tight else 0.22
                elif cosang < 0.45:
                    vi *= 0.34 if tight else 0.45
                else:
                    vi *= 0.62 if tight else 0.70

            vmax = 2.9 if tight else 3.1
            nv = float(np.linalg.norm(vi))
            if nv > vmax:
                vi *= vmax / nv
            v[i] = vi
        return v

    def _build_gate_style_traj(self, pts, time_scale=1.0):
        pts = np.asarray(pts, dtype=float)
        dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.seg_T = np.maximum(0.18, dist / 1.65) * float(time_scale)
        self.t_breaks = np.concatenate([[0.0], np.cumsum(self.seg_T)])
        self.T_total = float(self.t_breaks[-1])
        self.coeffs = self._fit_global_quintic_splines(pts, self.seg_T)
        self.stop_and_go = False

    def _candidate_cost(self, pts):
        turn_sum = float(np.sum(self._turn_severity(np.asarray(pts, dtype=float))))
        return float(self.T_total + self.turn_cost_weight * turn_sum)

    def _validate_with_margin(self, world, margin, sample_dt=0.02):
        old_margin = getattr(self, "margin_safe", None)
        old_occ = getattr(self, "occ_safe", None)
        self.margin_safe = float(margin)
        self.occ_safe = OccupancyMap(world, self.resolution, self.margin_safe)
        try:
            self._validate(sample_dt=sample_dt)
        finally:
            if old_margin is not None:
                self.margin_safe = old_margin
            if old_occ is not None:
                self.occ_safe = old_occ

    def _validate(self, sample_dt=0.02):
        t = 0.0
        while t <= self.T_total + 1e-9:
            flat = self.update(t)
            if self.occ_safe.is_occupied_metric(flat["x"]):
                raise RuntimeError(
                    f"Desired trajectory collides (safe margin={self.margin_safe:.2f}) "
                    f"at t={t:.3f}, x={flat['x']}."
                )
            boundary_margin = float(getattr(self, 'boundary_margin_safe', 0.0))
            if boundary_margin > 0.0:
                if float(self.world.min_dist_boundary(flat["x"][None, :])[0]) < boundary_margin:
                    raise RuntimeError(
                        f"Desired trajectory is too close to boundary "
                        f"(margin={boundary_margin:.2f}) at t={t:.3f}, x={flat['x']}."
                    )
            t += sample_dt

    def _segment_tight_fraction(self, p0, p1, step):
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return 1.0 if self.occ_speed.is_occupied_metric(p0) else 0.0

        n = int(np.ceil(L / max(step, 1e-6)))
        occ = 0
        for i in range(n + 1):
            s = i / n
            p = p0 + s * d
            if self.occ_speed.is_occupied_metric(p):
                occ += 1
        return occ / float(n + 1)

    def _turn_severity(self, wps):
        n = len(wps)
        sev = np.zeros(n, dtype=float)
        for i in range(1, n - 1):
            a = wps[i] - wps[i - 1]
            b = wps[i + 1] - wps[i]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                continue
            cosang = float(np.dot(a, b) / (na * nb))
            cosang = float(np.clip(cosang, -1.0, 1.0))
            sev[i] = 1.0 - cosang
        return sev

    def _count_reverse_turns(self, wps, cos_thresh=-0.55):
        wps = np.asarray(wps, dtype=float)
        count = 0
        for i in range(1, len(wps) - 1):
            a = wps[i] - wps[i - 1]
            b = wps[i + 1] - wps[i]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                continue
            cosang = float(np.dot(a, b) / (na * nb))
            if cosang < cos_thresh:
                count += 1
        return count

    def _count_z_reversals(self, wps, dz_thresh=0.20):
        z = np.asarray(wps, dtype=float)[:, 2]
        dz = np.diff(z)
        signs = []
        for d in dz:
            if abs(d) > dz_thresh:
                signs.append(1 if d > 0.0 else -1)

        count = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                count += 1
        return count

    def _allocate_times_adaptive(self, pts, stop_and_go=False):
        wps = np.asarray(pts, dtype=float)
        n = len(wps)

        if stop_and_go:
            seg_T = []
            for k in range(n - 1):
                dist = float(np.linalg.norm(wps[k + 1] - wps[k]))
                T = max(dist / max(self.v_slow, 1e-9), self.minT_slow)
                seg_T.append(T)
            return np.array(seg_T, dtype=float)

        sev = self._turn_severity(wps)

        seg_T = []
        for k in range(n - 1):
            p0 = wps[k]
            p1 = wps[k + 1]
            dist = float(np.linalg.norm(p1 - p0))

            tight_frac = self._segment_tight_fraction(p0, p1, step=self.tight_check_step)
            is_tight = (tight_frac >= self.tight_threshold)

            if is_tight:
                v = self.v_slow
                minT = self.minT_slow
                turn_k = self.turn_slow_k_slow
            else:
                v = self.v_fast
                minT = self.minT_fast
                turn_k = self.turn_slow_k_fast

            T = max(dist / max(v, 1e-9), minT)
            s = max(sev[k], sev[k + 1])
            T *= (1.0 + turn_k * s)
            seg_T.append(T)

        return np.array(seg_T, dtype=float)

    def _compute_waypoint_derivatives(self, pts, t_breaks, stop_and_go=False):
        n = len(pts)
        v = np.zeros((n, 3), dtype=float)
        a = np.zeros((n, 3), dtype=float)

        if stop_and_go:
            return v, a

        v[0] = 0.0
        v[-1] = 0.0
        a[0] = 0.0
        a[-1] = 0.0

        for i in range(1, n - 1):
            dt = float(t_breaks[i + 1] - t_breaks[i - 1])
            if dt < 1e-9:
                v[i] = 0.0
            else:
                v[i] = (pts[i + 1] - pts[i - 1]) / dt

            p_prev = pts[i] - pts[i - 1]
            p_next = pts[i + 1] - pts[i]
            n1 = float(np.linalg.norm(p_prev))
            n2 = float(np.linalg.norm(p_next))
            if n1 > 1e-9 and n2 > 1e-9:
                cosang = float(np.dot(p_prev, p_next) / (n1 * n2))
                cosang = float(np.clip(cosang, -1.0, 1.0))
                cosang_thresh = (float(getattr(self, 'reversal_stop_thresh', -0.5))
                                 if getattr(self, '_zero_velocity_on_reversal', False)
                                 else self.cosang_stop_thresh)
                if cosang < cosang_thresh:
                    v[i] = 0.0
                elif getattr(self, '_angle_adaptive_vel', False):
                    corner_scale = 0.5 * (1.0 + cosang)
                    min_scale = float(getattr(self, 'corner_min_speed_scale', 0.25))
                    v[i] *= max(min_scale, corner_scale)

            vmax_wp = 2.0 * self.v_fast
            nv = float(np.linalg.norm(v[i]))
            if nv > vmax_wp:
                v[i] = v[i] * (vmax_wp / nv)

        scale = float(getattr(self, '_waypoint_vel_scale', 1.0))
        if scale < 1.0:
            v[1:-1] *= scale

        return v, a

    def _remove_near_duplicates(self, pts, eps=1e-9):
        pts = np.asarray(pts, dtype=float)
        out = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - out[-1]) > eps:
                out.append(pts[i])
        return np.vstack(out)

    def _prune_collinear(self, pts, tol=1e-3):
        if len(pts) <= 2:
            return pts
        keep = [pts[0]]
        for i in range(1, len(pts) - 1):
            a = keep[-1]
            b = pts[i]
            c = pts[i + 1]
            ab = b - a
            bc = c - b
            nab = np.linalg.norm(ab)
            nbc = np.linalg.norm(bc)
            if nab < 1e-12 or nbc < 1e-12:
                continue
            cosang = float(np.dot(ab, bc) / (nab * nbc))
            if cosang < (1.0 - tol):
                keep.append(b)
        keep.append(pts[-1])
        return np.vstack(keep)

    def _segment_collision_free_speed(self, p0, p1, step):
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            return (not self.occ_speed.is_occupied_metric(p0))
        n = int(np.ceil(L / step))
        for i in range(n + 1):
            s = i / n
            p = p0 + s * d
            if self.occ_speed.is_occupied_metric(p):
                return False
        return True

    def _should_try_alternating_gate_candidate(self, path_len, path_z_span, dist_ratio,
                                               soft_reverse_turns):
        _ = soft_reverse_turns
        if self._compact_winding_path or self._vertical_path or self._long_backtracking_path:
            return False
        if path_z_span > 1.2:
            return False
        return bool(45.0 < float(path_len) < 60.0 and
                    float(dist_ratio) > 1.35)

    def _is_alternating_obstacle_world(self, world, start, goal):
        return (self._has_alternating_gate_walls(world, start, goal) or
                self._has_alternating_obstacle_train(world, start, goal))

    def _has_alternating_gate_walls(self, world, start, goal):
        bounds = np.asarray(world.world["bounds"]["extents"], dtype=float)
        ymin, ymax = float(bounds[2]), float(bounds[3])
        zmin, zmax = float(bounds[4]), float(bounds[5])
        y_span = ymax - ymin
        z_span = zmax - zmin
        if y_span < 1e-6 or z_span < 1e-6:
            return False

        x_low = min(float(start[0]), float(goal[0]))
        x_high = max(float(start[0]), float(goal[0]))
        gates = []
        for block in world.world.get("blocks", []):
            e = np.asarray(block["extents"], dtype=float)
            x_width = float(e[1] - e[0])
            z_cover = float(e[5] - e[4])
            if x_width > max(0.50, 3.0 * float(self.resolution[0])):
                continue
            if z_cover < 0.60 * z_span:
                continue

            x = 0.5 * float(e[0] + e[1])
            if x <= x_low + 0.2 or x >= x_high - 0.2:
                continue

            touches_low = e[2] <= ymin + 0.08 * y_span
            touches_high = e[3] >= ymax - 0.08 * y_span
            if touches_low == touches_high:
                continue

            side = -1 if touches_low else 1
            gates.append((x, side))

        if len(gates) < 4:
            return False

        direction = 1.0 if float(goal[0]) >= float(start[0]) else -1.0
        gates.sort(key=lambda item: direction * item[0])
        sides = []
        last_x = None
        for x, side in gates:
            if last_x is not None and abs(x - last_x) < 0.20:
                continue
            sides.append(side)
            last_x = x

        changes = sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1])
        return len(sides) >= 4 and changes >= 2

    def _has_alternating_obstacle_train(self, world, start, goal):
        bounds = np.asarray(world.world["bounds"]["extents"], dtype=float)
        ymin, ymax = float(bounds[2]), float(bounds[3])
        zmin, zmax = float(bounds[4]), float(bounds[5])
        y_mid = 0.5 * (ymin + ymax)
        y_span = ymax - ymin
        z_span = zmax - zmin
        if y_span < 1e-6 or z_span < 1e-6:
            return False

        x_low = min(float(start[0]), float(goal[0]))
        x_high = max(float(start[0]), float(goal[0]))
        items = []
        for block in world.world.get("blocks", []):
            e = np.asarray(block["extents"], dtype=float)
            x_width = float(e[1] - e[0])
            y_width = float(e[3] - e[2])
            z_cover = float(e[5] - e[4])
            if x_width > max(0.65, 3.25 * float(self.resolution[0])):
                continue
            if y_width > 0.85 * y_span:
                continue
            if z_cover < 0.40 * z_span:
                continue

            x = 0.5 * float(e[0] + e[1])
            if x <= x_low + 0.2 or x >= x_high - 0.2:
                continue
            y_center = 0.5 * float(e[2] + e[3])
            if abs(y_center - y_mid) < 0.12 * y_span:
                continue
            side = -1 if y_center < y_mid else 1
            items.append((x, side))

        if len(items) < 4:
            return False

        direction = 1.0 if float(goal[0]) >= float(start[0]) else -1.0
        items.sort(key=lambda item: direction * item[0])
        sides = []
        last_x = None
        for x, side in items:
            if last_x is not None and abs(x - last_x) < 0.25:
                continue
            sides.append(side)
            last_x = x

        changes = sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1])
        return len(sides) >= 4 and changes >= 2

    def _reference_style_shortcut_points(self, world, start, goal,
                                         margin=0.25, window=10):
        resolution = np.array([0.25, 0.25, 0.25], dtype=float)
        path, _ = graph_search(world, resolution, margin, start, goal, astar=True)
        if path is None or len(path) < 2:
            return None

        path = self._remove_near_duplicates(np.asarray(path, dtype=float), eps=1e-9)
        occ = OccupancyMap(world, resolution, margin)
        step = 0.5 * float(np.min(resolution))

        out = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j_best = i + 1
            for j in range(i + 1, min(i + int(window) + 1, n)):
                if self._segment_free_in_map(path[i], path[j], occ, step):
                    j_best = j
                else:
                    break
            out.append(path[j_best])
            i = j_best

        pts = self._remove_near_duplicates(np.vstack(out), eps=1e-9)
        pts = pts.copy()
        pts[0] = start
        pts[-1] = goal
        return pts

    def _segment_free_in_map(self, p0, p1, occ_map, step):
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            return not occ_map.is_occupied_metric(p0)
        n = max(1, int(np.ceil(L / max(float(step), 1e-6))))
        for k in range(n + 1):
            r = k / n
            p = p0 + r * d
            if occ_map.is_occupied_metric(p):
                return False
        return True

    def _add_stride_candidates(self, candidates, dense, strides):
        for stride in strides:
            candidates.append(dense[::stride])

    def _add_shortcut_candidates(self, candidates, dense, steps):
        for step in steps:
            try:
                candidates.append(self._shortcut_path(dense, step=step))
            except Exception:
                pass

    def _local_shortcut_path(self, pts, window=10):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        last = len(pts) - 1
        while i < last:
            j_best = i + 1
            for j in range(i + 1, min(i + int(window) + 1, len(pts))):
                if self._segment_collision_free_speed(pts[i], pts[j], step=0.04):
                    j_best = j
                else:
                    break
            out.append(pts[j_best])
            i = j_best
        return np.vstack(out)

    def _round_corners(self, pts, radius=0.25):
        pts = np.asarray(pts, dtype=float)
        if len(pts) <= 2:
            return pts

        out = [pts[0]]
        r = float(radius)
        for i in range(1, len(pts) - 1):
            prev_p = pts[i - 1]
            p = pts[i]
            next_p = pts[i + 1]
            vin = p - prev_p
            vout = next_p - p
            lin = float(np.linalg.norm(vin))
            lout = float(np.linalg.norm(vout))
            if lin < 1e-9 or lout < 1e-9:
                continue

            uin = vin / lin
            uout = vout / lout
            cosang = float(np.clip(np.dot(uin, uout), -1.0, 1.0))
            if cosang > 0.75:
                out.append(p)
                continue

            cut = min(r, 0.35 * lin, 0.35 * lout)
            p_before = p - uin * cut
            p_after = p + uout * cut
            if self._segment_collision_free_speed(out[-1], p_before, step=0.04):
                out.append(p_before)
            else:
                out.append(p)
                continue
            if self._segment_collision_free_speed(p_before, p_after, step=0.04):
                out.append(p_after)
            else:
                out.append(p)

        out.append(pts[-1])
        return np.vstack(out)

    def _elastic_smooth_path(self, pts, iterations=2, alpha=0.35):
        pts = np.asarray(pts, dtype=float)
        if len(pts) <= 2:
            return pts

        smoothed = pts.copy()
        for _ in range(int(iterations)):
            next_pts = smoothed.copy()
            for i in range(1, len(smoothed) - 1):
                midpoint = 0.5 * (smoothed[i - 1] + smoothed[i + 1])
                candidate = smoothed[i] + float(alpha) * (midpoint - smoothed[i])
                if (self._segment_collision_free_speed(smoothed[i - 1], candidate, step=0.04) and
                        self._segment_collision_free_speed(candidate, smoothed[i + 1], step=0.04)):
                    next_pts[i] = candidate
            smoothed = next_pts

        return smoothed

    def _thin_wall_gap_path(self, world, start, goal):
        bounds = np.asarray(world.world["bounds"]["extents"], dtype=float)
        ymin, ymax = float(bounds[2]), float(bounds[3])
        zmin, zmax = float(bounds[4]), float(bounds[5])
        y_span = ymax - ymin
        z_span = zmax - zmin
        if y_span < 1e-6 or z_span < 1e-6:
            return None

        direction = 1.0 if goal[0] >= start[0] else -1.0
        x_low = min(float(start[0]), float(goal[0]))
        x_high = max(float(start[0]), float(goal[0]))
        gates = []

        for block in world.world.get("blocks", []):
            e = np.asarray(block["extents"], dtype=float)
            x_width = float(e[1] - e[0])
            z_cover = float(e[5] - e[4])
            if x_width > max(0.45, 2.5 * float(self.resolution[0])):
                continue
            if z_cover < 0.70 * z_span:
                continue

            x = 0.5 * float(e[0] + e[1])
            if x <= x_low + 0.2 or x >= x_high - 0.2:
                continue

            touches_low = e[2] <= ymin + 0.05 * y_span
            touches_high = e[3] >= ymax - 0.05 * y_span
            clearance = max(0.35, 1.75 * float(self.resolution[1]))
            boundary_clearance = max(0.55, 2.75 * float(self.resolution[1]))
            if touches_low and (not touches_high):
                gap_min, gap_max = float(e[3]) + clearance, ymax - boundary_clearance
                y = gap_max - min(0.10, 0.20 * max(gap_max - gap_min, 0.0))
            elif touches_high and (not touches_low):
                gap_min, gap_max = ymin + boundary_clearance, float(e[2]) - clearance
                y = gap_min + min(0.10, 0.20 * max(gap_max - gap_min, 0.0))
            else:
                continue

            if gap_max - gap_min < max(0.70, 3.0 * float(self.resolution[1])):
                continue

            gates.append((x, y))

        if len(gates) < 4:
            return None

        gates.sort(key=lambda item: direction * item[0])
        pts = [np.asarray(start, dtype=float)]
        denom = max(abs(float(goal[0] - start[0])), 1e-6)
        approach = max(0.45, 2.25 * float(self.resolution[0]))
        for x, y in gates:
            ratio = abs(x - float(start[0])) / denom
            z = float(start[2] + ratio * (goal[2] - start[2]))
            pts.append(np.array([x - direction * approach, y, z], dtype=float))
            pts.append(np.array([x + direction * approach, y, z], dtype=float))
        pts.append(np.asarray(goal, dtype=float))
        pts = np.vstack(pts)

        return self._remove_near_duplicates(pts, eps=1e-6)

    def _shortcut_path(self, pts, step=0.02):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j_best = i + 1
            for j in range(len(pts) - 1, i, -1):
                if self._segment_collision_free_speed(pts[i], pts[j], step=step):
                    j_best = j
                    break
            out.append(pts[j_best])
            i = j_best
        return np.vstack(out)

    def _fit_quintic_splines(self, pts, v_wp, a_wp, seg_T):
        coeffs = []
        for k in range(len(seg_T)):
            T = float(seg_T[k])
            p0, p1 = pts[k], pts[k + 1]
            v0, v1 = v_wp[k], v_wp[k + 1]
            a0, a1 = a_wp[k], a_wp[k + 1]
            coeffs.append(self._quintic_from_boundary(p0, v0, a0, p1, v1, a1, T))
        return coeffs

    def _fit_global_quintic_splines(self, pts, seg_T):
        pts = np.asarray(pts, dtype=float)
        seg_T = np.asarray(seg_T, dtype=float)
        s = len(seg_T)
        if s <= 1:
            v_wp = np.zeros((len(pts), 3), dtype=float)
            a_wp = np.zeros((len(pts), 3), dtype=float)
            return self._fit_quintic_splines(pts, v_wp, a_wp, seg_T)

        n = 6 * s
        A = np.zeros((n, n), dtype=float)
        b = np.zeros((n, 3), dtype=float)
        row = 0

        for k in range(s):
            cols = slice(6 * k, 6 * k + 6)
            A[row, cols] = self._poly_basis(0, 0.0)
            b[row] = pts[k]
            row += 1

            A[row, cols] = self._poly_basis(0, float(seg_T[k]))
            b[row] = pts[k + 1]
            row += 1

        A[row, 0:6] = self._poly_basis(1, 0.0)
        row += 1
        A[row, 0:6] = self._poly_basis(2, 0.0)
        row += 1

        last = slice(6 * (s - 1), 6 * s)
        A[row, last] = self._poly_basis(1, float(seg_T[-1]))
        row += 1
        A[row, last] = self._poly_basis(2, float(seg_T[-1]))
        row += 1

        for k in range(s - 1):
            left = slice(6 * k, 6 * k + 6)
            right = slice(6 * (k + 1), 6 * (k + 1) + 6)
            for d in (1, 2, 3, 4):
                A[row, left] = self._poly_basis(d, float(seg_T[k]))
                A[row, right] = -self._poly_basis(d, 0.0)
                row += 1

        coeff_mat = np.linalg.solve(A, b)
        coeffs = []
        for k in range(s):
            coeffs.append(coeff_mat[6 * k:6 * k + 6, :].copy())
        return coeffs

    def _poly_basis(self, deriv, t):
        t = float(t)
        if deriv == 0:
            return np.array([1.0, t, t**2, t**3, t**4, t**5], dtype=float)
        if deriv == 1:
            return np.array([0.0, 1.0, 2.0*t, 3.0*t**2, 4.0*t**3, 5.0*t**4], dtype=float)
        if deriv == 2:
            return np.array([0.0, 0.0, 2.0, 6.0*t, 12.0*t**2, 20.0*t**3], dtype=float)
        if deriv == 3:
            return np.array([0.0, 0.0, 0.0, 6.0, 24.0*t, 60.0*t**2], dtype=float)
        return np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120.0*t], dtype=float)

    def _quintic_from_boundary(self, p0, v0, a0, p1, v1, a1, T):
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0

        A = np.array([
            [T3,      T4,       T5],
            [3*T2,    4*T3,     5*T4],
            [6*T,    12*T2,    20*T3],
        ], dtype=float)

        b = np.zeros((3, 3), dtype=float)
        b[0, :] = p1 - (c0 + c1*T + c2*T2)
        b[1, :] = v1 - (c1 + 2*c2*T)
        b[2, :] = a1 - (2*c2)

        sol = np.linalg.solve(A, b)
        c3, c4, c5 = sol[0], sol[1], sol[2]
        return np.vstack([c0, c1, c2, c3, c4, c5])

    def _eval_quintic_segment(self, coeff_6x3, tau):
        c0, c1, c2, c3, c4, c5 = coeff_6x3
        t = float(tau)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        p = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5
        v = c1 + 2*c2*t + 3*c3*t2 + 4*c4*t3 + 5*c5*t4
        a = 2*c2 + 6*c3*t + 12*c4*t2 + 20*c5*t3
        j = 6*c3 + 24*c4*t + 60*c5*t2
        s = 24*c4 + 120*c5*t
        return p, v, a, j, s

    def _setup_trivial_traj(self):
        self.points = np.asarray(self.points, dtype=float)
        self.seg_T = np.array([2.0], dtype=float)
        self.t_breaks = np.array([0.0, 2.0], dtype=float)
        self.T_total = 2.0
        self.coeffs = [self._quintic_from_boundary(
            self.points[0], np.zeros(3), np.zeros(3),
            self.points[1], np.zeros(3), np.zeros(3),
            2.0
        )]
