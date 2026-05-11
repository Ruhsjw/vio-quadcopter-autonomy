import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    def __init__(self, quad_params):
        self.mass            = quad_params['mass']
        self.Ixx             = quad_params['Ixx']
        self.Iyy             = quad_params['Iyy']
        self.Izz             = quad_params['Izz']
        self.arm_length      = quad_params['arm_length']
        self.rotor_speed_min = quad_params['rotor_speed_min']
        self.rotor_speed_max = quad_params['rotor_speed_max']
        self.k_thrust        = quad_params['k_thrust']
        self.k_drag          = quad_params['k_drag']

        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz], dtype=float))
        self.g = 9.81

        # Position gains.  These are intentionally milder than the Project 1
        # gains because Project 3 feeds the controller a noisy VIO estimate.
        self.Kp_pos = np.diag([7.0, 7.0, 10.0])
        self.Kd_pos = np.diag([5.5, 5.5, 7.5])

        # Attitude gains
        self.Kp_att = np.diag([240.0, 240.0, 120.0])
        self.Kd_att = np.diag([18.0, 18.0, 10.0])

        self._v_lpf = None
        self._v_lpf_alpha = 0.75

        # Mixer
        k = self.k_drag / self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1, 1, 1, 1],
                               [0, L, 0, -L],
                               [-L, 0, L, 0],
                               [k, -k, k, -k]], dtype=float)
        self.inv_to_TM = np.linalg.inv(self.to_TM)

    @staticmethod
    def _normalize(v, eps=1e-9):
        n = np.linalg.norm(v)
        if n < eps:
            return v * 0.0, 0.0
        return v / n, n

    @staticmethod
    def _vee(S):
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    @staticmethod
    def _soft_limit_xy(vec_xy, amax):
        n = float(np.linalg.norm(vec_xy))
        if n <= amax or n < 1e-12:
            return vec_xy
        return vec_xy * (amax / n)

    def update(self, t, state, flat_output):
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0.0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        x = np.asarray(state['x'], dtype=float).ravel()
        v = np.asarray(state['v'], dtype=float).ravel()
        q = np.asarray(state['q'], dtype=float).ravel()
        w = np.asarray(state['w'], dtype=float).ravel()
        control_mode = float(np.asarray(flat_output.get('control_mode', 0.0)).ravel()[0])
        if self._v_lpf is None:
            self._v_lpf = v.copy()
        else:
            alpha = self._v_lpf_alpha
            self._v_lpf = alpha * v + (1.0 - alpha) * self._v_lpf
        v = self._v_lpf.copy()

        R = Rotation.from_quat(q).as_matrix()
        b3 = R[:, 2]

        x_ref = np.asarray(flat_output['x'], dtype=float).ravel()
        v_ref = np.asarray(flat_output['x_dot'], dtype=float).ravel()
        a_ref = np.asarray(flat_output['x_ddot'], dtype=float).ravel()
        yaw_ref = float(flat_output['yaw'])

        e_x = x - x_ref
        e_v = v - v_ref
        a_cmd = a_ref - (self.Kd_pos @ e_v) - (self.Kp_pos @ e_x)

        # Smooth limiting (better than componentwise clip)
        a_max_xy = 4.4
        if 0.5 < control_mode < 1.5:
            a_max_xy = 4.68
        elif 1.5 < control_mode < 2.5:
            a_max_xy = 4.55
        a_max_z = 6.0
        a_cmd[0:2] = self._soft_limit_xy(a_cmd[0:2], a_max_xy)
        a_cmd[2] = float(np.clip(a_cmd[2], -a_max_z, a_max_z))

        F_des = self.mass * (a_cmd + np.array([0.0, 0.0, self.g], dtype=float))

        u1 = float(b3 @ F_des)
        u1 = max(0.0, u1)

        # Desired attitude
        b3_des, Fnorm = self._normalize(F_des)
        if Fnorm < 1e-9:
            b3_des = b3.copy()

        a_psi = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        b2_des = np.cross(b3_des, a_psi)
        b2_des, n_b2 = self._normalize(b2_des)
        if n_b2 < 1e-9:
            tmp = np.cross(b3_des, np.array([0.0, 0.0, 1.0]))
            tmp, n_tmp = self._normalize(tmp)
            if n_tmp < 1e-9:
                tmp = np.array([1.0, 0.0, 0.0])
            b2_des = tmp

        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack((b1_des, b2_des, b3_des))
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = self._vee(e_R_mat)

        w_des = np.zeros(3)
        e_w = w - w_des

        M = -(self.Kp_att @ e_R) - (self.Kd_att @ e_w) + np.cross(w, self.inertia @ w)

        # Mix to motors with feasibility shrink
        Fmax = self.k_thrust * (self.rotor_speed_max ** 2)
        k = self.k_drag / self.k_thrust
        L = self.arm_length

        Mxy_max = 1.0 * L * Fmax
        Mz_max = 1.0 * k * Fmax
        M_clip = np.clip(M, [-Mxy_max, -Mxy_max, -Mz_max],
                         [ Mxy_max,  Mxy_max,  Mz_max])

        alpha = 1.0
        F = None

        for _ in range(20):
            TM = np.array([u1, alpha * M_clip[0], alpha * M_clip[1], alpha * M_clip[2]], dtype=float)
            F_try = self.inv_to_TM @ TM
            if (F_try.min() >= 0.0) and (F_try.max() <= Fmax):
                F = F_try
                break
            alpha *= 0.6

        if F is None:
            F = np.ones(4) * (u1 / 4.0)
            alpha = 0.0

        F = np.clip(F, 0.0, Fmax)

        cmd_moment = alpha * M_clip
        cmd_thrust = float(np.sum(F))

        omega_sq = F / self.k_thrust
        omega_sq = np.maximum(omega_sq, 0.0)
        cmd_motor_speeds = np.sqrt(omega_sq)
        cmd_motor_speeds = np.clip(cmd_motor_speeds, 0.0, self.rotor_speed_max)

        return {
            'cmd_motor_speeds': cmd_motor_speeds,
            'cmd_thrust': cmd_thrust,
            'cmd_moment': cmd_moment,
            'cmd_q': cmd_q
        }
