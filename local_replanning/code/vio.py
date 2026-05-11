#%% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def skew_symmetric(v):
    """
    Return the 3x3 skew-symmetric matrix associated with a 3x1 vector.
    """
    x, y, z = v.reshape(3)
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    R = q.as_matrix()
    a = R @ (a_m - a_b) + g
    delta_q = Rotation.from_rotvec(((w_m - w_b) * dt).reshape(3))

    new_p = p + v * dt + 0.5 * a * (dt ** 2)
    new_v = v + a * dt
    new_q = q * delta_q

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    R = q.as_matrix()
    acc = a_m - a_b
    omega = (w_m - w_b).reshape(3)

    Fx = np.eye(18)
    Fx[0:3, 3:6] = np.eye(3) * dt
    Fx[3:6, 6:9] = -R @ skew_symmetric(acc) * dt
    Fx[3:6, 9:12] = -R * dt
    Fx[3:6, 15:18] = np.eye(3) * dt
    Fx[6:9, 6:9] = Rotation.from_rotvec(omega * dt).as_matrix().T
    Fx[6:9, 12:15] = -np.eye(3) * dt

    Fi = np.zeros((18, 12))
    Fi[3:6, 0:3] = np.eye(3)
    Fi[6:9, 3:6] = np.eye(3)
    Fi[9:12, 6:9] = np.eye(3)
    Fi[12:15, 9:12] = np.eye(3)

    Qi = np.zeros((12, 12))
    Qi[0:3, 0:3] = (accelerometer_noise_density ** 2) * (dt ** 2) * np.eye(3)
    Qi[3:6, 3:6] = (gyroscope_noise_density ** 2) * (dt ** 2) * np.eye(3)
    Qi[6:9, 6:9] = (accelerometer_random_walk ** 2) * dt * np.eye(3)
    Qi[9:12, 9:12] = (gyroscope_random_walk ** 2) * dt * np.eye(3)

    # return an 18x18 covariance matrix
    return Fx @ error_state_covariance @ Fx.T + Fi @ Qi @ Fi.T


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    R = q.as_matrix()
    Pc = R.T @ (Pw - p)
    Xc, Yc, Zc = Pc.reshape(3)

    if (not np.isfinite(Zc)) or Zc <= 1e-6:
        innovation = np.full((2, 1), np.inf)
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    z_hat = np.array([[Xc / Zc],
                      [Yc / Zc]])
    innovation = uv - z_hat

    if (not np.all(np.isfinite(innovation))) or norm(innovation) > error_threshold:
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    dz_dPc = np.array([[1.0 / Zc, 0.0, -Xc / (Zc ** 2)],
                       [0.0, 1.0 / Zc, -Yc / (Zc ** 2)]])

    H = np.zeros((2, 18))
    H[:, 0:3] = dz_dPc @ (-R.T)
    H[:, 6:9] = dz_dPc @ skew_symmetric(Pc)

    S = H @ error_state_covariance @ H.T + Q
    try:
        K = np.linalg.solve(S.T, H @ error_state_covariance.T).T
    except np.linalg.LinAlgError:
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    delta_x = K @ innovation

    if not np.all(np.isfinite(delta_x)):
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    delta_p = delta_x[0:3]
    delta_v = delta_x[3:6]
    delta_theta = delta_x[6:9]
    delta_a_b = delta_x[9:12]
    delta_w_b = delta_x[12:15]
    delta_g = delta_x[15:18]

    new_p = p + delta_p
    new_v = v + delta_v
    new_q = q * Rotation.from_rotvec(delta_theta.reshape(3))
    new_a_b = a_b + delta_a_b
    new_w_b = w_b + delta_w_b
    new_g = g + delta_g

    I = np.eye(18)
    new_covariance = (I - K @ H) @ error_state_covariance @ (I - K @ H).T + K @ Q @ K.T
    new_covariance = 0.5 * (new_covariance + new_covariance.T)

    return (new_p, new_v, new_q, new_a_b, new_w_b, new_g), new_covariance, innovation
