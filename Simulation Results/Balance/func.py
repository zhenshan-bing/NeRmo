import numpy as np
import math
#def quaternion_to_euler_angle_vectorized(w, x, y, z) -> list:
def quaternion_to_euler_angle_vectorized(quat) -> list:
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    """Standard equation to convert quaternions to Euler angles"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y*y)
    X = np.arctan2(t0, t1)*180/math.pi

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = (np.arcsin(t2)) *180/math.pi

    t3 = +2.0 * (w * z + x * y) 
    t4 = +1.0 - 2.0 * (y*y + z * z)
    Z = (np.arctan2(t3, t4)) *180/math.pi

    # X: roll || Y: pitch || Z: Yaw
    return X, Y, Z

