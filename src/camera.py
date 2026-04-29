import numpy as np
from typing import List, Optional

class CameraAgriGS:
    """
    Represents the configuration and state of a single camera, including topics,
    intrinsics, distortion coefficients, pose, and optional depth information.
    """

    def __init__(self,
                 camera_id: str,
                 extrinsic: List[float],
                 intrinsic: List[float],
                 distortion: Optional[List[float]] = None,
                 model: str = "pinhole"):
        """
        Initialize a Camera object with configuration parameters.

        :param camera_id: The camera identifier (e.g. "right_camera").
        :param extrinsic: The 7-element list describing the camera pose in the world frame.
                          Format: [x, y, z, qx, qy, qz, qw].
        :param intrinsic: The 4-element list of camera intrinsics [fx, fy, cx, cy].
        :param distortion: The list of distortion coefficients (e.g. [k1, k2, p1, p2, k3]).
        :param model: The camera model type (e.g. "pinhole", "fisheye").
        :param ros2_init: If True, defers printing camera parameters (useful in ROS2 contexts).
        """
        # Basic assignments
        self.camera_id = camera_id
        self.model = model
        self.extrinsic = extrinsic
        self.intrinsics = intrinsic
        self.distortion = distortion if distortion is not None else None

        # Optional width / height (maybe set later)
        self.width: Optional[int] = None
        self.height: Optional[int] = None

        # Decompose extrinsic: first 3 are translation [x, y, z],
        # last 4 are quaternion [qx, qy, qz, qw]
        translation = extrinsic[:3]
        rotation = extrinsic[3:]
        self.set_camera_pose(translation, rotation)

        # Create the camera matrix K
        fx, fy, cx, cy = self.intrinsics
        self.K = np.array([
            [fx,  0.0, cx ],
            [0.0,  fy, cy ],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def get_camera_matrix(self) -> np.ndarray:
        """
        Returns the 3×3 camera intrinsic matrix.
        :return: The camera intrinsic matrix K.
        """
        return self.K

    def get_distortion_coefficients(self) -> Optional[np.ndarray]:
        """
        Returns the distortion coefficients as a NumPy array, or None if not set.
        :return: The distortion coefficients (or None).
        """
        if self.distortion is not None:
            return np.array(self.distortion, dtype=np.float32)
        return None

    def get_camera_pose(self) -> np.ndarray:
        """
        Returns the 4×4 camera pose matrix.
        :return: The 4×4 homogeneous transformation.
        """
        return np.array(self.extrinsic, dtype=np.float32)

    def set_camera_pose(self, translation: List[float], rotation: List[float]) -> None:
        """
        Set the camera pose internally as a 4×4 transformation matrix.

        :param translation: [x, y, z].
        :param rotation: [qx, qy, qz, qw].
        """
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = rotation
        rotation_matrix = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ],
            dtype=np.float32,
        )

        # Build the 4×4 homogeneous transformation matrix
        transform_3x4 = np.hstack(
            (rotation_matrix, np.array(translation, dtype=np.float32).reshape(3, 1))
        )
        self.extrinsic = np.vstack(
            (transform_3x4, np.array([0, 0, 0, 1], dtype=np.float32))
        )
