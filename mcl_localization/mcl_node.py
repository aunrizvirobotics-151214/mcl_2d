#!/usr/bin/env python3
"""
Monte Carlo Localization — ROS 2 Humble node
=============================================
Subscribes to:
  /scan          (sensor_msgs/LaserScan)   — laser measurements
  /odom          (nav_msgs/Odometry)       — wheel odometry
  /map           (nav_msgs/OccupancyGrid)  — SLAM Toolbox map

Publishes:
  /mcl/particles (geometry_msgs/PoseArray) — full particle cloud
  /mcl/pose      (geometry_msgs/PoseStamped) — weighted mean estimate
  /mcl/likelihood_map (nav_msgs/OccupancyGrid) — likelihood field for debug
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
from scipy.ndimage import distance_transform_edt

from nav_msgs.msg       import Odometry, OccupancyGrid
from sensor_msgs.msg    import LaserScan
from geometry_msgs.msg  import PoseArray, Pose, PoseStamped, Quaternion, TransformStamped
from std_msgs.msg       import Header

import tf2_ros                          # sudo apt install ros-humble-tf2-ros
import tf_transformations               # sudo apt install ros-humble-tf-transformations



def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def get_sample(std):
    """Irwin-Hall approximation of N(0, std²) — scalar version."""
    tot = 0.0
    for _ in range(12):
        tot += np.random.uniform(-std, std)
    return 0.5 * tot


def get_sample_parallel(std, count):
    """Irwin-Hall approximation — returns (count,) array."""
    tot = np.zeros(count)
    for _ in range(12):
        tot += np.random.uniform(-std, std, size=count)
    return 0.5 * tot


def v2t(pose):
    """[x, y, θ]  →  3×3 homogeneous transform."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([[c, -s, pose[0]],
                     [s,  c, pose[1]],
                     [0,  0,       1]], dtype=float)


def v2t_parallel(poses):
    """(P, 3)  →  (P, 3, 3) batch of homogeneous transforms."""
    P = poses.shape[0]
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    tr = np.zeros((P, 3, 3))
    tr[:, 0, 0] =  c;  tr[:, 0, 1] = -s;  tr[:, 0, 2] = poses[:, 0]
    tr[:, 1, 0] =  s;  tr[:, 1, 1] =  c;  tr[:, 1, 2] = poses[:, 1]
    tr[:, 2, 2] =  1.0
    return tr


def world2map(pose, gridmap, map_res, map_origin):
    """
    Metric world coords → integer pixel (col, row).
    map_origin: [x0, y0] — bottom-left corner of the map in world frame
    (from OccupancyGrid.info.origin).
    """
    max_row = gridmap.shape[0] - 1
    col = int(np.round((pose[0] - map_origin[0]) / map_res))
    row = int(max_row - np.round((pose[1] - map_origin[1]) / map_res))
    return np.array([col, row], dtype=int)


def world2map_batch(poses_xy, gridmap, map_res, map_origin):
    """
    (N, 2) metric world x,y  →  (N, 2) integer [col, row].
    """
    max_row = gridmap.shape[0] - 1
    cols = np.round((poses_xy[:, 0] - map_origin[0]) / map_res).astype(int)
    rows = (max_row - np.round((poses_xy[:, 1] - map_origin[1]) / map_res)).astype(int)
    return np.stack([cols, rows], axis=1)


def ranges2points(ranges, angles):
    """Laser scan  →  (3, B) homogeneous points in robot frame."""
    pts = np.array([ranges * np.cos(angles),
                    ranges * np.sin(angles)])
    return np.vstack([pts, np.ones((1, pts.shape[1]))])


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res, map_origin):
    """
    Beam endpoints → map pixel indices (2, B) for a SINGLE particle.
    """
    r_pts   = ranges2points(r_ranges, r_angles)       # (3, B)
    w_pts   = v2t(w_pose) @ r_pts                     # (3, B)
    xy      = w_pts[:2, :].T                           # (B, 2)
    m_pts   = world2map_batch(xy, gridmap, map_res, map_origin)  # (B, 2)
    return m_pts.T                                     # (2, B)  [col, row]


def ranges2cells_parallel(r_ranges, r_angles, w_poses, gridmap, map_res, map_origin):
    """
    Beam endpoints → map pixel indices (P, 2, B) for ALL particles.
    """
    P = w_poses.shape[0]
    B = r_ranges.shape[0]
    r_pts  = ranges2points(r_ranges, r_angles)         # (3, B)
    w_T    = v2t_parallel(w_poses)                     # (P, 3, 3)
    w_pts  = np.einsum('pij,jb->pib', w_T, r_pts)    # (P, 3, B)
    xy     = w_pts[:, :2, :].transpose(0, 2, 1).reshape(P * B, 2)  # (P*B, 2)
    m_xy   = world2map_batch(xy, gridmap, map_res, map_origin)      # (P*B, 2)
    return m_xy.reshape(P, B, 2).transpose(0, 2, 1)                 # (P, 2, B)


def sample_motion_model(pose, u_t, alpha):
    """Propagate a single particle through the noisy odometry model."""
    rot1, trans, rot2 = u_t
    rot1_hat  = rot1  - get_sample(alpha[0]*abs(rot1)  + alpha[1]*abs(trans))
    trans_hat = trans - get_sample(alpha[2]*abs(trans) + alpha[3]*(abs(rot1)+abs(rot2)))
    rot2_hat  = rot2  - get_sample(alpha[0]*abs(rot2)  + alpha[1]*abs(trans))
    x, y, th  = pose
    return np.array([
        x  + trans_hat * np.cos(th + rot1_hat),
        y  + trans_hat * np.sin(th + rot1_hat),
        wrap_to_pi(th + rot1_hat + rot2_hat)
    ])


def sample_motion_model_parallel(poses, u_t, alpha):
    """Propagate ALL particles (P, 3) through the noisy odometry model."""
    rot1, trans, rot2 = u_t
    P = poses.shape[0]
    rot1_hat  = rot1  - get_sample_parallel(alpha[0]*abs(rot1)  + alpha[1]*abs(trans), P)
    trans_hat = trans - get_sample_parallel(alpha[2]*abs(trans) + alpha[3]*(abs(rot1)+abs(rot2)), P)
    rot2_hat  = rot2  - get_sample_parallel(alpha[0]*abs(rot2)  + alpha[1]*abs(trans), P)
    x_t = poses[:, 0] + trans_hat * np.cos(poses[:, 2] + rot1_hat)
    y_t = poses[:, 1] + trans_hat * np.sin(poses[:, 2] + rot1_hat)
    th_t = wrap_to_pi(poses[:, 2] + rot1_hat + rot2_hat)
    return np.stack([x_t, y_t, th_t], axis=1)


def compute_weights_parallel(poses, z_ranges, z_angles,
                              gridmap, likelihood_map,
                              map_res, map_origin,
                              p_outside=0.05, max_range=10.0):
    """
    Compute importance weights for ALL particles simultaneously.
    Returns (P,) weight array.
    """
    valid = z_ranges <= max_range
    rv, av = z_ranges[valid], z_angles[valid]

    if rv.size == 0:
        return np.ones(poses.shape[0]) / poses.shape[0]

    m_pts  = ranges2cells_parallel(rv, av, poses, gridmap, map_res, map_origin)
    i_idx  = m_pts[:, 1, :]   # (P, B)  row
    j_idx  = m_pts[:, 0, :]   # (P, B)  col

    max_row = gridmap.shape[0] - 1
    max_col = gridmap.shape[1] - 1
    inside  = (i_idx >= 0) & (i_idx <= max_row) & (j_idx >= 0) & (j_idx <= max_col)
    n_out   = np.sum(~inside, axis=1)                # (P,)

    i_s = np.clip(i_idx, 0, max_row)
    j_s = np.clip(j_idx, 0, max_col)
    p_ij = likelihood_map[i_s, j_s].astype(float)   # (P, B)
    p_ij[~inside] = 1.0

    weights = (p_outside ** n_out) * np.prod(p_ij, axis=1)
    w_sum = weights.sum()
    if w_sum < 1e-300:
        return np.ones(poses.shape[0]) / poses.shape[0]
    return weights / w_sum



def low_variance_resample(particles, weights):
    """Returns new (P, 3) particle array sampled proportional to weights."""
    N      = particles.shape[0]
    offset = np.random.uniform(0.0, 1.0 / N)
    pos    = offset + np.arange(N) / N
    cumsum = np.cumsum(weights)
    idx    = np.clip(np.searchsorted(cumsum, pos), 0, N - 1)
    return particles[idx]


def build_likelihood_map(occupancy_grid, sigma=0.5, map_res=0.05):
    """
    Build a likelihood field from an OccupancyGrid image.
    occupancy_grid: 2-D uint8 array (0=free, 255=obstacle, 128=unknown)
    sigma: Gaussian std in metres
    Returns float array in [0, 1] same shape as occupancy_grid.
    """
    obstacle_mask = occupancy_grid > 128          # treat ≥50% as obstacle
    dist_px       = distance_transform_edt(~obstacle_mask)   # pixels
    dist_m        = dist_px * map_res
    likelihood    = np.exp(-0.5 * (dist_m / sigma) ** 2)
    return likelihood.astype(np.float32)



def quat_to_yaw(q):
    return tf_transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w])[2]


def yaw_to_quat(yaw):
    q = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class MCLNode(Node):

    def __init__(self):
        super().__init__('mcl_node')

        # ── Parameters ──────────────────────────────────────────
        self.declare_parameter('num_particles',      3000)
        self.declare_parameter('alpha1',             0.1)   # rot noise from rot
        self.declare_parameter('alpha2',             0.1)   # rot noise from trans
        self.declare_parameter('alpha3',             0.1)   # trans noise from trans
        self.declare_parameter('alpha4',             0.1)   # trans noise from rot
        self.declare_parameter('likelihood_sigma',   0.5)   # metres
        self.declare_parameter('max_laser_range',   10.0)   # metres
        self.declare_parameter('p_outside',          0.05)
        self.declare_parameter('resample_interval',  1)     # resample every N steps
        self.declare_parameter('map_frame',         'map')
        self.declare_parameter('odom_frame',        'odom')
        self.declare_parameter('base_frame',        'base_link')

        p = lambda name: self.get_parameter(name).value
        self.num_particles     = p('num_particles')
        self.alpha             = [p('alpha1'), p('alpha2'), p('alpha3'), p('alpha4')]
        self.sigma             = p('likelihood_sigma')
        self.max_range         = p('max_laser_range')
        self.p_outside         = p('p_outside')
        self.resample_interval = p('resample_interval')
        self.map_frame         = p('map_frame')
        self.odom_frame        = p('odom_frame')

        # ── State ───────────────────────────────────────────────
        self.particles        = None    # (P, 3)  [x, y, θ] in world frame
        self.weights          = None    # (P,)    normalised
        self.gridmap          = None    # (H, W)  uint8 occupancy image
        self.likelihood_map   = None    # (H, W)  float32
        self.map_res          = None    # metres/pixel
        self.map_origin       = None    # [x0, y0] world coords of pixel (0,0)
        self.map_info         = None    # OccupancyGrid.info

        self.prev_odom_pose   = None    # [x, y, θ]  last odometry reading
        self.step_count       = 0

        # ── Stability guards ─────────────────────────────────────
        # 1. Warmup: skip the first N odom msgs while Gazebo physics settles
        self.warmup_count     = 0
        self.WARMUP_N         = 50

        # 2. Velocity gate: don't propagate particles when robot is still
        self.LINEAR_VEL_THR   = 0.01   # m/s
        self.ANGULAR_VEL_THR  = 0.01   # rad/s
        self._last_lin_vel    = 0.0
        self._last_ang_vel    = 0.0

        # 3. Pose-diff gate: skip negligible accumulated displacement
        self.TRANS_THR        = 5e-3   # metres
        self.ROT_THR          = 1.75e-3 # radians

        # ── QoS for map (latched) ───────────────────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscriptions ───────────────────────────────────────
        self.sub_map  = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # ── Publishers ──────────────────────────────────────────
        self.pub_particles = self.create_publisher(
            PoseArray, '/mcl/particles', 10)
        self.pub_pose = self.create_publisher(
            PoseStamped, '/mcl/pose', 10)
        self.pub_likelihood = self.create_publisher(
            OccupancyGrid, '/mcl/likelihood_map', map_qos)

        # ── TF broadcaster — publishes map → odom transform ─────
        # This is what makes 'map' appear in RViz2's Fixed Frame list.
        # The transform encodes the correction between odometry drift
        # and the MCL-estimated pose in the map frame.
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Timer: republish likelihood map every 2 s ────────────
        # The likelihood_map publisher uses Transient Local QoS, but
        # RViz2 may start after the single publish in map_callback and
        # miss it.  A periodic republish guarantees RViz2 always gets it.
        self.create_timer(2.0, self._publish_likelihood_map)

        self.get_logger().info('MCL node started — waiting for /map …')

    def map_callback(self, msg: OccupancyGrid):
        info = msg.info
        H, W = info.height, info.width
        self.map_res    = info.resolution
        self.map_origin = [info.origin.position.x,
                           info.origin.position.y]
        self.map_info   = info

        # OccupancyGrid data: -1=unknown, 0=free, 100=occupied
        raw = np.array(msg.data, dtype=np.int16).reshape(H, W)
        # Convert to uint8 obstacle image matching ex6.py convention:
        #   255 = obstacle, 128 = unknown, 0 = free
        occ = np.zeros((H, W), dtype=np.uint8)
        occ[raw == 100]  = 255   # obstacle
        occ[raw == -1]   = 128   # unknown
        # ROS OccupancyGrid row-0 is the BOTTOM of the world map,
        # but numpy image row-0 is TOP.  Flip so our world2map maths
        # (which also flips Y) stays consistent with ex6.py.
        self.gridmap = np.flipud(occ)

        self.likelihood_map = build_likelihood_map(
            self.gridmap, sigma=self.sigma, map_res=self.map_res)

        # Initialise particles uniformly over free space
        if self.particles is None:
            self._init_particles()
            self.get_logger().info(
                f'Map received ({W}×{H}, res={self.map_res:.3f} m/px). '
                f'Initialised {self.num_particles} particles.')

        # Publish likelihood map once for RViz2 inspection
        self._publish_likelihood_map()

    def _init_particles(self):
        """Scatter particles uniformly in free space cells."""
        H, W = self.gridmap.shape
        free  = np.argwhere(self.gridmap == 0)   # (N_free, 2) [row, col]
        if free.shape[0] == 0:
            self.get_logger().warn('No free cells found — scattering over full map.')
            rows = np.random.randint(0, H, self.num_particles)
            cols = np.random.randint(0, W, self.num_particles)
        else:
            idx  = np.random.choice(free.shape[0], self.num_particles, replace=True)
            rows = free[idx, 0]
            cols = free[idx, 1]

        # Pixel → world metric coords (inverse of world2map)
        max_row = H - 1
        x = cols * self.map_res + self.map_origin[0]
        y = (max_row - rows) * self.map_res + self.map_origin[1]
        th = np.random.uniform(-np.pi, np.pi, self.num_particles)

        self.particles = np.stack([x, y, th], axis=1)   # (P, 3)
        self.weights   = np.ones(self.num_particles) / self.num_particles

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)

        # Store latest velocity for the scan callback gate
        self._last_lin_vel = np.hypot(msg.twist.twist.linear.x,
                                      msg.twist.twist.linear.y)
        self._last_ang_vel = abs(msg.twist.twist.angular.z)

        # Warmup: absorb the junk poses Gazebo emits at t=0 while
        # physics settles, then seed prev_odom_pose from a stable reading
        if self.warmup_count < self.WARMUP_N:
            self.warmup_count  += 1
            self.prev_odom_pose = np.array([pos.x, pos.y, yaw])
            return

        self.prev_odom_pose = np.array([pos.x, pos.y, yaw])

    def scan_callback(self, msg: LaserScan):
        if self.particles is None or self.prev_odom_pose is None:
            return   # waiting for map or first settled odom

        # ── Warmup guard — don't update until odom has settled ──
        if self.warmup_count < self.WARMUP_N:
            return

        # ── Gate 1: velocity — skip if robot is stationary ──────
        # Floating-point noise in a stationary robot's odom still
        # produces small non-zero deltas on every callback, which
        # continuously jitter all particles.  Reading twist directly
        # is the cleanest signal that the robot is truly moving.
        if (self._last_lin_vel < self.LINEAR_VEL_THR and
                self._last_ang_vel < self.ANGULAR_VEL_THR):
            self._publish(msg.header.stamp)
            return

        # ── Extract odometry delta u_t ───────────────────────────
        u_t = self._odom_to_ut()
        if u_t is None:
            return

        rot1, trans, rot2 = u_t

        # ── Gate 2: pose-diff — skip negligible displacement ─────
        # Secondary guard: even with non-zero velocity the accumulated
        # delta since the last update might still be below sensor noise.
        if (abs(trans) < self.TRANS_THR and
                abs(rot1) + abs(rot2) < self.ROT_THR):
            self._publish(msg.header.stamp)
            return

        # ── Build laser arrays ───────────────────────────────────
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, self.max_range + 1.0)

        # ── Motion update ────────────────────────────────────────
        self.particles = sample_motion_model_parallel(
            self.particles, u_t, self.alpha)

        # ── Weight update ────────────────────────────────────────
        self.weights = compute_weights_parallel(
            self.particles, ranges, angles,
            self.gridmap, self.likelihood_map,
            self.map_res, self.map_origin,
            p_outside=self.p_outside,
            max_range=self.max_range)

        # ── Resample ─────────────────────────────────────────────
        self.step_count += 1
        if self.step_count % self.resample_interval == 0:
            self.particles = low_variance_resample(self.particles, self.weights)
            self.weights   = np.ones(self.num_particles) / self.num_particles

        # ── Publish ──────────────────────────────────────────────
        self._publish(msg.header.stamp)

    def _odom_to_ut(self):
        """
        Compute odometry motion parameters [rot1, trans, rot2] from
        the delta between the last published update and the current pose.
        Returns None on the very first call (no previous reference yet).
        """
        if self.prev_odom_pose is None:
            return None

        cur = self.prev_odom_pose

        if not hasattr(self, '_last_used_odom'):
            # First call after warmup — seed the reference and skip
            self._last_used_odom = cur.copy()
            return None

        prev  = self._last_used_odom
        dx    = cur[0] - prev[0]
        dy    = cur[1] - prev[1]
        trans = np.hypot(dx, dy)
        rot1  = wrap_to_pi(np.arctan2(dy, dx) - prev[2])
        rot2  = wrap_to_pi(cur[2] - prev[2] - rot1)

        self._last_used_odom = cur.copy()
        return np.array([rot1, trans, rot2])

    def _publish(self, stamp):
        self._publish_particles(stamp)
        self._publish_pose(stamp)
        self._broadcast_tf(stamp)

    def _publish_particles(self, stamp):
        msg = PoseArray()
        msg.header = Header(frame_id=self.map_frame, stamp=stamp)
        poses = []
        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = 0.0
            pose.orientation = yaw_to_quat(float(p[2]))
            poses.append(pose)
        msg.poses = poses
        self.pub_particles.publish(msg)

    def _publish_pose(self, stamp):
        """Publish weighted mean pose."""
        # Circular mean for angles to handle the ±π wrap
        wx = np.sum(self.weights * np.cos(self.particles[:, 2]))
        wy = np.sum(self.weights * np.sin(self.particles[:, 2]))
        mean_th = np.arctan2(wy, wx)
        mean_x  = float(np.sum(self.weights * self.particles[:, 0]))
        mean_y  = float(np.sum(self.weights * self.particles[:, 1]))

        msg = PoseStamped()
        msg.header = Header(frame_id=self.map_frame, stamp=stamp)
        msg.pose.position.x = mean_x
        msg.pose.position.y = mean_y
        msg.pose.position.z = 0.0
        msg.pose.orientation = yaw_to_quat(mean_th)
        self.pub_pose.publish(msg)

    def _broadcast_tf(self, stamp):
        """
        Broadcast map → odom transform.

        The MCL estimated pose is the robot in the MAP frame.
        The odometry gives the robot in the ODOM frame.
        The transform map → odom corrects for odometry drift:

            T_map_odom = T_map_robot * inv(T_odom_robot)

        where T_map_robot  = MCL weighted-mean pose
              T_odom_robot = latest raw odometry pose
        """
        if self.prev_odom_pose is None:
            return

        # MCL estimated pose (map frame)
        wx  = np.sum(self.weights * np.cos(self.particles[:, 2]))
        wy  = np.sum(self.weights * np.sin(self.particles[:, 2]))
        est_th = np.arctan2(wy, wx)
        est_x  = float(np.sum(self.weights * self.particles[:, 0]))
        est_y  = float(np.sum(self.weights * self.particles[:, 1]))

        # Raw odometry pose (odom frame)
        odom_x, odom_y, odom_th = self.prev_odom_pose

        # T_map_robot as 3×3
        c, s = np.cos(est_th), np.sin(est_th)
        T_map_robot = np.array([[c, -s, est_x],
                                [s,  c, est_y],
                                [0,  0,     1]])

        # T_odom_robot as 3×3
        c2, s2 = np.cos(odom_th), np.sin(odom_th)
        T_odom_robot = np.array([[c2, -s2, odom_x],
                                 [s2,  c2, odom_y],
                                 [0,   0,       1]])

        # T_map_odom = T_map_robot @ inv(T_odom_robot)
        T_odom_robot_inv = np.linalg.inv(T_odom_robot)
        T_map_odom = T_map_robot @ T_odom_robot_inv

        # Extract translation and rotation
        tx  = T_map_odom[0, 2]
        ty  = T_map_odom[1, 2]
        th  = np.arctan2(T_map_odom[1, 0], T_map_odom[0, 0])

        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = self.map_frame    # parent: map
        t.child_frame_id  = self.odom_frame   # child:  odom
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, th)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def _publish_likelihood_map(self):
        if self.likelihood_map is None or self.map_info is None:
            return

        # Scale [0, 1] → [0, 100] — stays within int8 range (max 127)
        # Flip back: ROS OccupancyGrid row-0 = bottom of world,
        # but our internal array has row-0 = top (numpy convention)
        scaled = (self.likelihood_map * 100.0)
        scaled = np.flipud(scaled).astype(np.int8)

        msg = OccupancyGrid()
        msg.header = Header(
            frame_id=self.map_frame,
            stamp=self.get_clock().now().to_msg())
        msg.info = self.map_info
        # OccupancyGrid.data must be a flat list of signed int8 in [-1, 100]
        msg.data = scaled.flatten().tolist()
        self.pub_likelihood.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
