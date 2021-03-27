from msg_to_se3 import msg_to_se3
import tf
import itertools
from nav_msgs.msg import Path
import struct
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Pose, TransformStamped, Transform
from std_msgs.msg import Header
import cv2
import gtsam
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from scipy.spatial.transform import Rotation as R
import gtsam.utils.plot as gtsam_plot
from warnings import warn
import rospy
import rosbag
import tf2_ros
from cv_bridge import CvBridge
bridge = CvBridge()

import os
os.system('roscore &')
os.system('rosrun rviz rviz -d ./rviz/gentest.rviz &')


class Viz():
    frame_id = "z_ahead"
    pub_image = rospy.Publisher('render', Image, queue_size=1)
    pub_camera = rospy.Publisher('camera', PoseStamped, queue_size=1)
    pub_path_initial_estimate = rospy.Publisher("initial_estimate", Path, queue_size=1)
    pub_path_ground_truth = rospy.Publisher("truth", Path, queue_size=1)
    pub_path_optimized = rospy.Publisher("optimized", Path, queue_size=1)
    pub_path_imu = rospy.Publisher("imu_path", Path, queue_size=1)

    @classmethod
    def keypoints(cls, img, frame):
        for x, y in [kp.pt for kp in frame.kps]:
            # print("center", center)
            # print(img.shape)
            cv2.circle(img, (int(x), int(y)), radius=1, color=(0,255,128), thickness=1)
        cls.pub_image.publish(bridge.cv2_to_imgmsg(img, encoding="mono8"))

    @classmethod
    def camera_pose(cls, transform):
        cls.pub_camera.publish(transform.to_ros_pose_stamped())

    @staticmethod
    def path(transforms):
        path = Path()
        path.poses = [t.to_ros_pose_stamped() for t in transforms]
        path.header.frame_id = Viz.frame_id
        return path
    @classmethod
    def path_raw(cls, transforms):
        cls.pub_path_initial_estimate.publish(cls.path(transforms))
    @classmethod
    def path_truth(cls, ros_poses):
        path = Path()
        path.poses = ros_poses
        path.header.frame_id = "ground_truth_zero"
        cls.pub_path_ground_truth.publish(path)
    @classmethod
    def path_optimized(cls, ros_poses):
        path = Path()
        path.poses = ros_poses
        path.header.frame_id = Viz.frame_id
        cls.pub_path_optimized.publish(path)
    
    @classmethod
    def path_imu(cls, ros_poses):
        path = Path()
        path.poses = ros_poses
        path.header.frame_id = "map"
        cls.pub_path_imu.publish(path)

class Frame():
    def __init__(self, id, kps, descs):
        self.id = id
        self.kps = kps
        self.descs = descs
    
    def symbol(self):
        return gtsam.symbol('X', self.id)

class Landmark():
    _id = itertools.count(0)

    def __init__(self, frame1, kp_index1, frame2, kp_index2):
        self.symbol = gtsam.symbol('S', next(self._id))
        self.observations = { frame1: kp_index1, frame2: kp_index2 }
    def add_observation(self,frame, kp_index):
        if frame in self.observations:
            if kp_index == self.observations[frame]:
                return
            # warn("Star: tried to add a Frame #{} observation twice with differect KP indices ({}, {})".format(frame.id, self.observations[frame], kp_index))
            del self.observations[frame]
        else:
            self.observations[frame] = kp_index
    def has(self, frame, kp_index):
        return frame in self.observations and self.observations[frame] == kp_index

class Transform():
    def __init__(self, frame1, frame2, R, t):
        assert (frame1 is None and frame2 is None) or (frame1.id < frame2.id)

        self.frame1 = frame1
        self.frame2 = frame2
        self.R = R
        self.t = t

    @staticmethod
    def from_ros_msg(msg):
        h = msg_to_se3(ts)
        return Transform(None, None, h[:3,:3], h[:3,3:])
    
    def to_ros_pose_stamped(self):
        pose = PoseStamped()
        pose.header.frame_id = Viz.frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.t[0][0]
        pose.pose.position.y = self.t[1][0]
        pose.pose.position.z = self.t[2][0]
        quaternion = R.from_matrix(self.R).as_quat()
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose
        
    def __add__(self, transform2):
        assert isinstance(transform2, Transform)
        t = self.t + (self.R @ transform2.t)
        R = transform2.R @ self.R
        return Transform(self.frame1, transform2.frame2, R, t)

    def projection_matrix(self):
        x,y,z = self.t.T[0]
        rot = self.R
        return np.hstack((rot, [[x],[y],[z]]))

class DecodeImageMsg():
    @classmethod
    def push(cls, msg):
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        id = int(msg.header.stamp.to_time() * 10**9)
        DetectAndComputeFrame.push(id, cv_image)

class DetectAndComputeFrame():
    detector = cv2.AKAZE_create()

    @classmethod
    def push(cls, id, img):
        (next_kps, next_descs) = cls.detector.detectAndCompute(img, None)
        frame = Frame(id, next_kps, next_descs)
        Viz.keypoints(img, frame)
        FuseImu.push_new_frame(frame)
        GroundTruth.push_new_frame(frame)
        CollectFrames.push(frame)

class CollectFrames():
    frames = []

    @classmethod
    def push(cls, frame):
        cls.frames.append(frame)
        if len(cls.frames) > 1 and CameraConfig.is_ready():
            EstimateTransform.push(cls.frames[-2], cls.frames[-1], raise_on_match_fail=True, add_to_initial_estimate=True)

            go_back = np.minimum(len(cls.frames)-2, 8)
            extra_count = np.minimum(go_back, 6)
            extras = -(np.random.choice(go_back, extra_count, replace=False) + 3)
            for i in extras:
                EstimateTransform.push(cls.frames[i], cls.frames[-1], raise_on_match_fail=False,add_to_initial_estimate=False)

class EstimateTransform():
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) # TODO replace?
    # focal = 277.1
    # pp = (160.5, 120.5)

    @classmethod
    def push(cls, frame1, frame2, raise_on_match_fail=False, add_to_initial_estimate=False):
        matches = cls.bf.knnMatch(frame1.descs, frame2.descs, k=2)
        
        matches_prev = []
        matches_next = []
        idx_prev = []
        idx_next = []

        matches = list(filter(lambda m: m[0].distance < 0.8*m[1].distance, matches))
        if len(matches) < 40:
            if raise_on_match_fail:
                raise Exception("Failed to match frames. Found {} matches".format(len(matches)))
            return None

        for m,n in matches:
            matches_prev.append(frame1.kps[m.queryIdx].pt)
            matches_next.append(frame2.kps[m.trainIdx].pt)
            idx_prev.append(m.queryIdx)
            idx_next.append(m.trainIdx)

        matches_prev = np.array(matches_prev)
        matches_next = np.array(matches_next)

        focal = CameraConfig.fx
        pp = (CameraConfig.cx, CameraConfig.cy)
        E, mask = cv2.findEssentialMat(matches_prev, matches_next, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # for i, is_good_kp_index in enumerate(mask):
        #     if is_good_kp_index:
        #         CollectLandmarks.push(frame1, idx_prev[i], frame2, idx_next[i])
            
        
        points, R_est, t_est, mask_pose = cv2.recoverPose(E, matches_prev, matches_next, focal=focal, pp=pp)
        
        # x,y,z,scale = self.truth.getPoseAndAbsoluteScale(frame1.id, frame2.id)
        # convert to Z-up
        # z_up_rotation = R.from_rotvec(np.array([np.pi/2,np.pi/2,0])).as_matrix()
        # t_est = z_up_rotation @ t_est
        # R_est = R_est @ z_up_rotation
        # t = self.t + (self.R @ transform2.t)
        # R = transform2.R @ self.R
        # t_est = np.array([t_est[0],t_est[2],t_est[1]])
        # scale = 0.05 # TODO
        # scale = FuseImu.pull_scale_btwn(frame1, frame2)
        scale = GroundTruth.pull_scale_btwn(frame1, frame2)
        t_est *= scale

        transform = Transform(frame1, frame2, R_est, t_est)

        CollectTransforms.push(transform)
        Graph.push_transform(transform)
        if add_to_initial_estimate:
            InitialPathEstimate.push(transform)

class CollectLandmarks():
    landmarks = []

    @classmethod
    def push(cls, frame1, kp_index1, frame2, kp_index2):
        landmark = next((s for s in cls.landmarks if s.has(frame1, kp_index1)), None)
        
        if landmark:
            landmark.add_observation(frame2, kp_index2)
        else:
            landmark = next((s for s in cls.landmarks if s.has(frame2, kp_index2)), None)
            if landmark:
                landmark.add_observation(frame1, kp_index2)
            else:
                cls.landmarks.append(Landmark(frame1, kp_index1, frame2, kp_index2))
        
class CollectTransforms():
    transforms = []

    @classmethod
    def push(cls, transform):
        cls.transforms.append(transform)

        
class FuseImu():
    pub_orientation = rospy.Publisher('orientation', PoseStamped, queue_size=1)
    pub_imu = rospy.Publisher('imu', Imu, queue_size=1)
    vel = np.zeros((3,1))
    translation = np.zeros((3,1))
    # gravity in gazebo
    gravity = np.array([0.0, 0.0, -9.8]).reshape((3,1))
    prev_time = None
    path = []
    frame_translations = {}

    @classmethod
    def push_new_frame(cls, frame):
        cls.frame_translations[frame] = cls.translation.copy()

    @classmethod
    def pull_scale_btwn(cls, frame1, frame2):
        t1 = cls.frame_translations[frame1]
        t2 = cls.frame_translations[frame2]
        return np.linalg.norm(t2-t1)

    @classmethod
    def push(cls, msg: Imu):
        is_first_message = cls.prev_time is None

        curr_time = msg.header.stamp.to_sec()
        if not is_first_message:
            delta = curr_time - cls.prev_time
        cls.prev_time = curr_time

        if is_first_message:
            return
        
        acc = np.array([
            [ msg.linear_acceleration.x ],
            [ msg.linear_acceleration.y ],
            [ msg.linear_acceleration.z ],
        ]) + cls.gravity
        cls.vel += acc * delta
        quat = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ])
        
        nudge = R.from_quat(quat).as_matrix() @ (cls.vel*delta)
        cls.translation += nudge
        # print("delta", delta)
        # print("nudge", nudge)
        # print("acc", acc)
        # print("cls.vel", cls.vel)
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = cls.translation[0][0]
        pose.pose.position.y = cls.translation[1][0]
        pose.pose.position.z = cls.translation[2][0]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        cls.path.append(pose)
        Viz.path_imu(cls.path)
        cls.pub_orientation.publish(pose)
        cls.pub_imu.publish(msg)


class InitialPathEstimate():
    path = []
    global_transform = None

    @classmethod
    def push(cls, transform): 
        if cls.global_transform is None:
            cls.global_transform = transform
        else:
            cls.global_transform = cls.global_transform + transform 
        cls.path.append(cls.global_transform)

        rot = gtsam.Rot3(cls.global_transform.R)
        point = gtsam.Point3(cls.global_transform.t.T[0])
        key = cls.global_transform.frame2.symbol()
        Graph.initial_estimate.insert(key, gtsam.Pose3(rot, point))

        Viz.camera_pose(cls.global_transform)
        Viz.path_raw(cls.path)

class GroundTruth():
    path = []
    last_transform = None
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    frame_translations = {}

    @classmethod
    def push_new_frame(cls, frame):
        cls.frame_translations[frame] = np.array([
            [ cls.last_transform.translation.x ],
            [ cls.last_transform.translation.y ],
            [ cls.last_transform.translation.z ],
        ])

    @classmethod
    def pull_scale_btwn(cls, frame1, frame2):
        t1 = cls.frame_translations[frame1]
        t2 = cls.frame_translations[frame2]
        return np.linalg.norm(t2-t1)

    @classmethod
    def push(cls, ros_transform):
        cls.last_transform = ros_transform
        pose = PoseStamped(pose=Pose(ros_transform.translation, ros_transform.rotation))
        cls.path.append(pose)
        Viz.path_truth(cls.path)
    
    @classmethod
    def trigger_zero_pose(cls):
        # TODO handle when last_transform = None
        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "ground_truth_zero"
        static_transformStamped.child_frame_id = "map"
        static_transformStamped.transform.translation.x = cls.last_transform.translation.x
        static_transformStamped.transform.translation.y = cls.last_transform.translation.y
        static_transformStamped.transform.translation.z = cls.last_transform.translation.z
        static_transformStamped.transform.rotation.x = 0
        static_transformStamped.transform.rotation.y = 0
        static_transformStamped.transform.rotation.z = 1
        static_transformStamped.transform.rotation.w = 0
        cls.broadcaster.sendTransform(static_transformStamped)

class Graph():
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3]*6))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3]*3 + [1.2]*3))

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    @classmethod
    def push_transform(cls, transform):
        rot = gtsam.Rot3(transform.R)
        point = gtsam.Point3(transform.t.T[0])
        key1 = transform.frame1.symbol()
        key2 = transform.frame2.symbol()
        cls.graph.add(gtsam.BetweenFactorPose3(key1, key2, gtsam.Pose3(rot, point), cls.ODOMETRY_NOISE))

    @classmethod
    def dogleg_optimizer(cls):
        params = gtsam.DoglegParams()
        params.setVerbosity('TERMINATION')
        return gtsam.Cal3_S2(cls.graph, cls.initial_estimate, params)

    @classmethod
    def gauss_newton_optimizer(cls):
        parameters = gtsam.GaussNewtonParams()
        # Stop iterating once the change in error between steps is less than this value
        parameters.setRelativeErrorTol(1e-5)
        # Do not perform more than N iteration steps
        parameters.setMaxIterations(1000)
        # Create the optimizer ...
        return gtsam.GaussNewtonOptimizer(cls.graph, cls.initial_estimate, parameters)

    @classmethod
    def lm_optimizer(cls):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        return gtsam.LevenbergMarquardtOptimizer(cls.graph, cls.initial_estimate, params)

    @classmethod
    def pull_result(cls):
        optimizer = cls.lm_optimizer() 
        # ... and optimize
        result = optimizer.optimize()

        def to_ros_pose_stamped(pose3):
            quaternion = pose3.rotation().quaternion()
            pose = PoseStamped()
            pose.header.frame_id = Viz.frame_id
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = pose3.x()
            pose.pose.position.y = pose3.y()
            pose.pose.position.z = pose3.z()
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            return pose

        optimized = np.array([to_ros_pose_stamped(result.atPose3(f.symbol())) for f in CollectFrames.frames])
        Viz.path_optimized(optimized)

        return result


class CameraConfig():
    fx = None
    fy = None
    cx = None
    cy = None
    width = None
    height = None

    @classmethod
    def is_ready(cls):
        return cls.fx is not None

    @classmethod
    def push_info(cls, msg: CameraInfo):
        cls.fx = msg.K[0]
        cls.fy = msg.K[5]
        cls.cx = msg.K[3]
        cls.cy = msg.K[6]
        cls.width = msg.width
        cls.height = msg.height

        
bag = rosbag.Bag('/bags/03.bag')
got_first_frame = False
for m in list(bag.read_messages())[:200]:
    # help(m.timestamp)
    # print(m.topic, m.timestamp)
    if m.topic.endswith("/front/image_raw"):
        DecodeImageMsg.push(m.message)
        if not got_first_frame:
            got_first_frame = True
            GroundTruth.trigger_zero_pose()
        # rospy.sleep(1.)
    elif m.topic.endswith("/pose_static"):
        t = next(t.transform for t in m.message.transforms if t.child_frame_id == "ugv")
        GroundTruth.push(t)
    elif m.topic.endswith("/imu/data"):
        FuseImu.push(m.message)
    elif m.topic.endswith("/front/camera_info"):
        CameraConfig.push_info(m.message)

Graph.graph.add(gtsam.PriorFactorPose3(CollectFrames.frames[0].symbol(), gtsam.Pose3(), Graph.PRIOR_NOISE))
Graph.initial_estimate.insert(CollectFrames.frames[0].symbol(), gtsam.Pose3())
result = Graph.pull_result()