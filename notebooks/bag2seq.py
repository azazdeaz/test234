import rosbag
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Pose, TransformStamped, Transform
from std_msgs.msg import Header
import cv2
import os

from cv_bridge import CvBridge
bridge = CvBridge()
# print()
bag = rosbag.Bag('/catkin_ws/src/bags/03.bag')
got_first_frame = False

folder = "/catkin_ws/src/dataset/bag3imgs/"
os.system("rm -rf {}".format(folder))
os.system("mkdir -p {}".format(folder))
im_num = 1
for m in list(bag.read_messages()):
    # help(m.timestamp)
    # print(m.topic, m.timestamp)
    if m.topic.endswith("/front/image_raw"):
        filename = folder + str(im_num).zfill(6) + ".png"
        im_num += 1
        cv_image = bridge.imgmsg_to_cv2(m.message, desired_encoding='mono8')
        cv2.imwrite(filename, cv_image)
        print("got image", filename)


