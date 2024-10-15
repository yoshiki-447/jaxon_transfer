import pyrealsense2 as rs
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from jsk_recognition_msgs.msg import HumanSkeletonArray
from sensor_msgs.msg import Image

bridge = CvBridge()

def cb_skeleton(msg):
    human = msg.skeletons[0]
    #rospy.loginfo(human)
    limb = human.bone_names
    #rospy.loginfo(limb)

def cb_depth(msg):
    try:
        # ROSのImageメッセージをOpenCVの画像に変換
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_image = np.array(depth_image, dtype=np.float32)

        # 深度画像の処理（例: カラーマップの適用）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 表示
        cv2.imshow('Depth Image', depth_colormap)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

def main():
    rospy.init_node("example")
    rospy.Subscriber("/rs_l515/depth/image_rect_raw", Image, cb_depth)
    rospy.Subscriber("/skeleton_with_depth/output/skeletons", HumanSkeletonArray, cb_skeleton)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
   #     cv2.destroyAllWindows()
   #     while True:
   #         # # フレーム待ち
   #         # frames = pipeline.wait_for_frames()
            
   #         # #IR１
   #         # # ir_frame1 = frames.get_infrared_frame(1)
   #         # # ir_image1 = np.asanyarray(ir_frame1.get_data())
           
   #         # # #IR2
   #         # # ir_frame2 = frames.get_infrared_frame(2)
   #         # # ir_image2 = np.asanyarray(ir_frame2.get_data())
           
   #         # # RGB
   #         # color_frame = frames.get_color_frame()
   #         # color_image = np.asanyarray(color_frame.get_data())
           
   #         # 深度
   #         # depth_frame = frames.get_depth_frame()
   #         # depth_image = np.asanyarray(depth_frame.get_data())
           
   #         # # # 2次元データをカラーマップに変換
   #         # # ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
   #         # # ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)

   #         # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)
           
   #         # # イメージの結合
   #         images =  np.hstack((color_image, depth_colormap)) 
           
   #         # # 表示
   #         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
   #         cv2.imshow('RealSense', images)
           
   #         #q キー入力で終了
   #         if cv2.waitKey(1) & 0xFF == ord('q'):
   #             cv2.destroyAllWindows()
   #             break
           
   # finally:
   #     # ストリーミング停止
   #     pipeline.stop()
   
if __name__ == '__main__':
    main()
