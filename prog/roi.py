import rospy
from jsk_recognition_msgs.msg import HumanSkeletonArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

def calculate_kmin_kmax(depth_frame, joint_coords):
    depths = [depth_frame.get_distance(x, y) for (x, y) in joint_coords]
    kmin = min(depths)
    kmax = max(depths)

    return kmin, kmax

def adjust_depth_value(pin, kmin, kmax):
    if pin < kmin:
        pout = pin
    elif kmin <= pin < kmax:
        pout = (1/3) * pin + (2/3) * kmin
    else:  # pin >= kmax
        pout = pin - (1/3) * kmax + (2/3) * kmin

    return pout

def recognition(msg, str1, str2, skeletons, bone_names, bones, coords):
    if len(skeletons(msg)):
        human = skeletons(msg)[0]
        #print(human)
        limb = bone_names(human)
        count = 0
        success = 0
        for limb_name in limb:
            if limb_name == str1:
                index = count
                success = 1
                count+1
                if success:
                    bones = bones(human)[index]
                    print("{}: {}".format(str2,coords(bones)))
                    return coords(bones)     #limb_nameの座標を取得
                    
                
def cb_skeleton(msg):
    r_shoulder_pos = recognition(msg, "left shoulder->right shoulder","right shoulder", lambda p: p.skeletons, lambda p: p.bone_names, lambda p: p.bones, lambda p: p.end_point)
    r_elbow_pos = recognition(msg, "right shoulder->right elbow","right elbos", lambda p: p.skeletons, lambda p: p.bone_names, lambda p: p.bones, lambda p: p.end_point)
    r_hip_pos = recognition(msg, "left hip->right hip", "right hip",lambda p: p.skeletons, lambda p: p.bone_names, lambda p: p.bones, lambda p: p.end_point)
    l_shoulder_pos = recognition(msg, "left shoulder->right shoulder", "left shoulder",lambda p: p.skeletons, lambda p: p.bone_names, lambda p: p.bones,lambda p: p.start_point)
    l_elbow_pos = recognition(msg, "left shoulder->left elbow","left elbow", lambda p: p.skeletons, lambda p: p.bone_names, lambda p: p.bones,lambda p: p.end_point)
    l_hip_pos = recognition(msg, "left hip->right hip", "left hip",lambda p: p.skeletons, lambda p: p.bone_names, lambda p: p.bones,lambda p: p.start_point)

# def cb_depth(msg):
#     try:
#         # ROSのImageメッセージをOpenCVの画像に変換
#         depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
#         depth_image = np.array(depth_image, dtype=np.float32)
        
#         shoulder_pos =
#         elbow_pos =
#         hip_pos =
#         joint_coords = [shoulder_pos, elbow_pos, hip_pos]
        
#         kmin, kmax = calculate_kmin_kmax(depth_frame, joint_coords)
#         print(f"kmin: {kmin}, kmax: {kmax}")
        
#         # ROIの定義
#         roi_points = np.array(joint_coords, np.int32).reshape((-1, 1, 2))
            
#         # ROI内のピクセルを取り出すためのマスクを作成
#         mask = np.zeros_like(depth_image, dtype=np.uint8)
#         cv2.fillPoly(mask, [roi_points], 255)
        
#         # ROI内のピクセルを抽出
#         roi_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        
#         # メディアンフィルタでデノイズ
#         roi_depth_denoised = cv2.medianBlur(roi_depth, 5)
#         # ROIの部分を表示用にスケール調整
#         roi_depth_normalized = cv2.normalize(roi_depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         adjusted_depth_image = np.zeros_like(roi_depth_normalized, dtype=np.float32)
        
#         # 各ピクセルの深度値を調整
#         for i in range(roi_depth_normalized.shape[0]):
#             for j in range(roi_depth_normalized.shape[1]):
#                 pin = roi_depth_normalized[i, j]  # 各ピクセルの深度値を取得
#                 adjusted_depth_image[i, j] = adjust_depth_value(pin, kmin, kmax)
                
#         # Otsuの画像分割アルゴリズムで二値化
#         _, binary_roi = cv2.threshold(adjusted_depth_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # カラーマップでの表示
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(adjusted_depth_image, alpha=0.03), cv2.COLORMAP_JET)
#         depth_colormap[binary_roi == 255] = (0, 0, 0)  # 二値化したROIを黒にする
#         cv2.imshow('Binary ROI', binary_roi)
#         cv2.imshow('Adjusted Depth Stream with ROI', depth_colormap)
#         cv2.waitKey(1)
#     except CvBridgeError as e:
#         rospy.logerr(f"CvBridge Error: {e}")
        
def main():
    rospy.init_node("example")
    #rospy.Subscriber("/rs_l515/depth/image_rect_raw", Image, cb_depth)
    rospy.Subscriber("/skeleton_with_depth/output/skeletons", HumanSkeletonArray, cb_skeleton)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():


        
        rate.sleep()
    cv2.destroyAllWindows()
                    
if __name__ == '__main__':
    main()
