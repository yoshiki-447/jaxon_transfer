import rospy
from jsk_recognition_msgs.msg import HumanSkeleton
#from functools import partial
import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #距離

profile = pipe.start(config)

def recognition_start(msg, str1, str2):
    
def cb(msg, str1, str2):
    human = msg.skeletons[0]
    limb = human.bone_names
    count = 0
    success = false
    for limb_name in limb:
        if limb_name = str1:
            index = count
            success = true
        count++
    if success:
        #limb_nameの座標を取得

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
    
try:
    # for i in range(0,100):
    #     frames = pipe.wait_for_frames()
    #     for f in frames:
    #         print(f.profile)
    rospy.init_node("roi")
    rospy.Subscriber("/skeleton_with_depth/output/skeletons", HumanSkeleton, cb)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        # frames = pipe.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # depth_image = np.asanyarray(depth_frame.get_data())
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)
        # # 表示
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', depth_colormap)
        
        # # q キー入力で終了
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

        # frameデータを取得
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        shoulder_pos =
        elbow_pos =
        hip_pos =
        joint_coords = [shoulder_pos, elbow_pos, hip_pos]

        kmin, kmax = calculate_kmin_kmax(depth_frame, joint_coords)
        print(f"kmin: {kmin}, kmax: {kmax}")

        depth_image = np.asanyarray(depth_frame.get_data())

        # ROIの定義
        roi_points = np.array(joint_coords, np.int32).reshape((-1, 1, 2))

        # ROI内のピクセルを取り出すためのマスクを作成
        mask = np.zeros_like(depth_image, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)

        # ROI内のピクセルを抽出
        roi_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        # メディアンフィルタでデノイズ
        roi_depth_denoised = cv2.medianBlur(roi_depth, 5)
        # ROIの部分を表示用にスケール調整
        roi_depth_normalized = cv2.normalize(roi_depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        adjusted_depth_image = np.zeros_like(roi_depth_normalized, dtype=np.float32)

        # 各ピクセルの深度値を調整
        for i in range(roi_depth_normalized.shape[0]):
            for j in range(roi_depth_normalized.shape[1]):
                pin = roi_depth_normalized[i, j]  # 各ピクセルの深度値を取得
                adjusted_depth_image[i, j] = adjust_depth_value(pin, kmin, kmax)

        # Otsuの画像分割アルゴリズムで二値化
        _, binary_roi = cv2.threshold(adjusted_depth_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
        # カラーマップでの表示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(adjusted_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap[binary_roi == 255] = (0, 0, 0)  # 二値化したROIを黒にする
        cv2.imshow('Binary ROI', binary_roi)
        cv2.imshow('Adjusted Depth Stream with ROI', depth_colormap)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        rate.sleep()

finally:
    pipe.stop()
    cv2.destroyAllWindows()
