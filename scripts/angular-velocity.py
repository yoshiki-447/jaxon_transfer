import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wakeup_velocity():
    df1 = pd.read_csv('../data/2025-01-25_human.csv')
    human_headhigh_y = df1['headhigh/Y']
    human_headhigh_x = df1['headhigh/Z']
    human_headlow_y = df1['headlow/Y']
    human_headlow_x = df1['headlow/Z']
    human_neckhigh_y = df1['neckhigh/Y']
    human_neckhigh_x = df1['neckhigh/Z']
    human_necklow_y = df1['necklow/Y']
    human_necklow_x = df1['necklow/Z']
    # human_time = df1['time']
    # human_time = human_time[850:1300]
    human_time = np.arange(0,7.0,0.01)
    human_headhigh_y = human_headhigh_y[500:1200]
    human_headhigh_x = human_headhigh_x[500:1200]
    human_headlow_y = human_headlow_y[500:1200]
    human_headlow_x = human_headlow_x[500:1200]
    human_neckhigh_y = human_neckhigh_y[500:1200]
    human_neckhigh_x = human_neckhigh_x[500:1200]
    human_necklow_y = human_necklow_y[500:1200]
    human_necklow_x = human_necklow_x[500:1200]

    human_mask = ~(np.isnan(human_headhigh_y) | np.isnan(human_headhigh_x) |np.isnan(human_headlow_y) |np.isnan(human_headlow_x) |np.isnan(human_neckhigh_y) |np.isnan(human_neckhigh_x) |np.isnan(human_necklow_y) |np.isnan(human_necklow_x))
    human_time = human_time[human_mask]
    human_headhigh_y = human_headhigh_y[human_mask]
    human_headhigh_x = human_headhigh_x[human_mask]
    human_headlow_y = human_headlow_y[human_mask]
    human_headlow_x = human_headlow_x[human_mask]
    human_neckhigh_y = human_neckhigh_y[human_mask]
    human_neckhigh_x = human_neckhigh_x[human_mask]
    human_necklow_y = human_necklow_y[human_mask]
    human_necklow_x = human_necklow_x[human_mask]

    human_head_angle = [(w-x) / (y-z) for w,x,y,z in zip(human_headhigh_y, human_headlow_y, human_headhigh_x,human_headlow_x)]
    human_neck_angle = [(w-x) / (y-z) for w,x,y,z in zip(human_neckhigh_y, human_necklow_y, human_neckhigh_x,human_necklow_x)]
    human_angle = np.degrees(np.arctan([(x-y)/(1+x*y) for x,y in zip(human_head_angle, human_neck_angle)]))

    # human_bed_angle = np.degrees(np.arctan(human_neck_angle))
    # human_angle = np.degrees([np.arctan2(x-y,1+x*y) for x,y in zip(human_head_angle, human_neck_angle)])
    # human_angular_velocity = np.diff(np.deg2rad(human_angle)) / np.diff(human_time)
    # human_time_adjust = (human_time[:-1] + human_time[1:]) / 2
    # human_bed_angle = np.mod(np.degrees(np.arctan2(human_neck_angle,1)),360)
    # human_bed_angle_adjust = (human_bed_angle[:-1] + human_bed_angle[1:]) / 2

    #関節角速度
    # human_omega = np.zeros_like(human_angle)
    # human_omega[1:-1] = np.deg2rad(human_angle[2:] - human_angle[:-2]) / (human_time[2:] - human_time[:-2])
    # human_omega[0] = np.deg2rad(human_angle[1] - human_angle[0]) / (human_time[1] - human_time[0])
    # human_omega[-1] = np.deg2rad(human_angle[-1] - human_angle[-2]) / (human_time[-1] - human_time[-2])

    #角速度
    human_head_x = human_headhigh_x - human_headlow_x
    human_head_y = human_headhigh_y - human_headlow_y
    human_head = np.stack([human_head_x,human_head_y],1)
    human_cos =np.inner(human_head,np.array([[1,0]])).flatten() / np.linalg.norm(human_head,ord=2,axis=1)
    human_theta = np.arccos(human_cos)
    human_omega = np.zeros_like(human_head_x)
    human_omega[1:-1] = (human_theta[2:] - human_theta[:-2]) / (human_time[2:] - human_time[:-2])
    human_omega[0] = (human_theta[1] - human_theta[0]) / (human_time[1] - human_time[0])
    human_omega[-1] = (human_theta[-1] - human_theta[-2]) / (human_time[-1] - human_time[-2])
    
    human_std_dev = np.std(human_omega)
    print(human_omega.min(),human_omega.max())
    print(f"human_角速度の標準偏差: {human_std_dev:.4f} rad/s")
    
    df2 = pd.read_csv('../data/2025-01-25_shoulder-daen.csv')
    shoulder_daen_headhigh_y = df2['headhigh/Y']
    shoulder_daen_headhigh_x = df2['headhigh/Z']
    shoulder_daen_headlow_y = df2['headlow/Y']
    shoulder_daen_headlow_x = df2['headlow/Z']
    shoulder_daen_neckhigh_y = df2['neckhigh/Y']
    shoulder_daen_neckhigh_x = df2['neckhigh/Z']
    shoulder_daen_necklow_y = df2['necklow/Y']
    shoulder_daen_necklow_x = df2['necklow/Z']
    # shoulder_daen_time = df1['time']
    # shoulder_daen_time = shoulder_daen_time[850:1300]
    shoulder_daen_time = np.arange(0,15,0.01)
    shoulder_daen_headhigh_y = shoulder_daen_headhigh_y[4000:5500]
    shoulder_daen_headhigh_x = shoulder_daen_headhigh_x[4000:5500]
    shoulder_daen_headlow_y = shoulder_daen_headlow_y[4000:5500]
    shoulder_daen_headlow_x = shoulder_daen_headlow_x[4000:5500]
    shoulder_daen_neckhigh_y = shoulder_daen_neckhigh_y[4000:5500]
    shoulder_daen_neckhigh_x = shoulder_daen_neckhigh_x[4000:5500]
    shoulder_daen_necklow_y = shoulder_daen_necklow_y[4000:5500]
    shoulder_daen_necklow_x = shoulder_daen_necklow_x[4000:5500]

    shoulder_daen_mask = ~(np.isnan(shoulder_daen_headhigh_y) | np.isnan(shoulder_daen_headhigh_x) |np.isnan(shoulder_daen_headlow_y) |np.isnan(shoulder_daen_headlow_x) |np.isnan(shoulder_daen_neckhigh_y) |np.isnan(shoulder_daen_neckhigh_x) |np.isnan(shoulder_daen_necklow_y) |np.isnan(shoulder_daen_necklow_x))
    shoulder_daen_time = shoulder_daen_time[shoulder_daen_mask]
    shoulder_daen_headhigh_y = shoulder_daen_headhigh_y[shoulder_daen_mask]
    shoulder_daen_headhigh_x = shoulder_daen_headhigh_x[shoulder_daen_mask]
    shoulder_daen_headlow_y = shoulder_daen_headlow_y[shoulder_daen_mask]
    shoulder_daen_headlow_x = shoulder_daen_headlow_x[shoulder_daen_mask]
    shoulder_daen_neckhigh_y = shoulder_daen_neckhigh_y[shoulder_daen_mask]
    shoulder_daen_neckhigh_x = shoulder_daen_neckhigh_x[shoulder_daen_mask]
    shoulder_daen_necklow_y = shoulder_daen_necklow_y[shoulder_daen_mask]
    shoulder_daen_necklow_x = shoulder_daen_necklow_x[shoulder_daen_mask]
    
    shoulder_daen_head_angle = [(w-x) / (y-z) for w,x,y,z in zip(shoulder_daen_headhigh_y, shoulder_daen_headlow_y, shoulder_daen_headhigh_x,shoulder_daen_headlow_x)]
    shoulder_daen_neck_angle = [(w-x) / (y-z) for w,x,y,z in zip(shoulder_daen_neckhigh_y, shoulder_daen_necklow_y, shoulder_daen_neckhigh_x,shoulder_daen_necklow_x)]
    shoulder_daen_angle = np.degrees(np.arctan([(x-y)/(1+x*y) for x,y in zip(shoulder_daen_head_angle, shoulder_daen_neck_angle)]))

    # shoulder_daen_omega = np.zeros_like(shoulder_daen_angle)
    # shoulder_daen_omega[1:-1] = np.deg2rad(shoulder_daen_angle[2:] - shoulder_daen_angle[:-2]) / (shoulder_daen_time[2:] - shoulder_daen_time[:-2])
    # shoulder_daen_omega[0] = np.deg2rad(shoulder_daen_angle[1] - shoulder_daen_angle[0]) / (shoulder_daen_time[1] - shoulder_daen_time[0])
    # shoulder_daen_omega[-1] = np.deg2rad(shoulder_daen_angle[-1] - shoulder_daen_angle[-2]) / (shoulder_daen_time[-1] - shoulder_daen_time[-2])
    #角速度
    shoulder_daen_head_x = shoulder_daen_headhigh_x - shoulder_daen_headlow_x
    shoulder_daen_head_y = shoulder_daen_headhigh_y - shoulder_daen_headlow_y
    shoulder_daen_head = np.stack([shoulder_daen_head_x,shoulder_daen_head_y],1)
    shoulder_daen_cos =np.inner(shoulder_daen_head,np.array([[1,0]])).flatten() / np.linalg.norm(shoulder_daen_head,ord=2,axis=1)
    shoulder_daen_theta = np.arccos(shoulder_daen_cos)
    shoulder_daen_omega = np.zeros_like(shoulder_daen_head_x)
    shoulder_daen_omega[1:-1] = (shoulder_daen_theta[2:] - shoulder_daen_theta[:-2]) / (shoulder_daen_time[2:] - shoulder_daen_time[:-2])
    shoulder_daen_omega[0] = (shoulder_daen_theta[1] - shoulder_daen_theta[0]) / (shoulder_daen_time[1] - shoulder_daen_time[0])
    shoulder_daen_omega[-1] = (shoulder_daen_theta[-1] - shoulder_daen_theta[-2]) / (shoulder_daen_time[-1] - shoulder_daen_time[-2])
    shoulder_daen_std_dev = np.std(shoulder_daen_omega)
    print(shoulder_daen_omega.min(),shoulder_daen_omega.max())
    print(f"shoulder_daen_角速度の標準偏差: {shoulder_daen_std_dev:.4f} rad/s")
    
    # shoulder_daen_angular_velocity = np.diff(np.deg2rad(shoulder_daen_angle)) / np.diff(shoulder_daen_time)
    # shoulder_daen_time_adjust = (shoulder_daen_time[:-1] + shoulder_daen_time[1:]) / 2
    # shoulder_daen_bed_angle = np.mod(np.degrees(np.arctan2(shoulder_daen_neck_angle,1)),360)
    # shoulder_daen_bed_angle_adjust = (shoulder_daen_bed_angle[:-1] + shoulder_daen_bed_angle[1:]) / 2
    
    df3 = pd.read_csv('../data/2025-01-25_shoulder-circle.csv')
    shoulder_circle_headhigh_y = df3['headhigh/Y']
    shoulder_circle_headhigh_x = df3['headhigh/Z']
    shoulder_circle_headlow_y = df3['headlow/Y']
    shoulder_circle_headlow_x = df3['headlow/Z']
    shoulder_circle_neckhigh_y = df3['neckhigh/Y']
    shoulder_circle_neckhigh_x = df3['neckhigh/Z']
    shoulder_circle_necklow_y = df3['necklow/Y']
    shoulder_circle_necklow_x = df3['necklow/Z']
    # shoulder_circle_time = df1['time']
    # shoulder_circle_time = shoulder_circle_time[850:1300]
    shoulder_circle_time = np.arange(0,15,0.01)
    shoulder_circle_headhigh_y = shoulder_circle_headhigh_y[4000:5500]
    shoulder_circle_headhigh_x = shoulder_circle_headhigh_x[4000:5500]
    shoulder_circle_headlow_y = shoulder_circle_headlow_y[4000:5500]
    shoulder_circle_headlow_x = shoulder_circle_headlow_x[4000:5500]
    shoulder_circle_neckhigh_y = shoulder_circle_neckhigh_y[4000:5500]
    shoulder_circle_neckhigh_x = shoulder_circle_neckhigh_x[4000:5500]
    shoulder_circle_necklow_y = shoulder_circle_necklow_y[4000:5500]
    shoulder_circle_necklow_x = shoulder_circle_necklow_x[4000:5500]

    shoulder_circle_mask = ~(np.isnan(shoulder_circle_headhigh_y) | np.isnan(shoulder_circle_headhigh_x) |np.isnan(shoulder_circle_headlow_y) |np.isnan(shoulder_circle_headlow_x) |np.isnan(shoulder_circle_neckhigh_y) |np.isnan(shoulder_circle_neckhigh_x) |np.isnan(shoulder_circle_necklow_y) |np.isnan(shoulder_circle_necklow_x))
    shoulder_circle_time = shoulder_circle_time[shoulder_circle_mask]
    shoulder_circle_headhigh_y = shoulder_circle_headhigh_y[shoulder_circle_mask]
    shoulder_circle_headhigh_x = shoulder_circle_headhigh_x[shoulder_circle_mask]
    shoulder_circle_headlow_y = shoulder_circle_headlow_y[shoulder_circle_mask]
    shoulder_circle_headlow_x = shoulder_circle_headlow_x[shoulder_circle_mask]
    shoulder_circle_neckhigh_y = shoulder_circle_neckhigh_y[shoulder_circle_mask]
    shoulder_circle_neckhigh_x = shoulder_circle_neckhigh_x[shoulder_circle_mask]
    shoulder_circle_necklow_y = shoulder_circle_necklow_y[shoulder_circle_mask]
    shoulder_circle_necklow_x = shoulder_circle_necklow_x[shoulder_circle_mask]
    
    shoulder_circle_head_angle = [(w-x) / (y-z) for w,x,y,z in zip(shoulder_circle_headhigh_y, shoulder_circle_headlow_y, shoulder_circle_headhigh_x,shoulder_circle_headlow_x)]
    shoulder_circle_neck_angle = [(w-x) / (y-z) for w,x,y,z in zip(shoulder_circle_neckhigh_y, shoulder_circle_necklow_y, shoulder_circle_neckhigh_x,shoulder_circle_necklow_x)]
    shoulder_circle_angle = np.degrees(np.arctan([(x-y)/(1+x*y) for x,y in zip(shoulder_circle_head_angle, shoulder_circle_neck_angle)]))

    # shoulder_circle_omega = np.zeros_like(shoulder_circle_angle)
    # shoulder_circle_omega[1:-1] = np.deg2rad(shoulder_circle_angle[2:] - shoulder_circle_angle[:-2]) / (shoulder_circle_time[2:] - shoulder_circle_time[:-2])
    # shoulder_circle_omega[0] = np.deg2rad(shoulder_circle_angle[1] - shoulder_circle_angle[0]) / (shoulder_circle_time[1] - shoulder_circle_time[0])
    # shoulder_circle_omega[-1] = np.deg2rad(shoulder_circle_angle[-1] - shoulder_circle_angle[-2]) / (shoulder_circle_time[-1] - shoulder_circle_time[-2])
    #角速度
    shoulder_circle_head_x = shoulder_circle_headhigh_x - shoulder_circle_headlow_x
    shoulder_circle_head_y = shoulder_circle_headhigh_y - shoulder_circle_headlow_y
    shoulder_circle_head = np.stack([shoulder_circle_head_x,shoulder_circle_head_y],1)
    shoulder_circle_cos =np.inner(shoulder_circle_head,np.array([[1,0]])).flatten() / np.linalg.norm(shoulder_circle_head,ord=2,axis=1)
    shoulder_circle_theta = np.arccos(shoulder_circle_cos)
    shoulder_circle_omega = np.zeros_like(shoulder_circle_head_x)
    shoulder_circle_omega[1:-1] = (shoulder_circle_theta[2:] - shoulder_circle_theta[:-2]) / (shoulder_circle_time[2:] - shoulder_circle_time[:-2])
    shoulder_circle_omega[0] = (shoulder_circle_theta[1] - shoulder_circle_theta[0]) / (shoulder_circle_time[1] - shoulder_circle_time[0])
    shoulder_circle_omega[-1] = (shoulder_circle_theta[-1] - shoulder_circle_theta[-2]) / (shoulder_circle_time[-1] - shoulder_circle_time[-2])
    shoulder_circle_std_dev = np.std(shoulder_circle_omega)
    print(shoulder_circle_omega.min(),shoulder_circle_omega.max())
    print(f"shoulder_circle_角速度の標準偏差: {shoulder_circle_std_dev:.4f} rad/s")
    
    # shoulder_circle_angular_velocity = np.diff(np.deg2rad(shoulder_circle_angle)) / np.diff(shoulder_circle_time)
    # shoulder_circle_time_adjust = (shoulder_circle_time[:-1] + shoulder_circle_time[1:]) / 2
    # shoulder_circle_bed_angle = np.mod(np.degrees(np.arctan(shoulder_circle_neck_angle)),360)
    # shoulder_circle_bed_angle_adjust = (shoulder_circle_bed_angle[:-1] + shoulder_circle_bed_angle[1:]) / 2
    
    df4 = pd.read_csv('../data/2025-01-25_neck-daen.csv')
    neck_daen_headhigh_y = df4['headhigh/Y']
    neck_daen_headhigh_x = df4['headhigh/Z']
    neck_daen_headlow_y = df4['headlow/Y']
    neck_daen_headlow_x = df4['headlow/Z']
    neck_daen_neckhigh_y = df4['neckhigh/Y']
    neck_daen_neckhigh_x = df4['neckhigh/Z']
    neck_daen_necklow_y = df4['necklow/Y']
    neck_daen_necklow_x = df4['necklow/Z']
    # neck_daen_time = df1['time']
    # neck_daen_time = neck_daen_time[850:1300]
    neck_daen_time = np.arange(0,15,0.01)
    neck_daen_headhigh_y = neck_daen_headhigh_y[4000:5500]
    neck_daen_headhigh_x = neck_daen_headhigh_x[4000:5500]
    neck_daen_headlow_y = neck_daen_headlow_y[4000:5500]
    neck_daen_headlow_x = neck_daen_headlow_x[4000:5500]
    neck_daen_neckhigh_y = neck_daen_neckhigh_y[4000:5500]
    neck_daen_neckhigh_x = neck_daen_neckhigh_x[4000:5500]
    neck_daen_necklow_y = neck_daen_necklow_y[4000:5500]
    neck_daen_necklow_x = neck_daen_necklow_x[4000:5500]

    neck_daen_mask = ~(np.isnan(neck_daen_headhigh_y) | np.isnan(neck_daen_headhigh_x) |np.isnan(neck_daen_headlow_y) |np.isnan(neck_daen_headlow_x) |np.isnan(neck_daen_neckhigh_y) |np.isnan(neck_daen_neckhigh_x) |np.isnan(neck_daen_necklow_y) |np.isnan(neck_daen_necklow_x))
    neck_daen_time = neck_daen_time[neck_daen_mask]
    neck_daen_headhigh_y = neck_daen_headhigh_y[neck_daen_mask]
    neck_daen_headhigh_x = neck_daen_headhigh_x[neck_daen_mask]
    neck_daen_headlow_y = neck_daen_headlow_y[neck_daen_mask]
    neck_daen_headlow_x = neck_daen_headlow_x[neck_daen_mask]
    neck_daen_neckhigh_y = neck_daen_neckhigh_y[neck_daen_mask]
    neck_daen_neckhigh_x = neck_daen_neckhigh_x[neck_daen_mask]
    neck_daen_necklow_y = neck_daen_necklow_y[neck_daen_mask]
    neck_daen_necklow_x = neck_daen_necklow_x[neck_daen_mask]
    
    neck_daen_head_angle = [(w-x) / (y-z) for w,x,y,z in zip(neck_daen_headhigh_y, neck_daen_headlow_y, neck_daen_headhigh_x,neck_daen_headlow_x)]
    neck_daen_neck_angle = [(w-x) / (y-z) for w,x,y,z in zip(neck_daen_neckhigh_y, neck_daen_necklow_y, neck_daen_neckhigh_x,neck_daen_necklow_x)]
    neck_daen_angle = np.degrees(np.arctan([(x-y)/(1+x*y) for x,y in zip(neck_daen_head_angle, neck_daen_neck_angle)]))

    # neck_daen_omega = np.zeros_like(neck_daen_angle)
    # neck_daen_omega[1:-1] = np.deg2rad(neck_daen_angle[2:] - neck_daen_angle[:-2]) / (neck_daen_time[2:] - neck_daen_time[:-2])
    # neck_daen_omega[0] = np.deg2rad(neck_daen_angle[1] - neck_daen_angle[0]) / (neck_daen_time[1] - neck_daen_time[0])
    # neck_daen_omega[-1] = np.deg2rad(neck_daen_angle[-1] - neck_daen_angle[-2]) / (neck_daen_time[-1] - neck_daen_time[-2])
    #角速度
    neck_daen_head_x = neck_daen_headhigh_x - neck_daen_headlow_x
    neck_daen_head_y = neck_daen_headhigh_y - neck_daen_headlow_y
    neck_daen_head = np.stack([neck_daen_head_x,neck_daen_head_y],1)
    neck_daen_cos =np.inner(neck_daen_head,np.array([[1,0]])).flatten() / np.linalg.norm(neck_daen_head,ord=2,axis=1)
    neck_daen_theta = np.arccos(neck_daen_cos)
    neck_daen_omega = np.zeros_like(neck_daen_head_x)
    neck_daen_omega[1:-1] = (neck_daen_theta[2:] - neck_daen_theta[:-2]) / (neck_daen_time[2:] - neck_daen_time[:-2])
    neck_daen_omega[0] = (neck_daen_theta[1] - neck_daen_theta[0]) / (neck_daen_time[1] - neck_daen_time[0])
    neck_daen_omega[-1] = (neck_daen_theta[-1] - neck_daen_theta[-2]) / (neck_daen_time[-1] - neck_daen_time[-2])
    
    neck_daen_std_dev = np.std(neck_daen_omega)
    print(neck_daen_omega.min(),neck_daen_omega.max())
    print(f"neck_daen_角速度の標準偏差: {neck_daen_std_dev:.4f} rad/s")
    # neck_daen_angular_velocity = np.diff(np.deg2rad(neck_daen_angle)) / np.diff(neck_daen_time)
    # neck_daen_time_adjust = (neck_daen_time[:-1] + neck_daen_time[1:]) / 2
    # neck_daen_bed_angle = np.mod(np.degrees(np.arctan(neck_daen_neck_angle)),360)
    # neck_daen_bed_angle_adjust = (neck_daen_bed_angle[:-1] + neck_daen_bed_angle[1:]) / 2

    plt.plot(human_time, human_omega, label="human", color="red")
    plt.plot(shoulder_daen_time, shoulder_daen_omega, label="neck-ellipse", color="blue")
    plt.plot(shoulder_circle_time, shoulder_circle_omega, label="neck-circle", color="green")
    plt.plot(neck_daen_time, neck_daen_omega, label="head-ellipse", color="black")
    plt.xlabel("time [s]")

    # plt.plot(human_time_adjust, human_angular_velocity, label="human", color="red")
    # plt.plot(shoulder_daen_time_adjust, shoulder_daen_angular_velocity,label="neck-ellipse", color="blue")
    # plt.plot(shoulder_circle_time_adjust, shoulder_circle_angular_velocity,label="neck-circle", color="green")
    # plt.plot(neck_daen_time_adjust, neck_daen_angular_velocity,label="head-ellipse", color="black")
    # plt.xlabel("time [s]")

    # plt.plot(human_time_adjust, human_bed_angle_adjust, label="human", color="red")
    # plt.plot(shoulder_daen_time_adjust, shoulder_daen_bed_angle_adjust,label="neck-ellipse", color="blue")
    # plt.plot(shoulder_circle_time_adjust, shoulder_circle_bed_angle_adjust,label="neck-circle", color="green")
    # plt.plot(neck_daen_time_adjust, neck_daen_bed_angle_adjust,label="head-ellipse", color="black")
    
    # plt.plot(human_bed_angle_adjust, human_angular_velocity, label="human", color="red")
    # plt.plot(shoulder_daen_bed_angle_adjust, shoulder_daen_angle,label="neck-ellipse(this study)", color="blue")
    # plt.plot(shoulder_circle_bed_angle_adjust, shoulder_circle_angle,label="neck-circle", color="green")
    # plt.plot(neck_daen_bed_angle_adjust, neck_daen_angle,label="head-ellipse", color="black")
    # plt.xlim(0,90)
    # plt.xlabel("Angle between bed and body [degree]")
    
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Neck's angular velocity during movement")
    plt.legend()
    plt.show()

    
# ipython

# with open('mocap_csv.py') as f:
#     code = f.read()
# exec(code)
    
# human_head_yz()
