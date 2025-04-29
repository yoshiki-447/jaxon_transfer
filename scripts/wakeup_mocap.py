import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cal_angle(coord1_x, coord1_y, coord2_x, coord2_y, coord3_x, coord3_y, horizon): #coord1 top
    vec1_x = coord1_x - coord2_x
    vec1_y = coord1_y - coord2_y
    vec1 = np.stack([vec1_x,vec1_y],1)
    if(horizon):
        cos = np.inner(vec1,np.array([[1,0]])).flatten() / np.linalg.norm(vec1,ord=2,axis=1)
        angle = np.arccos(cos)
    else:
        vec2_x = coord2_x - coord3_x
        vec2_y = coord2_y - coord3_y
        vec2 = np.stack([vec2_x,vec2_y],1)
        cos = np.diag(np.inner(vec1,vec2)) / np.linalg.norm(vec1,ord=2,axis=1) / np.linalg.norm(vec2,ord=2,axis=1)
        angle = np.arccos(cos)
    return angle

def setup(file, start, end):
    df1 = pd.read_csv(file) #'../data/2025-01-25_human.csv'
    head_y = df1['headhigh/Y']
    head_x = df1['headhigh/Z']
    neck_y = df1['headlow/Y']
    neck_x = df1['headlow/Z']
    chest_y = df1['neckhigh/Y']
    chest_x = df1['neckhigh/Z']
    waist_y = df1['necklow/Y']
    waist_x = df1['necklow/Z']

    time = np.arange(0,(end-start)/100,0.01)
    head_y = head_y[start:end]
    head_x = head_x[start:end]
    neck_y = neck_y[start:end]
    neck_x = neck_x[start:end]
    chest_y = chest_y[start:end]
    chest_x = chest_x[start:end]
    waist_y = waist_y[start:end]
    waist_x = waist_x[start:end]

    mask = ~(np.isnan(head_y) | np.isnan(head_x) |np.isnan(neck_y) |np.isnan(neck_x) |np.isnan(chest_y) |np.isnan(chest_x) |np.isnan(waist_y) |np.isnan(waist_x))
    time = time[mask]
    head_y = head_y[mask]
    head_x = head_x[mask]
    neck_y = neck_y[mask]
    neck_x = neck_x[mask]
    chest_y = chest_y[mask]
    chest_x = chest_x[mask]
    waist_y = waist_y[mask]
    waist_x = waist_x[mask]

    neck_angle = np.rad2deg(cal_angle(head_x,head_y,neck_x,neck_y,chest_x,chest_y,False))
    chest_angle = np.rad2deg(cal_angle(neck_x,neck_y,chest_x,chest_y,waist_x,waist_y,False))

    theta = cal_angle(head_x,head_y,neck_x,neck_y,_,_,True) #angle between head and horizon
    omega = np.zeros_like(time)
    omega[1:-1] = (theta[2:] - theta[:-2]) / (time[2:] - time[:-2])
    omega[0] = (theta[1] - theta[0]) / (time[1] - time[0])
    omega[-1] = (theta[-1] - theta[-2]) / (time[-1] - time[-2])
    std_dev = np.std(omega)

    return time, neck_angle, chest_angle, omega, std_dev

def omega():
    human_time,_,_,human_omega,human_std_dev = setup('../data/2025-01-25_human.csv',500,1200)
    shoulder_daen_time,_,_,shoulder_daen_omega,shoulder_daen_std_dev = setup('../data/2025-01-25_shoulder-daen.csv',4000,5500)
    shoulder_circle_time,_,_,shoulder_circle_omega,shoulder_circle_std_dev = setup('../data/2025-01-25_shoulder-circle.csv',4000,5500)
    neck_daen_time,_,_,neck_daen_omega,neck_daen_std_dev = setup('../data/2025-01-25_neck-daen.csv',4000,5500)
    print(human_omega.min(),human_omega.max())
    print(f"human_角速度の標準偏差: {human_std_dev:.4f} rad/s")
    print(shoulder_daen_omega.min(),shoulder_daen_omega.max())
    print(f"shoulder_daen_角速度の標準偏差: {shoulder_daen_std_dev:.4f} rad/s")
    print(shoulder_circle_omega.min(),shoulder_circle_omega.max())
    print(f"shoulder_circle_角速度の標準偏差: {shoulder_circle_std_dev:.4f} rad/s")
    print(neck_daen_omega.min(),neck_daen_omega.max())
    print(f"neck_daen_角速度の標準偏差: {neck_daen_std_dev:.4f} rad/s")

    fig, ax = plt.subplots() #figsize=(10, 5)
    ax.plot(human_time, human_omega, label="human", color="red")
    ax.plot(shoulder_daen_time, shoulder_daen_omega, label="neck-ellipse", color="blue")
    ax.plot(shoulder_circle_time, shoulder_circle_omega, label="neck-circle", color="green")
    ax.plot(neck_daen_time, neck_daen_omega, label="head-ellipse", color="black")
    ax.set_aspect('auto')
    ax.set_xlabel("time [s]",fontsize=15)
    ax.set_ylabel("Angular velocity [rad/s]",fontsize=15)
    ax.set_title("Head's angular velocity during movement",fontsize=20)
    ax.legend(fontsize=15)
    plt.show()

def neck_angle():
    human_time,human_neck_angle,_,_,_ = setup('../data/2025-01-25_human.csv',500,1200)
    shoulder_daen_time,shoulder_daen_neck_angle,_,_,_ = setup('../data/2025-01-25_shoulder-daen.csv',4000,5500)
    shoulder_circle_time,shoulder_circle_neck_angle,_,_,_ = setup('../data/2025-01-25_shoulder-circle.csv',4000,5500)
    neck_daen_time,neck_daen_neck_angle,_,_,_ = setup('../data/2025-01-25_neck-daen.csv',4000,5500)
    print(f"human_最大角度: {human_neck_angle.max():.4f} degree")
    print(f"human_最小角度: {human_neck_angle.min():.4f} degree")
    print(f"shoulder_daen_最大角度: {shoulder_daen_neck_angle.max():.4f} degree")
    print(f"shoulder_daen_最小角度: {shoulder_daen_neck_angle.min():.4f} degree")
    print(f"shoulder_circle_最大角度: {shoulder_circle_neck_angle.max():.4f} degree")
    print(f"shoulder_circle_最小角度: {shoulder_circle_neck_angle.min():.4f} degree")
    print(f"neck_daen_最大角度: {neck_daen_neck_angle.max():.4f} degree")
    print(f"neck_daen_最小角度: {neck_daen_neck_angle.min():.4f} degree")
    
    fig, ax = plt.subplots() #figsize=(10, 5)
    ax.plot(human_time, human_neck_angle, label="human", color="red")
    ax.plot(shoulder_daen_time, shoulder_daen_neck_angle, label="neck-ellipse", color="blue")
    ax.plot(shoulder_circle_time, shoulder_circle_neck_angle, label="neck-circle", color="green")
    ax.plot(neck_daen_time, neck_daen_neck_angle, label="head-ellipse", color="black")
    ax.set_aspect('auto')
    ax.set_xlabel("time [s]",fontsize=15)
    ax.set_ylabel("theta_neck [degree]",fontsize=15)
    ax.set_title("theta_neck during movement",fontsize=20)
    ax.legend(fontsize=15)
    plt.show()
    
def chest_angle():
    human_time,_,human_chest_angle,_,_ = setup('../data/2025-01-25_human.csv',500,1200)
    shoulder_daen_time,_,shoulder_daen_chest_angle,_,_ = setup('../data/2025-01-25_shoulder-daen.csv',4000,5500)
    shoulder_circle_time,_,shoulder_circle_chest_angle,_,_ = setup('../data/2025-01-25_shoulder-circle.csv',4000,5500)
    neck_daen_time,_,neck_daen_chest_angle,_,_ = setup('../data/2025-01-25_neck-daen.csv',4000,5500)
    print(f"human_最大角度: {human_chest_angle.max():.4f} degree")
    print(f"human_最小角度: {human_chest_angle.min():.4f} degree")
    print(f"shoulder_daen_最大角度: {shoulder_daen_chest_angle.max():.4f} degree")
    print(f"shoulder_daen_最小角度: {shoulder_daen_chest_angle.min():.4f} degree")
    print(f"shoulder_circle_最大角度: {shoulder_circle_chest_angle.max():.4f} degree")
    print(f"shoulder_circle_最小角度: {shoulder_circle_chest_angle.min():.4f} degree")
    print(f"neck_daen_最大角度: {neck_daen_chest_angle.max():.4f} degree")
    print(f"neck_daen_最小角度: {neck_daen_chest_angle.min():.4f} degree")
    
    fig, ax = plt.subplots() #figsize=(10, 5)
    ax.plot(human_time, human_chest_angle, label="human", color="red")
    ax.plot(shoulder_daen_time, shoulder_daen_chest_angle, label="neck-ellipse", color="blue")
    ax.plot(shoulder_circle_time, shoulder_circle_chest_angle, label="neck-circle", color="green")
    ax.plot(neck_daen_time, neck_daen_chest_angle, label="head-ellipse", color="black")
    ax.set_aspect('auto')
    ax.set_xlabel("time [s]",fontsize=15)
    ax.set_ylabel("theta_chest [degree]",fontsize=15)
    ax.set_title("theta_chest during movement",fontsize=20)
    ax.legend(fontsize=15)
    plt.show()
