import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def distance(a, b, c, x0, y0):
    numerator = abs(a * x0 + b * y0 + c)
    denominator = np.sqrt(a**2 + b**2)
    distance = numerator / denominator
    return distance

def human_yz():
    df = pd.read_csv('../data/2025-01-16_human.csv')
    head_y = df['head/Y']
    head_z = df['head/Z']
    neck_y = df['neck/Y']
    neck_z = df['neck/Z']
    # new_head_y = [x for x in head_y if not np.isnan(x)]
    # new_head_z = [x for x in head_z if not np.isnan(x)]
    # new_head_z = [-x for x in new_head_z]

    time = df['time']
    time = time[850:1300]
    
    head_y = head_y[850:1300]
    head_z = head_z[850:1300]
    neck_y = neck_y[850:1300]
    neck_z = neck_z[850:1300]

    center_y = neck_y[850]
    center_z = neck_z[1299]

    diff_y = [x-y for x,y in zip(head_y, neck_y)]
    diff_z = [x-y for x,y in zip(head_z, neck_z)]
    # angle = np.degrees(np.arctan2(diff_y,diff_z))

    diff = [np.sqrt((x ** 2) + (y ** 2)) for x,y in zip(diff_y, diff_z)]
    point2line = [distance(d-center_y, center_z-c ,center_y*c - center_z*d ,a,b) for a,b,c,d in zip(head_z, head_y, neck_z, neck_y)]
    angle = np.degrees(np.arcsin([y/x for x,y in zip(diff, point2line)]))
    
    plt.plot(time, angle)
    plt.xlabel("time")
    plt.ylabel("angle")
    plt.title("head to neck angle")
    
    # plt.plot(head_z, head_y, label="head")
    # plt.plot(neck_z, neck_y, label="neck")  # mocap_y,z -> world_z,-y
    # plt.xlabel('y')
    # plt.ylabel('z')
    # plt.title('human_yz')
    
    plt.legend()
    plt.show()

def shoulder_daen_yz():
    df = pd.read_csv('../data/2025-01-16_shoulder-daen.csv')
    head_y = df['head/Y']
    head_z = df['head/Z']
    neck_y = df['neck/Y']
    neck_z = df['neck/Z']
    # new_head_y = [x for x in head_y if not np.isnan(x)]
    # new_head_z = [x for x in head_z if not np.isnan(x)]
    # new_head_z = [-x for x in new_head_z]

    head_y = head_y[3000:5800]
    head_z = head_z[3000:5800]
    neck_y = neck_y[3000:5800]
    neck_z = neck_z[3000:5800]
    
    plt.plot(head_z, head_y, label="head")
    plt.plot(neck_z, neck_y, label="neck")
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('shoulder-daen_yz')
    plt.legend()
    plt.show()

def shoulder_circle_yz():
    df = pd.read_csv('../data/2025-01-16_shoulder-circle.csv')
    head_y = df['head/Y']
    head_z = df['head/Z']
    neck_y = df['neck/Y']
    neck_z = df['neck/Z']
    # new_head_y = [x for x in head_y if not np.isnan(x)]
    # new_head_z = [x for x in head_z if not np.isnan(x)]
    # new_head_z = [-x for x in new_head_z]

    head_y = head_y[3000:5500]
    head_z = head_z[3000:5500]
    neck_y = neck_y[3000:5500]
    neck_z = neck_z[3000:5500]
    
    plt.plot(head_z, head_y, label="head")
    plt.plot(neck_z, neck_y, label="neck")
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('shoulder-circle_yz')
    plt.legend()
    plt.show()

def neck_daen_yz():
    df = pd.read_csv('../data/2025-01-16_neck-daen.csv')
    head_y = df['head/Y']
    head_z = df['head/Z']
    neck_y = df['neck/Y']
    neck_z = df['neck/Z']
    # new_head_y = [x for x in head_y if not np.isnan(x)]
    # new_head_z = [x for x in head_z if not np.isnan(x)]
    # new_head_z = [-x for x in new_head_z]

    head_y = head_y[4000:6000]
    head_z = head_z[4000:6000]
    neck_y = neck_y[4000:6000]
    neck_z = neck_z[4000:6000]
    
    plt.plot(head_z, head_y, label="head")
    plt.plot(neck_z, neck_y, label="neck")
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('neck-daen_yz')
    plt.legend()
    plt.show()

def wakeup_yz():
    df1 = pd.read_csv('../data/2025-01-16_human.csv')
    human_head_y = df1['head/Y']
    human_head_z = df1['head/Z']
    human_neck_y = df1['neck/Y']
    human_neck_z = df1['neck/Z']

    human_head_y = human_head_y[850:1300]
    human_head_z = human_head_z[850:1300]
    human_neck_y = human_neck_y[850:1300]
    human_neck_z = human_neck_z[850:1300]
    
    df2 = pd.read_csv('../data/2025-01-16_shoulder-daen.csv')
    shoulder_daen_head_y = df2['head/Y']
    shoulder_daen_head_z = df2['head/Z']
    shoulder_daen_neck_y = df2['neck/Y']
    shoulder_daen_neck_z = df2['neck/Z']

    shoulder_daen_head_y = shoulder_daen_head_y[4500:5800]
    shoulder_daen_head_z = shoulder_daen_head_z[4500:5800]
    shoulder_daen_neck_y = shoulder_daen_neck_y[4500:5800]
    shoulder_daen_neck_z = shoulder_daen_neck_z[4500:5800]
    
    df3 = pd.read_csv('../data/2025-01-16_shoulder-circle.csv')
    shoulder_circle_head_y = df3['head/Y']
    shoulder_circle_head_z = df3['head/Z']
    shoulder_circle_neck_y = df3['neck/Y']
    shoulder_circle_neck_z = df3['neck/Z']

    shoulder_circle_head_y = shoulder_circle_head_y[3700:5000]
    shoulder_circle_head_z = shoulder_circle_head_z[3700:5000]
    shoulder_circle_neck_y = shoulder_circle_neck_y[3700:5000]
    shoulder_circle_neck_z = shoulder_circle_neck_z[3700:5000]
    
    df4 = pd.read_csv('../data/2025-01-16_neck-daen.csv')
    neck_daen_head_y = df4['head/Y']
    neck_daen_head_z = df4['head/Z']
    neck_daen_neck_y = df4['neck/Y']
    neck_daen_neck_z = df4['neck/Z']

    neck_daen_head_y = neck_daen_head_y[4500:5500]
    neck_daen_head_z = neck_daen_head_z[4500:5500]
    neck_daen_neck_y = neck_daen_neck_y[4500:5500]
    neck_daen_neck_z = neck_daen_neck_z[4500:5500]
    
    plt.plot(human_head_z, human_head_y, label="human-head", color="red")
    plt.plot(shoulder_daen_head_z, shoulder_daen_head_y, label="shoulder-daen-head", color="blue")
    plt.plot(shoulder_circle_head_z, shoulder_circle_head_y, label="shoulder-circle-head", color="green")
    plt.plot(neck_daen_head_z, neck_daen_head_y, label="neck-daen-head", color="black")

    plt.plot(human_neck_z, human_neck_y, "--", label="human-neck" , color="red")
    plt.plot(shoulder_daen_neck_z, shoulder_daen_neck_y, "--" ,label="shoulder-daen-neck", color="blue")
    plt.plot(shoulder_circle_neck_z, shoulder_circle_neck_y, "--" ,label="shoulder-circle-neck", color="green")
    plt.plot(neck_daen_neck_z, neck_daen_neck_y, "--" ,label="neck-daen-neck", color="black")
    
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('head-neck_yz')
    plt.legend()
    plt.show()

def wakeup_angle():
    df1 = pd.read_csv('../data/2025-01-16_human.csv')
    human_head_y = df1['head/Y']
    human_head_z = df1['head/Z']
    human_neck_y = df1['neck/Y']
    human_neck_z = df1['neck/Z']
    # human_time = df1['time']
    # human_time = human_time[850:1300]
    human_time = np.arange(0,4.5,0.01)
    human_head_y = human_head_y[850:1300]
    human_head_z = human_head_z[850:1300]
    human_neck_y = human_neck_y[850:1300]
    human_neck_z = human_neck_z[850:1300]
    human_center_y = human_neck_y[850]
    human_center_z = human_neck_z[1299]
    human_diff_y = [x-y for x,y in zip(human_head_y, human_neck_y)]
    human_diff_z = [x-y for x,y in zip(human_head_z, human_neck_z)]
    human_diff = [np.sqrt((x ** 2) + (y ** 2)) for x,y in zip(human_diff_y, human_diff_z)]
    human_point2line = [distance(d-human_center_y, human_center_z-c ,human_center_y*c - human_center_z*d ,a,b) for a,b,c,d in zip(human_head_z, human_head_y, human_neck_z, human_neck_y)]
    human_angle = np.degrees(np.arcsin([y/x for x,y in zip(human_diff, human_point2line)]))

    df2 = pd.read_csv('../data/2025-01-16_shoulder-daen.csv')
    shoulder_daen_head_y = df2['head/Y']
    shoulder_daen_head_z = df2['head/Z']
    shoulder_daen_neck_y = df2['neck/Y']
    shoulder_daen_neck_z = df2['neck/Z']
    # shoulder_daen_time = df2['time']
    # shoulder_daen_time = shoulder_daen_time[3000:5800]
    shoulder_daen_time = np.arange(0,13,0.01)
    shoulder_daen_head_y = shoulder_daen_head_y[4500:5800]
    shoulder_daen_head_z = shoulder_daen_head_z[4500:5800]
    shoulder_daen_neck_y = shoulder_daen_neck_y[4500:5800]
    shoulder_daen_neck_z = shoulder_daen_neck_z[4500:5800]
    shoulder_daen_center_y = shoulder_daen_neck_y[4500]
    shoulder_daen_center_z = shoulder_daen_neck_z[5799]
    shoulder_daen_diff_y = [x-y for x,y in zip(shoulder_daen_head_y, shoulder_daen_neck_y)]
    shoulder_daen_diff_z = [x-y for x,y in zip(shoulder_daen_head_z, shoulder_daen_neck_z)]
    shoulder_daen_diff = [np.sqrt((x ** 2) + (y ** 2)) for x,y in zip(shoulder_daen_diff_y, shoulder_daen_diff_z)]
    shoulder_daen_point2line = [distance(d-shoulder_daen_center_y, shoulder_daen_center_z-c ,shoulder_daen_center_y*c - shoulder_daen_center_z*d ,a,b) for a,b,c,d in zip(shoulder_daen_head_z, shoulder_daen_head_y, shoulder_daen_neck_z, shoulder_daen_neck_y)]
    shoulder_daen_angle = np.degrees(np.arcsin([y/x for x,y in zip(shoulder_daen_diff, shoulder_daen_point2line)]))

    df3 = pd.read_csv('../data/2025-01-16_shoulder-circle.csv')
    shoulder_circle_head_y = df3['head/Y']
    shoulder_circle_head_z = df3['head/Z']
    shoulder_circle_neck_y = df3['neck/Y']
    shoulder_circle_neck_z = df3['neck/Z']
    # shoulder_circle_time = df3['time']
    # shoulder_circle_time = shoulder_circle_time[3000:5500]
    shoulder_circle_time = np.arange(0,13,0.01)
    shoulder_circle_head_y = shoulder_circle_head_y[3700:5000]
    shoulder_circle_head_z = shoulder_circle_head_z[3700:5000]
    shoulder_circle_neck_y = shoulder_circle_neck_y[3700:5000]
    shoulder_circle_neck_z = shoulder_circle_neck_z[3700:5000]
    shoulder_circle_center_y = shoulder_circle_neck_y[3700]
    shoulder_circle_center_z = shoulder_circle_neck_z[4999]
    shoulder_circle_diff_y = [x-y for x,y in zip(shoulder_circle_head_y, shoulder_circle_neck_y)]
    shoulder_circle_diff_z = [x-y for x,y in zip(shoulder_circle_head_z, shoulder_circle_neck_z)]
    shoulder_circle_diff = [np.sqrt((x ** 2) + (y ** 2)) for x,y in zip(shoulder_circle_diff_y, shoulder_circle_diff_z)]
    shoulder_circle_point2line = [distance(d-shoulder_circle_center_y, shoulder_circle_center_z-c ,shoulder_circle_center_y*c - shoulder_circle_center_z*d ,a,b) for a,b,c,d in zip(shoulder_circle_head_z, shoulder_circle_head_y, shoulder_circle_neck_z, shoulder_circle_neck_y)]
    shoulder_circle_angle = np.degrees(np.arcsin([y/x for x,y in zip(shoulder_circle_diff, shoulder_circle_point2line)]))

    df4 = pd.read_csv('../data/2025-01-16_neck-daen.csv')
    neck_daen_head_y = df4['head/Y']
    neck_daen_head_z = df4['head/Z']
    neck_daen_neck_y = df4['neck/Y']
    neck_daen_neck_z = df4['neck/Z']
    # neck_daen_time = df4['time']
    # neck_daen_time = neck_daen_time[4000:6000]
    neck_daen_time = np.arange(0,10,0.01)
    neck_daen_head_y = neck_daen_head_y[4500:5500]
    neck_daen_head_z = neck_daen_head_z[4500:5500]
    neck_daen_neck_y = neck_daen_neck_y[4500:5500]
    neck_daen_neck_z = neck_daen_neck_z[4500:5500]
    neck_daen_center_y = neck_daen_neck_y[4500]
    neck_daen_center_z = neck_daen_neck_z[5499]
    neck_daen_diff_y = [x-y for x,y in zip(neck_daen_head_y, neck_daen_neck_y)]
    neck_daen_diff_z = [x-y for x,y in zip(neck_daen_head_z, neck_daen_neck_z)]
    neck_daen_diff = [np.sqrt((x ** 2) + (y ** 2)) for x,y in zip(neck_daen_diff_y, neck_daen_diff_z)]
    neck_daen_point2line = [distance(d-neck_daen_center_y, neck_daen_center_z-c ,neck_daen_center_y*c - neck_daen_center_z*d ,a,b) for a,b,c,d in zip(neck_daen_head_z, neck_daen_head_y, neck_daen_neck_z, neck_daen_neck_y)]
    neck_daen_angle = np.degrees(np.arcsin([y/x for x,y in zip(neck_daen_diff, neck_daen_point2line)]))
    
    plt.plot(human_time, human_angle, label="human", color="red")
    plt.plot(shoulder_daen_time, shoulder_daen_angle,label="shoulder-daen", color="blue")
    plt.plot(shoulder_circle_time, shoulder_circle_angle,label="shoulder-circle", color="green")
    plt.plot(neck_daen_time, neck_daen_angle,label="neck-daen", color="black")
    plt.xlabel("time[s]")
    plt.ylabel("angle[degree]")
    plt.title("angle between head and neck")
    plt.legend()
    plt.show()
    
# ipython

# with open('mocap_csv.py') as f:
#     code = f.read()
# exec(code)
    
# human_head_yz()
