import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def human_xz():
    df = pd.read_csv('../data/2025-01-17_board-human.csv')
    head_x = df['head/X']
    head_z = df['head/Z']
    neck_x = df['neck/X']
    neck_z = df['neck/Z']
    head_x = [x for x in head_x if not np.isnan(x)]
    head_z = [x for x in head_z if not np.isnan(x)]
    neck_x = [x for x in neck_x if not np.isnan(x)]
    neck_z = [x for x in neck_z if not np.isnan(x)]
    # new_head_z = [-x for x in new_head_z]

    head_x = head_x[750:1100]
    head_z = head_z[750:1100]
    neck_x = neck_x[750:1100]
    neck_z = neck_z[750:1100]
    
    plt.plot(head_z, head_x,label="head")  # mocap_y,z -> world_z,-y
    plt.plot(neck_z, neck_x,label="neck")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('human_xy')
    plt.legend()
    plt.show()

def jaxon_xz():
    df = pd.read_csv('../data/2025-01-17_board-jaxon.csv')
    head_x = df['head/X']
    head_z = df['head/Z']
    neck_x = df['neck/X']
    neck_z = df['neck/Z']
    # head_x = [x for x in head_x if not np.isnan(x)]
    # head_z = [x for x in head_z if not np.isnan(x)]
    # neck_x = [x for x in neck_x if not np.isnan(x)]
    # neck_z = [x for x in neck_z if not np.isnan(x)]
    # new_head_z = [-x for x in new_head_z]

    head_x = head_x[6900:8500]
    head_z = head_z[6900:8500]
    neck_x = neck_x[6900:8500]
    neck_z = neck_z[6900:8500]
    
    plt.plot(head_z, head_x)  # mocap_y,z -> world_z,-y
    plt.plot(neck_z, neck_x)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('jaxon_xy')
    plt.legend()
    plt.show()

def board_xz():
    df1 = pd.read_csv('../data/2025-01-17_board-human.csv')
    human_head_x = df1['head/X']
    human_head_z = df1['head/Z']
    human_neck_x = df1['neck/X']
    human_neck_z = df1['neck/Z']
    human_head_x = [x for x in human_head_x if not np.isnan(x)]
    human_head_z = [x for x in human_head_z if not np.isnan(x)]
    human_neck_x = [x for x in human_neck_x if not np.isnan(x)]
    human_neck_z = [x for x in human_neck_z if not np.isnan(x)]

    human_head_x = human_head_x[750:1100]
    human_head_z = human_head_z[750:1100]
    human_neck_x = human_neck_x[750:1100]
    human_neck_z = human_neck_z[750:1100]
    
    df2 = pd.read_csv('../data/2025-01-17_board-jaxon.csv')
    jaxon_head_x = df2['head/X']
    jaxon_head_z = df2['head/Z']
    jaxon_neck_x = df2['neck/X']
    jaxon_neck_z = df2['neck/Z']
    
    jaxon_head_x = jaxon_head_x[6900:8500]
    jaxon_head_z = jaxon_head_z[6900:8500]
    jaxon_neck_x = jaxon_neck_x[6900:8500]
    jaxon_neck_z = jaxon_neck_z[6900:8500]
    
    plt.plot(human_head_z, human_head_x, label="human-head", color="red")
    plt.plot(human_neck_z, human_neck_x, "--", label="human-neck", color="red")
    plt.plot(jaxon_head_z, jaxon_head_x, label="jaxon-head", color="blue")
    plt.plot(jaxon_neck_z, jaxon_neck_x, "--", label="jaxon-neck", color="blue")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('board_xy')
    plt.legend()
    plt.show()

def wakeup_angle():
    df1 = pd.read_csv('../data/2025-01-17_board-human.csv')
    human_head_x = df1['head/X']
    human_head_z = df1['head/Z']
    human_neck_x = df1['neck/X']
    human_neck_z = df1['neck/Z']
    human_head_x = [x for x in human_head_x if not np.isnan(x)]
    human_head_z = [x for x in human_head_z if not np.isnan(x)]
    human_neck_x = [x for x in human_neck_x if not np.isnan(x)]
    human_neck_z = [x for x in human_neck_z if not np.isnan(x)]
    # human_time = df1['time']
    # human_time = human_time[850:1300]
    human_time = np.arange(0,3.5,0.01)
    human_head_x = human_head_x[750:1100]
    human_head_z = human_head_z[750:1100]
    human_neck_x = human_neck_x[750:1100]
    human_neck_z = human_neck_z[750:1100]
    human_center_x = human_neck_y[1099]
    human_center_z = human_neck_z[750]
    human_diff_x = [x-y for x,y in zip(human_head_x, human_neck_x)]
    human_diff_z = [x-y for x,y in zip(human_head_z, human_neck_z)]
    human_diff = [np.sqrt((x ** 2) + (y ** 2)) for x,y in zip(human_diff_y, human_diff_z)]
    human_point2line = [distance(d-human_center_y, human_center_z-c ,human_center_y*c - human_center_z*d ,a,b) for a,b,c,d in zip(human_head_z, human_head_y, human_neck_z, human_neck_y)]
    human_angle = np.degrees(np.arcsin([y/x for x,y in zip(human_diff, human_point2line)]))    

# ipython

# with open('mocap_board.py') as f:
#     code = f.read()
# exec(code)
    
# human_head_yz()
