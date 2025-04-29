from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def ellipse(x, xc, yc, a, b):
    inside_sqrt = 1 - ((x - xc) / a) ** 2
    inside_sqrt[inside_sqrt < 0] = 0
    return yc + b * np.sqrt(inside_sqrt)

df1 = pd.read_csv('../data/2025-01-25_human.csv')
human_head_y = df1['headhigh/Y']
human_head_x = df1['headhigh/Z']
human_head_y = human_head_y[500:1200]
human_head_x = human_head_x[500:1200]
human_neck_y = df1['neckhigh/Y']
human_neck_x = df1['neckhigh/Z']
human_neck_y = human_neck_y[500:1200]
human_neck_x = human_neck_x[500:1200]

head_x_start = human_head_x[500]
head_y_start = human_head_y[500]
head_x_end = human_head_x[1199]
head_y_end = human_head_y[1199]
head_initial = [head_x_end, head_y_start, head_x_start - head_x_end, head_y_end - head_y_start]

neck_x_start = human_neck_x[500]
neck_y_start = human_neck_y[500]
neck_x_end = human_neck_x[1199]
neck_y_end = human_neck_y[1199]
neck_initial = [neck_x_end, neck_y_start, neck_x_start - neck_x_end, neck_y_end - neck_y_start]

head_popt, head_pcov = curve_fit(ellipse, human_head_x, human_head_y, p0=head_initial)
neck_popt, neck_pcov = curve_fit(ellipse, human_neck_x, human_neck_y, p0=neck_initial)

head_x_draw = np.linspace(head_popt[0]-head_popt[2], head_popt[0] + head_popt[2], 500)
head_y_fit = head_popt[1] + head_popt[3] * np.sqrt(1 - ((head_x_draw - head_popt[0]) / head_popt[2]) ** 2)
neck_x_draw = np.linspace(neck_popt[0]-neck_popt[2], neck_popt[0] + neck_popt[2], 500)
neck_y_fit = neck_popt[1] + neck_popt[3] * np.sqrt(1 - ((neck_x_draw - neck_popt[0]) / neck_popt[2]) ** 2)

# print(head_initial,head_popt,head_pcov)
param_std = np.sqrt(np.diagonal(head_pcov))
for i, param in enumerate(head_popt):
    print(f"パラメータ {i} : 推定値 = {param}, 標準誤差 = {param_std[i]}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(human_head_x, human_head_y, label="human_head", color="red")
ax.plot(head_x_draw, head_y_fit, label="head fitted ellipse", color="blue")
# plt.plot(human_neck_x, human_neck_y, label="human_neck", color="red")
# plt.plot(neck_x_draw, neck_y_fit, label="neck fitted quarter-ellipse", color="blue")
ax.set_aspect('auto')
ax.legend(fontsize=15)
ax.set_title("ellipse-fitting",fontsize=20)
ax.set_xlabel("x",fontsize=15)
ax.set_ylabel("y",fontsize=15)
plt.show()
