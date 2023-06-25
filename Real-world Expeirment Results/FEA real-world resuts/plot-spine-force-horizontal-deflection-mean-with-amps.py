import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib

import matplotlib

from matplotlib import font_manager

font_path = font_manager.findfont(font_manager.FontProperties(family="Myriad Pro"))

font_dict = {
    'family': 'Myriad Pro',
    'size': 8,
    # 'weight': 'bold'
}

plt.rcParams['font.family'] = font_dict['family']
plt.rcParams['font.size'] = font_dict['size']
plt.rcParams['lines.markersize'] = 3
plt.rcParams['font.weight'] = 'bold'

plt.rcParams['grid.color'] = (0., 0., 0., 0.1)

# sns.set_style("whitegrid")

deflection_force = np.array([1, 2, 3, 4, 5])

# helper = np.full((5, 5), 90)

simulated_horizontal_deflection_angle = np.array([25.3, 37.2, 51.4, 62.2, 75.6])

measured_horizontal_deflection_angle = np.array(
    [[28, 45, 58, 75, 85],
    [30, 40, 53, 70, 89],
    [32, 41, 50, 68, 80],
    [27, 46, 48, 73, 78],
    [29, 45, 55, 71, 86]])

measured_mampere_servo = np.array(
    [[48, 62, 81, 113, 166],
    [45, 70, 99, 108, 175],
    [42, 58, 87, 104, 172],
    [33, 56, 82, 121, 168],
    [45, 66, 85, 111, 160]])

milliwatts_servo = measured_mampere_servo * 5.16 * 0.001  # Power Supply provided 5.16 V
# print(milliwatts_servo)

# horizontal_deflection_angle = np.subtract(helper, measured_horizontal_deflection_angle)

horizontal_deflection_angle_mean = np.mean(measured_horizontal_deflection_angle, axis=0)
# print(horizontal_deflection_angle_mean)

horizontal_deflection_angle_std = np.std(measured_horizontal_deflection_angle, axis=0)
# print(horizontal_deflection_angle_std)

milliwatts_servo_mean = np.mean(milliwatts_servo, axis=0)
milliwatts_servo_std = np.std(milliwatts_servo, axis=0)


#find line of best fit
a_lateral, b_lateral = np.polyfit(deflection_force, horizontal_deflection_angle_mean, 1)
a_amp_lateral, b_amp_lateral = np.polyfit(deflection_force, milliwatts_servo_mean, 1)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(deflection_force, a_lateral*deflection_force+b_lateral, color='red', linestyle='-', linewidth=2, alpha=0.5, label='measured')

ax1.plot(deflection_force, simulated_horizontal_deflection_angle, marker='o', linestyle='--', color='orange', linewidth=2, alpha=0.5, label='simulated')

ax1.errorbar(deflection_force,
             horizontal_deflection_angle_mean,
             yerr=horizontal_deflection_angle_std, fmt='.k', color=color)
ax1.set_ylim([0, 90])
ax1.set_xlabel('Load force [N]', fontweight="bold")
ax1.set_ylabel('Horizontal deflection angle [deg]', color=color, fontweight='bold')
ax1.xaxis.labelpad=-1
ax1.legend(loc='lower right', frameon=False, handlelength=0.9, labelspacing=0.06)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(deflection_force, a_amp_lateral*deflection_force+b_amp_lateral, color=color, linestyle='-', linewidth=2, alpha=0.5, label='measured')
ax2.errorbar(deflection_force,
             milliwatts_servo_mean,
             yerr=milliwatts_servo_std, fmt='.k', color=color)

ax2.set_ylim([0, 1])
ax2.set_ylabel('Power [W]', color=color)
ax2.xaxis.labelpad=-6
ax2.legend(loc=1, bbox_to_anchor=(1, 0.15), frameon=False, handlelength=0.9, labelspacing=0.06)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped

# plt.show()
# plt.grid(linestyle='--')
# plt.grid(True)

fig.set_size_inches(w=2, h=1.8)
plt.subplots_adjust(left=0.2,  bottom=0.17, right=0.8, top=0.98,
        wspace=0, hspace=0.1)

plt.savefig("fig_lateral_load_amp.pdf")
plt.show()