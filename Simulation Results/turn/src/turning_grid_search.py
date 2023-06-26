from ToSim import SimModel
from TurningController import MouseController
import numpy as np
import circle_fit as cf
from math import pi
import math
import csv
import os

# --------------------
RUN_TIME_LENGTH = 20
if __name__ == '__main__':
	duty_factor = 0.5 # ideal trot: 0.5; walking trot: 0.6; lateral sequence walk: 0.68
	fre_gs = [0.3,0.5,0.7,0.9]
	time_step = 0.0025
	run_steps_num = int(RUN_TIME_LENGTH / time_step)
	for fre in fre_gs:
		model_num = 1
		left_leg_scale_factor_gs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		right_leg_scale_factor_gs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

		# spine angle <-0, turn right, left leg stride > right leg stride
		spine_angle_gs = [-8, - 4, -2, 0]
		with open('turning_grid_search_spine_turn_right_df_'+str(duty_factor)+'_INFO_f-'+str(fre)+'.csv', mode='w') as spine_turn_right_csv_file, \
				open('turning_grid_search_spine_turn_left_df_'+str(duty_factor)+'_INFO_f-'+str(fre)+'.csv', mode='w') as spine_turn_left_csv_file, \
				open('turning_grid_search_no_spine_turn_right_df_'+str(duty_factor)+'_INFO_f-'+str(fre)+'.csv', mode='w') as no_spine_turn_right_csv_file, \
				open('turning_grid_search_no_spine_turn_left_df_'+str(duty_factor)+'_INFO_f-'+str(fre)+'.csv', mode='w') as no_spine_turn_left_csv_file:
			fieldnames = ['fre', 'spine_angle', 'fl_scale_factor', 'fr_scale_factor', 'hr_scale_factor', 'hl_scale_factor',
						  'fitted_turning_radius', 'average_angular_velocity', 'average_velocity', 'speed', 'fitted_s','duty_factor']
			spine_turn_right_csv_writer = csv.DictWriter(spine_turn_right_csv_file, fieldnames=fieldnames)
			spine_turn_right_csv_writer.writeheader()
			spine_turn_left_csv_writer = csv.DictWriter(spine_turn_left_csv_file, fieldnames=fieldnames)
			spine_turn_left_csv_writer.writeheader()
			no_spine_turn_right_csv_writer = csv.DictWriter(no_spine_turn_right_csv_file, fieldnames=fieldnames)
			no_spine_turn_right_csv_writer.writeheader()
			no_spine_turn_left_csv_writer = csv.DictWriter(no_spine_turn_left_csv_file, fieldnames=fieldnames)
			no_spine_turn_left_csv_writer.writeheader()

			for spine_angle in spine_angle_gs:
				for left_leg_scale_factor in left_leg_scale_factor_gs:
					for right_leg_scale_factor in right_leg_scale_factor_gs:
						if spine_angle * (left_leg_scale_factor - right_leg_scale_factor) <= 0:
							theMouse = SimModel(os.getcwd()+'/Simulation Results/turn/models/dynamic_4l_t3.xml')
							fl_scale_factor = left_leg_scale_factor
							fr_scale_factor = right_leg_scale_factor
							hr_scale_factor = right_leg_scale_factor
							hl_scale_factor = left_leg_scale_factor
							theController = MouseController(fre, time_step, spine_angle, fl_scale_factor, fr_scale_factor, hr_scale_factor, hl_scale_factor, duty_factor)
							for i in range(100):
								ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
								theMouse.runStep(ctrlData, time_step)

							# move foot end from initial position to desired trajectory
							for i in range(500):
								ctrlData = theController.runStepInit(i)
								theMouse.runStep(ctrlData, time_step)

							# simulation for 20 sec
							theMouse.initializing()
							for i in range(run_steps_num):
								tCtrlData,_ = theController.runStep()
								ctrlData = tCtrlData
								theMouse.runStep(ctrlData, time_step)

							# get travelled path
							position = [info["position"] for info in theMouse.episode_step_infos]
							x = []
							y = []
							for pos in position:
								# change signs for intuitive visualization like in simulation
								x.append(-pos[0])
								y.append(-pos[1])

							# calculate center point, radius and variance of fitted circle
							x_y_positions = np.column_stack([x, y])
							xc, yc, r, s = cf.least_squares_circle(x_y_positions)
							# calculate travelled rad
							if xc > 0:
								if -position[-1][1] - yc > 0:
									travelled_rad = math.atan2(-position[-1][1] - yc, xc + position[-1][0])
								else:
									travelled_rad = 2 * pi - abs(math.atan2(-position[-1][1] - yc, xc + position[-1][0]))
								theta_travelled = np.linspace(0, travelled_rad, 100)
								x_fit = xc - r * np.cos(theta_travelled)
								y_fit = yc + r * np.sin(theta_travelled)
							else:
								if -position[-1][1] - yc > 0:
									travelled_rad = math.atan2(-position[-1][1] - yc, -xc - position[-1][0])
								else:
									travelled_rad = 2 * pi - abs(math.atan2(-position[-1][1] - yc, -xc - position[-1][0]))
								theta_travelled = np.linspace(0, travelled_rad, 100)
								x_fit = xc + r * np.cos(theta_travelled)
								y_fit = yc + r * np.sin(theta_travelled)

							if xc > 0:
								if -position[0][1] - yc > 0:
									travelled_rad_init = math.atan2(-position[0][1] - yc, xc + position[0][0])
									theta_travelled_init = np.linspace(0, travelled_rad_init, 10)
									x_fit_init = xc - r * np.cos(theta_travelled_init)
									y_fit_init = yc + r * np.sin(theta_travelled_init)
								else:
									travelled_rad_init = math.atan2(-position[0][1] - yc, xc + position[0][0])
									theta_travelled_init = np.linspace(travelled_rad_init, 0, 10)
									x_fit_init = xc - r * np.cos(theta_travelled_init)
									y_fit_init = yc + r * np.sin(theta_travelled_init)

							else:
								if -position[0][1] - yc > 0:
									travelled_rad_init = math.atan2(-position[0][1] - yc, -xc - position[0][0])
									theta_travelled_init = np.linspace(0, travelled_rad_init, 10)
									x_fit_init = xc + r * np.cos(theta_travelled_init)
									y_fit_init = yc + r * np.sin(theta_travelled_init)
								else:
									travelled_rad_init = math.atan2(-position[0][1] - yc, -xc - position[0][0])
									theta_travelled_init = np.linspace(travelled_rad_init, 0, 10)
									x_fit_init = xc + r * np.cos(theta_travelled_init)
									y_fit_init = yc + r * np.sin(theta_travelled_init)

							travelled_rad = travelled_rad - travelled_rad_init
							speed = [info["speed"] for info in theMouse.episode_step_infos]
							fitted_turning_radius = np.sign(xc) * r
							average_angular_velocity = travelled_rad / 20
							average_velocity = travelled_rad / 20 * r
							info = {
								"fre": fre,
								"spine_angle": spine_angle,
								"fl_scale_factor": fl_scale_factor,
								"fr_scale_factor": fr_scale_factor,
								"hr_scale_factor": hr_scale_factor,
								"hl_scale_factor": hl_scale_factor,
								"fitted_turning_radius": math.copysign(1, xc) * r,
								"average_angular_velocity": travelled_rad / 20,
								"average_velocity": travelled_rad / 20 * r,
								"speed": np.mean(speed),
								"fitted_s": s,
							    "duty_factor": duty_factor
							}

							# with spine turn right
							if spine_angle < 0 and left_leg_scale_factor >= right_leg_scale_factor:
								spine_turn_right_csv_writer.writerow(info)
							# with spine turn left
							elif spine_angle > 0 and left_leg_scale_factor <= right_leg_scale_factor:
								spine_turn_left_csv_writer.writerow(info)
							# without spine
							elif spine_angle == 0 and left_leg_scale_factor > right_leg_scale_factor:
								no_spine_turn_right_csv_writer.writerow(info)
							elif spine_angle == 0 and left_leg_scale_factor < right_leg_scale_factor:
								no_spine_turn_left_csv_writer.writerow(info)
							elif spine_angle == 0 and left_leg_scale_factor == right_leg_scale_factor:
								pass
							else:
								raise AssertionError("Unexpected  turning!", spine_angle, fl_scale_factor,
													 fr_scale_factor, hr_scale_factor, hl_scale_factor)


							print(fre, fl_scale_factor, fr_scale_factor, hr_scale_factor, hl_scale_factor)
							print("model number: ", model_num)
							model_num += 1
