from ToSim import SimModel
from TurningController import MouseController
import numpy as np
import circle_fit as cf
from math import pi
import math
import csv
import os
from bayes_opt import BayesianOptimization

# --------------------
RUN_TIME_LENGTH = 20
if __name__ == '__main__':
	duty_factor = 0.5 # ideal trot: 0.5; walking trot: 0.6; lateral sequence walk: 0.68
	fre_vector = [0.3, 0.5, 0.7, 0.9]
	time_step = 0.0025
	run_steps_num = int(RUN_TIME_LENGTH / time_step)

	opt_max = []
	fre_scatter = []
	velocity_scatter = []
	best_model = None
	max_vel = 0
	model_num = 1

	pbounds = {'spine_angle': (-8, 0), 'left_stride_length_scale_factor': (0.6, 1), 'right_stride_length_scale_factor': (0.2, 0.6)}

	for n_fre in range(4):
		fre = fre_vector[n_fre]
		INFO = []
		def black_box(spine_angle, left_stride_length_scale_factor, right_stride_length_scale_factor):
			theMouse = SimModel(os.getcwd()+'/Simulation Results/turn/models/dynamic_4l_t3.xml')
			theController = MouseController(fre, time_step, spine_angle, left_stride_length_scale_factor,
											right_stride_length_scale_factor, right_stride_length_scale_factor,
											left_stride_length_scale_factor, duty_factor)

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
				"fl_scale_factor": left_stride_length_scale_factor,
				"fr_scale_factor": right_stride_length_scale_factor,
				"hr_scale_factor": right_stride_length_scale_factor,
				"hl_scale_factor": left_stride_length_scale_factor,
				"fitted_turning_radius": math.copysign(1, xc) * r,
				"average_angular_velocity": travelled_rad / 20,
				"average_velocity": travelled_rad / 20 * r,
				"speed": np.mean(speed),
				"fitted_s": s
			}
			INFO.append(info)
			return -r


		optimizer = BayesianOptimization(f=black_box, pbounds=pbounds, random_state=1,allow_duplicate_points=True)
		optimizer.maximize(init_points=10, n_iter=100)
		opt_max.append({'min_turning_radius': -optimizer.max['target'],
						'fre': fre,
						'spine_angle': optimizer.max['params']['spine_angle'],
						'left_stride_length_scale_factor': optimizer.max['params']['left_stride_length_scale_factor'],
						'right_stride_length_scale_factor': optimizer.max['params']['right_stride_length_scale_factor'],
						'duty_factor':duty_factor})

	with open('bayes_opt_min_turn_radius_df_'+str(duty_factor)+'.csv', mode='w') as csv_file:
		fieldnames = ['min_turning_radius', 'fre', 'spine_angle',
					  'left_stride_length_scale_factor','right_stride_length_scale_factor','duty_factor']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(opt_max)):
			writer.writerow(opt_max[i])

