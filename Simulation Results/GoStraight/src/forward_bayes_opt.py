from ToSim import SimModel
from Controller import MouseController
import numpy as np
import csv
import os
from bayes_opt import BayesianOptimization

# --------------------
RUN_TIME_LENGTH = 20
if __name__ == '__main__':
	duty_factor_gs = [0.5] # ideal trot: 0.5; walking trot: 0.6; lateral sequence walk: 0.68
	spine_angle_gs = [0, 2, 4, 8, 16]
	time_step = 0.0025
	run_steps_num = int(RUN_TIME_LENGTH / time_step)

	opt_max = []
	fre_scatter = []
	velocity_scatter = []
	best_model = None
	max_vel = 0
	model_num = 1

	pbounds = {'fre': (0.8, 1), 'stride_length_scale_factor': (0.8, 1)}
	for n_duty_factor in range(len(duty_factor_gs)):
		duty_factor = duty_factor_gs[n_duty_factor]
		for n_spine in range(len(spine_angle_gs)):
			spine_angle = spine_angle_gs[n_spine]
			INFO = []
			def black_box(fre, stride_length_scale_factor):
				theMouse = SimModel(os.getcwd()+'/Simulation Results/GoStraight/models/dynamic_4l_t3.xml')
				theController = MouseController(fre, time_step, spine_angle, stride_length_scale_factor, duty_factor)

				for i in range(100):
					ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
					theMouse.runStep(ctrlData, time_step)

				for i in range(500):
					ctrlData = theController.runStepInit(i)
					theMouse.runStep(ctrlData, time_step)

				theMouse.initializing()
				for i in range(run_steps_num):
					tCtrlData,_ = theController.runStep()
					ctrlData = tCtrlData
					theMouse.runStep(ctrlData, time_step)

				velocity = [info["velocity"] for info in theMouse.episode_step_infos]
				mean_velocity = np.mean(velocity[1000:])
				info = {
					"fre": fre,
					"stride_length_scale_factor": stride_length_scale_factor,
					"spine_angle": spine_angle,
					"mean_velocity": mean_velocity,
				}
				print("fre", fre)
				print("duty_factor", duty_factor)
				print("spine_angle", spine_angle)
				print("stride_length_factor", stride_length_scale_factor)
				print("mean_velocity", mean_velocity)
				INFO.append(info)
				return mean_velocity

			optimizer = BayesianOptimization(f=black_box, pbounds=pbounds, random_state=1)
			optimizer.maximize(init_points=10, n_iter=100)

			opt_max.append({'fre': optimizer.max['params']['fre'],
							'stride_length_scale_factor': optimizer.max['params']['stride_length_scale_factor'],
							'spine_angle': spine_angle,
							'max_velocity': optimizer.max['target'],
							'duty_factor':duty_factor})

		with open('forward_bayes_opt_df_'+str(duty_factor)+'.csv', mode='w') as csv_file:
			fieldnames = ['fre','stride_length_scale_factor','spine_angle', 'max_velocity','duty_factor']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			for i in range(len(opt_max)):
				writer.writerow(opt_max[i])
