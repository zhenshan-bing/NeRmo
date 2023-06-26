from ToSim import SimModel
from Controller import MouseController
import numpy as np
import csv
import os

# --------------------
RUN_TIME_LENGTH = 20
if __name__ == '__main__':
	fre_gs = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
	spine_angle_gs = [0, 2, 4, 8, 16]
	stride_length_scale_factor_gs = [0.2, 0.4, 0.6, 0.8, 1]
	duty_factor = 0.68
	time_step = 0.0025
	run_steps_num = int(RUN_TIME_LENGTH / time_step)

	INFO = []
	info = {}
	fre_scatter = []
	velocity_scatter = []
	best_model = None
	max_vel = 0
	model_num = 1

	for n_fre in range(len(fre_gs)):
		for n_sl in range(len(stride_length_scale_factor_gs)):
			for n_spine in range(len(spine_angle_gs)):
				spine_angle = spine_angle_gs[n_spine]
				fre = fre_gs[n_fre]
				stride_length_scale_factor = stride_length_scale_factor_gs[n_sl]

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
				fre_scatter.append(fre)
				velocity_scatter.append(mean_velocity)
				print("model: ", model_num)
				if max_vel < mean_velocity:
					best_model = model_num
					max_vel = mean_velocity
				info = {
					"fre": fre,
					"stride_length_scale_factor": stride_length_scale_factor,
					"spine_angle": spine_angle,
					"mean_velocity": mean_velocity,
					"duty_factor": duty_factor
				}
				print("fre", fre)
				print("stride_length_factor", stride_length_scale_factor)
				print("mean_velocity", mean_velocity)
				INFO.append(info)
				model_num += 1

	with open('forward_grid_search_df_'+str(duty_factor)+'_INFO.csv', mode='w') as csv_file:
		fieldnames = ['fre', 'stride_length_scale_factor', 'spine_angle', 'mean_velocity','duty_factor']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(INFO)):
			writer.writerow(INFO[i])
