from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math
import glfw

class SimModel(object):

	"""docstring for SimModel"""
	def __init__(self, modelPath, ):
		super(SimModel, self).__init__()
		self.model = load_model_from_path(modelPath)
		self.sim = MjSim(self.model)

		self.sim_state = self.sim.get_state()
		self.sim.set_state(self.sim_state)
		self.legPosName = [
			["router_shoulder_fl", "foot_s_fl"],
			["router_shoulder_fr", "foot_s_fr"],
			["router_hip_rl", "foot_s_rl"],
			["router_hip_rr", "foot_s_rr"]]
		self.fixPoint = "body_ss"#"neck_ss"
		self.legRealPoint_x = [[],[],[],[]]
		self.legRealPoint_y = [[],[],[],[]]
		self.movePath = [[],[],[]]

		self.info = {}
		self.previous_mouse_xy_position = None
		self.current_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)
		self.desired_direction = [0, -1]
		self.posx=[]
		self.posy = []
		self.velocity = []
		self.episode_step_infos = []
	def initializing(self):
		self.movePath = [[],[],[]]
		self.legRealPoint_x = [[],[],[],[]]
		self.legRealPoint_y = [[],[],[],[]]
		self.velocity = []
		self.episode_step_infos = []

	def close(self):
		glfw.destroy_window(self.viewer.window)
	def runStep(self, ctrlData, cur_time_step):
		# ------------------------------------------ #
		# ID 0, 1 left-fore leg and coil 
		# ID 2, 3 right-fore leg and coil
		# ID 4, 5 left-hide leg and coil
		# ID 6, 7 right-hide leg and coil
		# Note: For leg, it has [-1: front; 1: back]
		# Note: For fore coil, it has [-1: leg up; 1: leg down]
		# Note: For hide coil, it has [-1: leg down; 1: leg up]
		# ------------------------------------------ #
		# ID 08 is neck		(Horizontal)
		# ID 09 is head		(vertical)
		# ID 10 is spine	(Horizontal)  [-1: right, 1: left]
		# Note: range is [-1, 1]
		# ------------------------------------------ #
		step_num = int(cur_time_step/0.0025)
		self.sim.data.ctrl[:] = ctrlData
		self.previous_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)
		for i in range(step_num):
			self.sim.step()

		self.current_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)
		self._compute_velocities(cur_time_step)
		self.info = {
			**self.info,
			"velocity": self.velocity_in_desired_direction,
			"position": self.current_mouse_xy_position,
			"speed": self.speed,
			"foot_contacts": {
				"lf": self._has_contact(["foot_fl"]),
				"rf": self._has_contact(["foot_fr"]),
				"rh": self._has_contact(["foot_rr"]),
				"lh": self._has_contact(["foot_rl"])
			}
		}
		self.episode_step_infos.append(self.info)

	def get_sensor(self, name, dimensions=1, use_correct_sensors=None):
		if use_correct_sensors is None:
			use_correct_sensors = True #self.use_correct_sensors
		# reference: https://github.com/openai/mujoco-py/issues/193#issuecomment-458697015 + maintain own map from sensor name to sensor index
		name2index_map = {
            "m1_fl": 0,
            "m2_fl": 1,
            "m1_fr": 2,
            "m2_fr": 3,
            "m1_rl": 4,
            "m2_rl": 5,
            "m1_rr": 6,
            "m2_rr": 7,
            "m1_tail": 8,
            "neck": 9,
            "head": 10,
            "spine": 11,
            "fl_t1": 12,
            "fr_t1": 13,
            "rl_t1": 14,
            "rr_t1": 15,
            "com_pos": 16,
            "com_quat": 19,
            "com_vel": 23,
            "imu_acc": 26,
            "imu_gyro": 29,
            "fl_foot_pos": 32,
            "fr_foot_pos": 35,
            "rl_foot_pos": 38,
            "rr_foot_pos": 41,
        }

		if use_correct_sensors:
			sensor_index = name2index_map[name]
		else:
			sensor_index = self.sim.model.sensor_name2id(name)

		return self.sim.data.sensordata[sensor_index:sensor_index + dimensions].copy()

	def _compute_velocities(self, cur_time_step):
		self.desired_direction /= np.linalg.norm(self.desired_direction)
		displacement = self.current_mouse_xy_position - self.previous_mouse_xy_position
		distance_traveled = np.linalg.norm(displacement)
		self.speed = distance_traveled / cur_time_step
		displacement = self.current_mouse_xy_position - self.previous_mouse_xy_position
		self.displacement_in_desired_direction = np.dot(displacement, self.desired_direction)
		self.velocity_in_desired_direction = self.displacement_in_desired_direction / cur_time_step

	def _has_contact(self, geoms):
		# Inspired by: https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
		for i in range(self.sim.data.ncon):
			contact = self.sim.data.contact[i]
			# print(self.model.geom_id2name(contact.geom1),self.model.geom_id2name(contact.geom2))
			if self.model.geom_id2name(contact.geom1) in geoms or self.model.geom_id2name(contact.geom2) in geoms:

				return True
		return False
