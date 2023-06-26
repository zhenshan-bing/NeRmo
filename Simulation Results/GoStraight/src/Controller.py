import numpy as np
import math

from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.foreLeg import ForeLegM
from LegModel.hindLeg import HindLegM

class MouseController(object):
	"""docstring for MouseController"""
	def __init__(self, fre, time_step, spine_angle, stride_length_scale_factor, duty_factor):
		super(MouseController, self).__init__()
		PI = np.pi
		self.curStep = 0# Spine
		self.stride_length_scale_factor = stride_length_scale_factor
		self.duty_factor = duty_factor
		
		# Spine A = 0
		#self.turn_F = 0*PI/180
		#self.turn_H = 8*PI/180
		# Spine A = 20
		self.turn_F = 0*PI/180
		self.turn_H = 12*PI/180
		self.pathStore = LegPath()
		# [LF, RF, LH, RH]
		# --------------------------------------------------------------------- #
		# self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
		# self.period = 3/2
		# self.fre_cyc = fre
		# self.SteNum = 36							#32 # Devide 2*PI to multiple steps
		# self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		# Trot
		# self.phaseDiff = [PI, 0, 0, PI]
		# self.phaseDiff = [PI, 0, PI, 0]
		# self.phaseDiff = [PI, PI, 0, 0]
		# lateral sequence walk
		self.phaseDiff = [0.63*2*PI, 0.14*2*PI, 0, 0.54*2*PI]
		# self.phaseDiff = [0.63 * 2 * PI, 0.14 * 2 * PI, 0, 0.54 * 2 * PI]
		# self.phaseDiff = [0, PI, PI*1/2, PI*3/2]# Trot
		# self.phaseDiff = [0, PI*1/2, PI * 1 / 4, PI * 3 / 4]
		self.period = 2/2
		self.fre_cyc = fre#1.25#0.80
		self.SteNum = int(1/(time_step*self.fre_cyc))  #
		# print("----> ", self.SteNum)
		self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		self.spine_A =2*spine_angle#10 a_s = 2theta_s
		# print("angle --> ", spine_angle)#self.spine_A)
		self.spine_A = self.spine_A*PI/180
		# --------------------------------------------------------------------- #
		fl_params = {'lr0':0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295, 
			'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145,'alpha':23*np.pi/180}
		self.fl_left = ForeLegM(fl_params)
		self.fl_right = ForeLegM(fl_params)
        # --------------------------------------------------------------------- #
		hl_params = {'lr0':0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317, 
			'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205,'alpha':73*np.pi/180}
		self.hl_left = HindLegM(hl_params)
		self.hl_right = HindLegM(hl_params)
		# --------------------------------------------------------------------- #
		self.stepDiff = [0,0,0,0]
		for i in range(4):
			self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i]/(2*PI))
		self.stepDiff.append(int(self.SteNum * self.spinePhase/(2*PI)))
		self.trgXList = [[],[],[],[]]
		self.trgYList = [[],[],[],[]]

	def getLegCtrl(self, leg_M, curStep, leg_ID, stride_length_scale_factor, duty_factor):
		curStep = curStep % self.SteNum
		turnAngle = self.turn_F
		leg_flag = "F"
		if leg_ID > 1:
			leg_flag = "H"
			turnAngle = self.turn_H

		radian_swing = 2 * math.pi * (1 - duty_factor)

		radian_stance = 2 * math.pi*duty_factor
		# print("radian_swing,radian_stance",radian_swing,radian_stance)
		swing_stance_state = 10
		radian = 2*np.pi * curStep/self.SteNum
		if radian < radian_swing:
			radian = radian / radian_swing * math.pi
			# print("swing")
			swing_stance_state = 0
		else:
			radian = math.pi + (radian - radian_swing) / radian_stance * math.pi
			# print("stance")
			swing_stance_state = 1
		#currentPos = self.pathStore.getRectangle(radian, leg_flag)
		currentPos = self.pathStore.getOvalPathPoint(radian, leg_flag, self.period, stride_length_scale_factor)
		trg_x = currentPos[0]
		trg_y = currentPos[1]
		self.trgXList[leg_ID].append(trg_x)
		self.trgYList[leg_ID].append(trg_y)

		tX = math.cos(turnAngle)*trg_x - math.sin(turnAngle)*trg_y;
		tY = math.cos(turnAngle)*trg_y + math.sin(turnAngle)*trg_x;
		qVal = leg_M.pos_2_angle(tX, tY)
		return qVal, swing_stance_state

	def getSpineVal(self, spineStep, duty_factor):
		temp_step = int(spineStep)# / 5)
		# radian_swing = 2 * math.pi * (1 - duty_factor)
		# radian_stance = 2 * math.pi * duty_factor
		radian_swing = 2*math.pi-self.spinePhase
		radian_stance = self.spinePhase
		radian = 2*np.pi * temp_step/self.SteNum
		if radian < radian_swing:
			radian = radian / radian_swing * math.pi
			# print("swing")
			swing_stance_state = 0
		else:
			radian = math.pi + (radian - radian_swing) / radian_stance * math.pi
			# print("stance")
			swing_stance_state = 1

		return self.spine_A * math.cos(radian - 2 * math.pi * (1 - duty_factor))

		# radian = 2 * np.pi * temp_step / self.SteNum
		# return self.spine_A*math.cos(radian - self.spinePhase)
		#spinePhase = 2*np.pi*spineStep/self.SteNum
		#return self.spine_A*math.sin(spinePhase)

	def runStep(self):
		foreLeg_left_q, swing_stance_state_fl = self.getLegCtrl(self.fl_left,
			self.curStep + self.stepDiff[0], 0, self.stride_length_scale_factor, self.duty_factor)
		foreLeg_right_q, swing_stance_state_fr = self.getLegCtrl(self.fl_right,
			self.curStep + self.stepDiff[1], 1, self.stride_length_scale_factor, self.duty_factor)
		hindLeg_left_q, swing_stance_state_hl = self.getLegCtrl(self.hl_left,
			self.curStep + self.stepDiff[2], 2, self.stride_length_scale_factor, self.duty_factor)
		hindLeg_right_q, swing_stance_state_hr = self.getLegCtrl(self.hl_right,
			self.curStep + self.stepDiff[3], 3, self.stride_length_scale_factor, self.duty_factor)

		spineStep = self.curStep #+ self.stepDiff[4]
		spine = self.getSpineVal(spineStep, self.duty_factor)
		# spine = 0
		self.curStep = (self.curStep + 1) % self.SteNum
		# print(self.curStep)

		ctrlData = []
		swing_stance_states = []
		swing_stance_states.append(swing_stance_state_fl)
		swing_stance_states.append(swing_stance_state_fr)
		swing_stance_states.append(swing_stance_state_hl)
		swing_stance_states.append(swing_stance_state_hr)

		#foreLeg_left_q = [1,0]
		#foreLeg_right_q = [1,0]
		#hindLeg_left_q = [-1,0]
		#hindLeg_right_q = [-1,0]
		ctrlData.extend(foreLeg_left_q)
		ctrlData.extend(foreLeg_right_q)
		ctrlData.extend(hindLeg_left_q)
		ctrlData.extend(hindLeg_right_q)
		for i in range(3):
			ctrlData.append(0)
		ctrlData.append(spine)
		return ctrlData, swing_stance_states

	def runStepInit(self, timestep):
		foreLeg_left_q_first_timestep, _ = self.getLegCtrl(self.fl_left,
														self.stepDiff[0], 0, self.stride_length_scale_factor, self.duty_factor)
		# FR
		foreLeg_right_q_first_timestep, _ = self.getLegCtrl(self.fl_right,
														 self.stepDiff[1], 1, self.stride_length_scale_factor, self.duty_factor)
		# HL
		hindLeg_left_q_first_timestep, _ = self.getLegCtrl(self.hl_left,
														self.stepDiff[2], 2, self.stride_length_scale_factor, self.duty_factor)
		# HR
		hindLeg_right_q_first_timestep, _ = self.getLegCtrl(self.hl_right,
														 self.stepDiff[3], 3, self.stride_length_scale_factor, self.duty_factor)
		spine_q_first_timestep = self.getSpineVal(0, self.duty_factor)

		ctrlData = []

		# foreLeg_left_q = [1,0]
		# foreLeg_right_q = [1,0]
		# hindLeg_left_q = [-1,0]
		# hindLeg_right_q = [-1,0]

		ctrlData.extend(np.array(foreLeg_left_q_first_timestep) / 500 * timestep)
		ctrlData.extend(np.array(foreLeg_right_q_first_timestep) / 500 * timestep)
		ctrlData.extend(np.array(hindLeg_left_q_first_timestep) / 500 * timestep)
		ctrlData.extend(np.array(hindLeg_right_q_first_timestep) / 500 * timestep)
		spine = spine_q_first_timestep / 500 * timestep
		# spine = 0
		for i in range(3):
			ctrlData.append(0)
		ctrlData.append(spine)
		return ctrlData
		