#!/usr/bin/env python3
# *******************************************************
# Type: Motion controller
# 
# Motion controller for the mouse
# Handles state machine and low level spine and leg control.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 26.04.2021
# *********************************************************

# Import the leg and motionplanner modules
from mouse_controller.leg_controller import Leg_Controller
from mouse_controller.state_machine.leg_state_machine import Leg_State_Machine
from mouse_controller.mouse_parameters_dir import Gait_Parameters, Mouse_Parameters
from mujoco_py import load_model_from_path, MjSim, MjViewer

from time import sleep
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import sys, getopt

from get_all_angles import *
from func import *

class Motion_Module:
    def __init__(self, vis_flag):
        self.time_step = 0.01

        self.model_name = "dynamic_4l_t3.xml"
        self.model_path = "models/"+self.model_name
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)
        self.sim_state = self.sim.get_state()
        self.sim.set_state(self.sim_state)
        self.visual = vis_flag
        if self.visual:
            self.viewer = MjViewer(self.sim)

        self.current_pos = []
        self.current_pos.append([0, 0, -5e-02])
        self.current_pos.append([0, 0, -5e-02])
        self.current_pos.append([0, -30e-3, -5.25e-02])
        self.current_pos.append([0, -30e-3, -5.25e-02])
        #self.q_values_init = [0.42,1,0.42,1,-0.15,-0.66,-0.15,-0.66]
        self.q_values_init = [0.7,0.75,0.7,0.75,-0.3,-0.66,-0.3,-0.66]
        
        self.angle_value = {"Pitch":[], "Roll":[], "Yaw":[]}
        self.imu_data_list = []
        self.time_x = []
        self.time_counter = 0
        self.dead_time = 500
        self.run_time = 500
        
        self.init_mouse_variables()
        self.init_controllers()
        #self.main()


    def init_mouse_variables(self):
        self.gait_parameters2 = Gait_Parameters()
        self.mouse_parameters = Mouse_Parameters()
        self.general_st_parameters2 = self.gait_parameters2.st_trot_parameters
        self.front_leg_parameters2 = self.gait_parameters2.st_trot_param_f
        self.rear_leg_parameters2 = self.gait_parameters2.st_trot_param_r

    def init_controllers(self):
        # Initialize the key components of the motion module
        # Spine modes:
        # 0: purely turning motion, nothing else
        # 1: turning motion + spine modulation
        # 2: turning motion + balance mode (balance mode for 2 and 3 leg contact)
        self.spine_mode = 0
        self.offset_mode = False
        self.balance_mode = True
        self.fsm = Leg_State_Machine(self.general_st_parameters2)
        self.leg_controller = Leg_Controller(self.gait_parameters2, self.mouse_parameters)
        self.vel_in = 0.0

    def motion_node(self, test_case, trg_vel):
        # main starter method
        # print("Starting mouse node")
        dry = 0

        # Initialize the node
        self.fsm.timer.reset_times()
        sleep(0.002)
        
        self.vel_in = 0
        vel_step = trg_vel / 100
        for i in range(self.run_time):
            leg_states, leg_timings, norm_time = self.fsm.run_state_machine()
            q_tail = 0

            vel = 0 * np.ones((4,))
            q_legs, q_spine = self.leg_balance_tester(test_case, leg_timings, norm_time, vel, self.spine_mode, self.offset_mode)
            
            if abs(self.vel_in) < abs(trg_vel):
                self.vel_in += vel_step
            q_legs[test_case*2] += self.vel_in*3
            #print(q_spine)
            self.gen_messages(q_legs, q_spine, q_tail, True)
    
    def leg_balance_tester(self, test_case, leg_timings, norm_time, vel, spine_mode, offset_mode):
        """ Balance mode to test the spines ability to compensate COM shifts. This is purely a test mode. """
        
        leg_states = np.array([1,1,1,1])
        leg_states[test_case] = 0
        turn_rate = 0
        tl, ql, q_spine = self.leg_controller.run_controller_pos(leg_states, self.current_pos, norm_time, vel, turn_rate, spine_mode, offset_mode)
        # Forward
        q_legs = self.q_values_init.copy()
        return (q_legs, q_spine*3)


    def oriention_data_store(self):
        self.time_x.append(self.time_counter*self.time_step)
        self.time_counter += 1

        data = self.sim.data.sensordata
        servo_pos_leg = data[:8]
        servo_pos_aux = data[8:12]
        contact_sensors = data[12:16]
        imu_sensor = data[16:]
        framepos = imu_sensor[:3]
        framequat = imu_sensor[3:7]
        framelinvel = imu_sensor[7:10]
        accelerometer = imu_sensor[10:13]
        gyro = imu_sensor[13:]
        
        roll, pitch, yaw = quaternion_to_euler_angle_vectorized(framequat)

        return pitch, roll, yaw
        

    def gen_messages(self, q_legs, q_spine, q_tail, data_flag):
        q_values = np.concatenate((q_legs,np.array(([q_tail,0,0,q_spine]))))
        q_values.astype(dtype=np.float32)
        step_length = int(self.time_step/0.002)
        for i in range(step_length):
            self.sim.data.ctrl[:] = q_values
            self.sim.step()
            if self.visual:
                self.viewer.render()

        if data_flag:
            pitch, roll, yaw = self.oriention_data_store()
            self.angle_value["Pitch"].append(pitch)
            self.angle_value["Roll"].append(roll)
            self.angle_value["Yaw"].append(yaw)
            
    def idle_motion(self, test_case, data_flag):
        q_legs = self.q_values_init.copy()
        for i in range(self.dead_time):
            leg_states, leg_timings, norm_time = self.fsm.run_state_machine()
            vel = 0 * np.ones((4,))
            q_legs, q_spine = self.leg_balance_tester(test_case, leg_timings, norm_time, vel, self.spine_mode, self.offset_mode)
            self.gen_messages(q_legs, q_spine, 0, data_flag)


    def main(self, cur_spine_mode, cur_leg_id, flting_angle):
        # Non-Balance --> 0        Balance --> 2
        self.spine_mode = cur_spine_mode

        # This is for balance mode. Uncomment if you want to use that mode
        # 0 --> Fore Left   1 --> Fore Right
        # 2 --> Hind Left   3 --> Hind Right
        test_case = cur_leg_id
        trg_vel = flting_angle
        self.idle_motion(test_case, False)
        self.idle_motion(test_case, True)
        self.motion_node(test_case, trg_vel)

if __name__ == "__main__":
    cur_spine_mode = 0
    cur_leg_id = 0
    flting_angle = -0.3
    err_flag = 0

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "m:l:a:")
    for opt, arg in opts:
        if opt in ['-m']:
            if arg == 'sb':
                cur_spine_mode = 2
            else:
                cur_spine_mode = 0
        elif opt in ['-l']:
            cur_leg_id = int(arg)
            if cur_leg_id > 3:
                err_flag = 1
        elif opt in ['-a']:
            flting_angle = float(arg)
            if abs(flting_angle) > 0.7:
                err_flag = 1
        else:
            err_flag = 1

    if err_flag:
        print("Error with parameters !!!")
    else:
        cur_robot = Motion_Module(True)
        cur_robot.main(cur_spine_mode, cur_leg_id, flting_angle)
