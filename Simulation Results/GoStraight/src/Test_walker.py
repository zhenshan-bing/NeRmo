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
from time import sleep

import numpy as np
import math
from time import sleep
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import sys, getopt
import random

class Motion_Module:
    def __init__(self, vis_flag, cur_normal, cur_spine_mode, cur_gait_mode, the_fre):
        self.model_name = "dynamic_4l_t3_empty.xml"
        self.model_path = "../models/"+self.model_name
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)
        self.vis_flag = vis_flag
        if self.vis_flag:
            self.viewer = MjViewer(self.sim)

        self.fixPoint = "body_ss"
        self.movePath = [[],[],[]]
        self.moveVel = []
        self.run_time = 5000
        self.dead_time = 1000
        self.init_mouse_variables(cur_normal, cur_spine_mode, cur_gait_mode, the_fre)
        self.init_controllers()
        self.main()

    def init_mouse_variables(self, cur_normal, cur_spine_mode, cur_gait_mode, the_fre):
        fre = the_fre
        scale = cur_normal
        self.trg_spine_mode = cur_spine_mode
        self.trg_vel = 0.52*fre*scale
        gaie_mode = cur_gait_mode
        self.gait_parameters2 = Gait_Parameters(fre)
        self.mouse_parameters = Mouse_Parameters()
        if gaie_mode ==0:
            self.general_st_parameters2 = self.gait_parameters2.st_trot_parameters
        elif gaie_mode == 1:
            self.general_st_parameters2 = self.gait_parameters2.st_trot_parameters1
        else:
            self.general_st_parameters2 = self.gait_parameters2.walk_trot_parameters
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
        self.turn_rate = 0.0
        self.buttons = [0]*4
        self.prev_buttons = [0]*4
        self.leg_n = 0

    def motion_node(self):
        # Initialize the node
        self.fsm.timer.reset_times()
        sleep(0.002)
        
        self.vel_in = self.trg_vel
        self.turn_rate = 0

        # Subscribe to the ps4 controller
        for i in range(self.dead_time):
            q_legs = [0.42,1,0.42,1,-0.15,-0.66,-0.15,-0.66]
            vel_val = self.gen_messages(q_legs, 0, 0)

        start_steps = 500
        for i in range(self.run_time + start_steps):
            vel = self.vel_in * np.ones((4,))
            leg_states, leg_timings, norm_time = self.fsm.run_state_machine()
 
            # Steps of the full controller to generate values
            self.spine_mode = 0
            target_leg_positions, q_legs, q_spine = self.leg_controller.run_controller(leg_states, leg_timings, norm_time, vel, self.turn_rate, self.spine_mode, self.offset_mode)
            self.spine_mode = self.trg_spine_mode
            _, _, q_spine = self.leg_controller.run_controller(leg_states, leg_timings, norm_time, vel, self.turn_rate, self.spine_mode, self.offset_mode)
            q_spine *= 1
          
            q_tail = self.tail_extension(norm_time, self.vel_in)

            vel_val = self.gen_messages(q_legs, q_spine, q_tail)
            if i >= start_steps:
                tData = self.sim.data.get_site_xpos(self.fixPoint)
                self.moveVel.append(vel_val)
                for i in range(3):
                    self.movePath[i].append(tData[i])

    def tail_extension(self, timing: float,vel: float, offset=0, scaling=0.5) -> float:
        # THis function helps extend the spine stride during gait.
        # Timing value: is normalized time value [0,1]
        scale = min(4.5*np.abs(vel)**2, scaling)
        q_tail = scale*np.cos(2*np.pi*timing+offset)
        return q_tail

    def oriention_data_store(self):
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
        
        return framelinvel

    def gen_messages(self, q_legs, q_spine, q_tail):
        vel_x, vel_y, vel_z = self.oriention_data_store()
        vel_val = math.sqrt(vel_x*vel_x + vel_y*vel_y)

        q_values = np.concatenate((q_legs,np.array(([q_tail,0,0,q_spine]))))
        q_values.astype(dtype=np.float32)
        self.sim.data.ctrl[:] = q_values
        self.sim.step()
        if self.vis_flag:
            self.viewer.render() 
        return vel_val

    def main(self):
        start_time = time.time()
        self.motion_node()
        time_cost = self.run_time * 0.002

        plt.plot(self.movePath[0], self.movePath[1])
        plt.show()
        start_p = [self.movePath[0][0], self.movePath[1][0]]
        end_p = [self.movePath[0][-1], self.movePath[1][-1]]
        dis = math.sqrt((end_p[0]-start_p[0])*(end_p[0]-start_p[0])
            + (end_p[1]-start_p[1])*(end_p[1]-start_p[1]))

        print(">>>>>>>>>>>> Sim velocity: ", dis/time_cost)


if __name__ == "__main__":
    err_flag = 0

    cur_spine_mode = 0
    cur_gait_mode = 0
    cur_normal = 0.8
    the_fre = 0.8

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "n:s:w:f:")
    #n (normal rate):  a float [0, 1]
    #s (spine mode): 0: only leg, 1: with spine
    #w (walk rate): 0: trt, 1: walk, 2: lat
    gaits_mode = {"trt": 0, "walk": 1, "lat": 2}
    #f (frequency):  a float [0, 1.5]
    for opt, arg in opts:
        if opt in ['-n']:
            cur_normal = float(arg)
            if abs(cur_normal) > 1:
                err_flag = 1
        elif opt in ['-s']:
            cur_spine_mode = int(arg)
            if cur_spine_mode != 0 and cur_spine_mode != 1:
                err_flag = 1
        elif opt in ['-w']:
            if arg in gaits_mode.keys():
                cur_gait_mode = gaits_mode[arg]
            else:
                err_flag = 1
        elif opt in ['-f']:
            fre = float(arg)
            if abs(fre) > 1:
                err_flag = 1
        else:
            err_flag = 1

    if err_flag:
        print("Error with parameters !!!")
    else:
        vis_flag = True
        Motion_Module(vis_flag, cur_normal, cur_spine_mode, cur_gait_mode, the_fre)

