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
from func.cam_info import GetCommand
from time import sleep

import mujoco_py
from mujoco_py import load_model_from_path, MjSim
from mujoco_py import  MjViewer, MjRenderContext, MjRenderContextOffscreen

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time, sys, getopt

#Note: Change state_machine/time_machine to fix the real time as simulation time
#Note: To fix spine behavior, change hind leg center point from (0,-0.05) to (0,-0.055)

class Motion_Module:
    def __init__(self, vis_flag, cur_spine_mode):
        self.model_name = "dynamic_4l_t3_maze.xml"
        self.model_path = "../models/"+self.model_name
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)
        self.vis_flag = vis_flag

        self.camera_name = "maze_camera"
        self.camera_id = self.model.camera_name2id(self.camera_name)
        self.RenderContext =  MjRenderContextOffscreen(self.sim, 0)
        if self.vis_flag:
            self.viewer = MjViewer(self.sim)
        self.cv_com = GetCommand(cur_spine_mode)

        self.fixPoint = "body_ss"
        self.movePath = [[],[],[]]
        self.moveVel = []
        self.run_time = 50000
        self.dead_time = 500
        self.cam_step = 0
        self.init_mouse_variables(cur_spine_mode)
        self.init_controllers()
        self.main()

    def init_mouse_variables(self, cur_spine_mode):
        fre = 1
        self.trg_spine_mode = cur_spine_mode
        self.trg_vel = 0.45
        gaie_mode = 0
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
        # main starter method
        # Initialize the node
        self.fsm.timer.reset_times()
        sleep(0.002)
        
        self.vel_in = self.trg_vel
        self.turn_rate = 0

        # Subscribe to the ps4 controller
        init_flag = True
        for i in range(self.dead_time):
            q_legs = [0.42,1,0.42,1,-0.15,-0.66,-0.15,-0.66]
            vel_val = self.gen_messages(q_legs, 0, 0, init_flag)

        start_steps = 500
        init_flag = False
        for i in range(self.run_time + start_steps):
            if i < 100 and self.trg_spine_mode:
                self.turn_rate = 1
            vel = self.vel_in * np.ones((4,))
            leg_states, leg_timings, norm_time = self.fsm.run_state_machine()

            self.spine_mode = 0
            target_leg_positions, q_legs, q_spine = self.leg_controller.run_controller(leg_states, leg_timings, norm_time, vel, self.turn_rate, self.spine_mode, self.offset_mode)
            self.spine_mode = self.trg_spine_mode
            _, _, q_spine = self.leg_controller.run_controller(leg_states, leg_timings, norm_time, vel, self.turn_rate, self.spine_mode, self.offset_mode)
            if self.trg_spine_mode:
                q_spine *= 1
            else:
                q_spine = 0
            q_tail = 0

            vel_val = self.gen_messages(q_legs, q_spine, q_tail, init_flag)
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

    def gen_messages(self, q_legs, q_spine, q_tail, init_flag):
        vel_x, vel_y, vel_z = self.oriention_data_store()
        vel_val = math.sqrt(vel_x*vel_x + vel_y*vel_y)

        q_values = np.concatenate((q_legs,np.array(([q_tail,0,0,q_spine]))))
        q_values.astype(dtype=np.float32)
        self.sim.data.ctrl[:] = q_values

        self.sim.step()
        if self.vis_flag:
            self.viewer.render()
            self.cam_step += 1

            if self.cam_step > 100:
                c_h =600
                c_w =600
                self.RenderContext.render(c_h, c_w, self.camera_id)
                img = self.RenderContext.read_pixels(height=c_h, width=c_w, depth=True)
                trg_side = self.cv_com.image_process(img[0][...,::-1], init_flag)
                self.turn_rate = trg_side*1
                self.cam_step  = 0

        return vel_val

    def main(self):
        start_time = time.time()
        self.motion_node()
        time_cost = self.run_time * 0.002

if __name__ == "__main__":
    vis_flag = True
    cur_spine_mode = 0
    err_flag = 0

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "m:", ["mode="])
    for opt, arg in opts:
        if opt in ['-m', '--mode']:
            if arg == 'sb':
                cur_spine_mode = 1
            else:
                cur_spine_mode = 0
        else:
            err_flag = 1

    if err_flag:
        print("Error with parameters !!!")
    else:
        Motion_Module(vis_flag, cur_spine_mode)
