import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
import time

# *******************************************************
# Type: Model Script
# Model of the rear leg type 3. 
# The mathematical foundations are described via the 
# respective thesis sections. Type 3 legs avoid tendon 
# wrapping by clever tendon routing.
# This class stores a digital twin of the rear leg with all
# relevant leg parameters. 
# This model contains functions for computing the forward
# and inverse kinematics of the legs.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 04.02.2021
# *********************************************************

class Rear_Leg3:

    # Dictionary contains all relevant leg parameters for kinematic modelling
    # Constants: l1, l2, l3, l4, rp, d1, lr0 [initialized with]
    # Motor inputs: q1, q2
    # Velocities: q1_dot, q2_dot
    # Modelling variables: theta1, theta2
    # End point: yB, zB, yH, zH
    # Target end_point: yH_t, zH_t
    # Also contains all parameters of previous position (k-1) with subscript *_p
    leg_param = 0
    # Currently theta2_dot / theta2_dot_p not being updated/considered

    # If necessary servo motor constraints (currently not considered)
    constraints = {'q1_min': -np.pi/2, 'q1_max': np.pi/2,
                    'q2_min': -3/2*np.pi, 'q2_max': np.pi/2}


    def __init__(self, leg_params):
        self.leg_param = {'q1': 0, 'q2': 0,
                        'q1_p': 0, 'q2_p': 0,
                        'q1_dot':0, 'q2_dot':0,
                        'q1_dot_p': 0, 'q2_dot_p': 0,
                        'yB': 0, 'zB': 0,
                        'yB_p': 0, 'zB_p': 0,
                        'yB_dot': 0, 'zB_dot': 0,
                        'yB_dot_p': 0, 'zB_dot_p': 0,
                        'yH': 0, 'zH': 0,
                        'yH_p': 0, 'zH_p': 0,
                        'yH_dot': 0, 'zH_dot': 0,
                        'yH_dot_p': 0, 'zH_dot_p': 0,
                        'theta1': 0, 'theta2': 0, 'theta3': 20*np.pi/180,
                        'theta1_p': 0, 'theta2_p': 0, 'theta3_p': 20*np.pi/180,
                        'theta1_dot':0, 'theta2_dot':0, 'theta3_dot': 0,
                        'theta1_dot_p':0, 'theta2_dot_p':0, 'theta3_dot_p': 0} 
        for i, value in enumerate(leg_params):
            self.leg_param[value] = leg_params[value]

        #print("Constant import complete.")

        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        rp = self.leg_param['rp']
        lr0 = self.leg_param['lr0']

        self.q2_crit = ((lr0 - np.sqrt(l1**2+l2**2-2*l1*l2))/rp-0.05)

        # Right after initialization compute all necessary modelling values
        # Default position q1,q2 = (0,0)
        self.forward_kinematics()

        # Type 3 leg theta2 is a fixed value
        self.leg_param['theta2'] = np.pi/2 - np.arccos(self.leg_param['rp']/self.leg_param['d1'])

        #print("First pass FK complete.")

    def update_servo_pos(self, pos):
        # This method feeds back the measured real servo values into the model
        # INPUT: 
        #       pos: (2,) numpy vector   || current measured servo values
        # OUTPUT: 
        #       none

        # self.leg_param['q1'] = pos[0]
        # self.leg_param['q2'] = pos[3]
        self.leg_param['q1'] = (self.leg_param['q1'] + pos[0])/2
        self.leg_param['q2'] = (self.leg_param['q2'] + pos[1])/2

    def perform_kinematic_update(self, vel_B, time_step = 0.02):
        # Performs a full motion update for the front leg
        # Should be used during trajectory following 
        # Does a single forward kinematics pass
        # Then computes the next set of servo angles to follow velocity
        # INPUT: vel_B - array of size (2,) - velocity of end point as needed by trajectory
        # RETURN: Array of size (2,) - (q1, q2) 

        self.inverse_kinematics(vel_B, time_step)
        self.forward_kinematics()
        # Return the new q1,q2 parameters to be controlled
        return [self.leg_param['q1'], self.leg_param['q2']]

    # ******************* FORWARD KINEMATIC ***********************     
    def forward_kinematics(self):
        # This methods performs the full kinematic update given q1 and q2
        # INPUT: none
        # OUTPUT: (2,) numpy array of current end-effector position

        # Compute current values of the modelling variables
        self.compute_theta_1()

        # Run the forward kinematics chain here
        l1 = self.leg_param['l1']
        l3 = self.leg_param['l3']
        l4 = self.leg_param['l4']
        q1 = self.leg_param['q1']
        theta1 = self.leg_param['theta1']
        theta3 = self.leg_param['theta3']

        # Compute forward elements
        # yB = l1*np.sin(q1) + l2*np.sin(theta1 + q1) 
        # zB = -l1*np.cos(q1) - l2*np.cos(theta1 + q1)
        yH = l1*np.sin(q1) + l3*np.sin(theta1 + q1) - l4*np.sin(theta3 - theta1 - q1)
        zH = -l1*np.cos(q1) - l3*np.cos(theta1 + q1) - l4*np.cos(theta3 - theta1 - q1)

        self.leg_param['yH_p'] = self.leg_param['yH']
        self.leg_param['zH_p'] = self.leg_param['zH']
        self.leg_param['yH'] = yH
        self.leg_param['zH'] = zH  

        return np.array([self.leg_param['yH'],self.leg_param['zH']])

    def compute_theta_1(self):
        # This function solves the theta1 angle for the current 
        # configuration of servo q1,q2. 
        # INPUT: none
        # OUTPUT: none

        # Unload necessary equation parameters
        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        rp = self.leg_param['rp']
        lr0 = self.leg_param['lr0']
        q1 = self.leg_param['q1']
        q2 = self.leg_param['q2']
        
        # Function for theta1 value computation
        # Comes from the cosine rule of the tendon/leg enclosed triangle
        # CHECK TO IMPROVE THIS FUNCTION
        q2 = min(self.q2_crit,q2)

        sol_theta1 = np.pi - np.arccos((l1**2/2 - (lr0 - q2*rp)**2/2 + l2**2/2)/(l1*l2))
        yB = l1*np.sin(q1) + l2*np.sin(sol_theta1 + q1)
        zB = l1*np.sin(q1) + l2*np.sin(sol_theta1 + q1)

        # Re-assignment of values
        self.leg_param['theta1_p'] = self.leg_param['theta1']
        self.leg_param['theta1'] = sol_theta1
        self.leg_param['yB_p'] = self.leg_param['yB']
        self.leg_param['zB_p'] = self.leg_param['zB']
        self.leg_param['yB'] = yB
        self.leg_param['zB'] = zB
    
    # ******************* INVERSE KINEMATIC ***********************
    def inverse_kinematics(self, vel_B, time_step = 0.02):
        # This methods run through the inverse kinematic chain
        # INPUT: 
        #       vel_B: (2,) numpy vector|| velocity from motion planner 
        #       time_step: int          || integration timestep
        # OUTPUT: 
        #       q_val: (2,) numpy vector|| new servo values

        # Loading of relevant values
        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        l3 = self.leg_param['l3']
        l4 = self.leg_param['l4']
        lr0 = self.leg_param['lr0']
        rp = self.leg_param['rp']
        theta3 = self.leg_param['theta3']
        q1 = self.leg_param['q1']
        q2 = self.leg_param['q2']

        yH_dot = vel_B[0]
        zH_dot = vel_B[1]
        
        # Symbolic solutions pulled from Matlab. 
        # Provide direct solution for better computation of Heun and RK4 integration
        sol_q1_dot = lambda x,y: (zH_dot*l4*np.cos(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) + yH_dot*l4*np.sin(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) + zH_dot*l3*np.cos(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) - yH_dot*l3*np.sin(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))))/(l1*(l3*np.cos(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.sin(x) - l3*np.sin(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.cos(x) + l4*np.cos(x)*np.sin(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) + l4*np.cos(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.sin(x)))
        sol_q2_dot = lambda x,y: -(l2*(zH_dot*l4*np.cos(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) - zH_dot*l1*np.cos(x) + yH_dot*l1*np.sin(x) + yH_dot*l4*np.sin(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) + zH_dot*l3*np.cos(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))) - yH_dot*l3*np.sin(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))))/((l4*lr0*rp*np.cos(x)*np.sin(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) + (l4*lr0*rp*np.cos(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.sin(x))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) - (l4*y*rp**2*np.cos(x)*np.sin(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2))))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) - (l4*y*rp**2*np.cos(theta3 - x + np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.sin(x))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) + (l3*lr0*rp*np.cos(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.sin(x))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) - (l3*lr0*rp*np.sin(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.cos(x))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) - (l3*y*rp**2*np.cos(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.sin(x))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)) + (l3*y*rp**2*np.sin(x - np.arccos((l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)/(l1*l2)))*np.cos(x))/np.sqrt(1 - (l1**2/2 - (lr0 - y*rp)**2/2 + l2**2/2)**2/(l1**2*l2**2)))

        # Heun Integration
        # *************************************************
        # OPTION: implement a maximum velocity control here
        q2 = min(self.q2_crit,q2)
        K1_q1 = sol_q1_dot(q1,q2)
        K1_q2 = sol_q2_dot(q1,q2)
        
        K2_forward = min(self.q2_crit,q2+time_step*K1_q2)
        K2_q1 = sol_q1_dot(q1+time_step*K1_q1,K2_forward)
        K2_q2 = sol_q2_dot(q1+time_step*K1_q1,K2_forward)

        sol_q1 = q1 + time_step/2*(K1_q1 + K2_q1)
        sol_q2 = min(self.q2_crit,q2 + time_step/2*(K1_q2 + K2_q2))

        # Reassign previous timestep values
        self.leg_param['yH_dot_p'] = self.leg_param['yH_dot']
        self.leg_param['zH_dot_p'] = self.leg_param['zH_dot']
        # Assign new values
        self.leg_param['yH_dot'] = vel_B[0]
        self.leg_param['zH_dot'] = vel_B[1]

        # Overwrite previous values
        self.leg_param['q1_p'] = q1
        self.leg_param['q2_p'] = q2
        self.leg_param['q1'] = sol_q1
        self.leg_param['q2'] = sol_q2

        return np.array([self.leg_param['q1'], self.leg_param['q2']])