import math
import re
from func import *

def data_extraction(data_str):
	str_list = re.split(":|,|\[|\]",data_str)[2:-1]
	data_list = []
	dl = len(str_list)
	for i in range(dl):
		data_list.append(float(str_list[i]))
	return data_list

def get_pitch(ax, ay, az):
	#f_norm = math.sqrt(ax*ax + ay*ay + az*az)
	#pitch_val = math.atan2(-ax,f_norm)*180/math.pi

	f_norm = math.sqrt(ay*ay + az*az)
	pitch_val = math.atan2(-ax,f_norm)*180/math.pi
	return pitch_val

def get_roll(ax, ay, az):
	roll_val = math.atan2(-ay,az)*180/math.pi
	#f_norm = math.sqrt(ax*ax + az*az)
	#roll_val = math.atan2(ay,f_norm)*180/math.pi
	return roll_val

def get_yaw(ax, ay, az):
	#yaw_val = math.atan2(ax, ay)*180/math.pi
	f_norm = math.sqrt(ax*ax + az*az)
	yaw_val = math.atan2(az, f_norm)*180/math.pi
	return yaw_val



def data_str_split(data_str):
	data_str_list = data_str.split(";")
	acc = data_extraction(data_str_list[0])
	gyro = data_extraction(data_str_list[1])

	for i in range(3):
		acc[i] /= 16384.0 #unit g
	for i in range(3):
		gyro[i] /= 131.0

	ax = -acc[2]
	ay = acc[1]
	az = -acc[0]
	print("ax -- ", ax, "; ay -- ", ay, "; az -- ", az)

	pitch = get_pitch(ax, ay, az)
	roll = get_roll(ax, ay, az)
	yaw = get_yaw(ax, ay, az)

	print(" ----> ", acc)
	print([pitch, roll, yaw])
	return acc, gyro, pitch, roll, yaw