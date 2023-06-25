import cv2
import numpy as np
import math

class GetCommand(object):
	"""docstring for GetCommand"""
	def __init__(self, cur_mode):
		super(GetCommand, self).__init__()
		self.color_dist = {
			#'red': {'Lower': np.array([0, 90, 200]), 'Upper': np.array([0, 255, 255])},
			'red': {'Lower': np.array([0, 180, 50]), 'Upper': np.array([0, 255, 255])},
			'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
			#'white': {'Lower': np.array([0, 0, 128]), 'Upper': np.array([180, 30, 255])}
			'white': {'Lower': np.array([0, 0, 60]), 'Upper': np.array([180, 30, 200])}
			}

		cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
		cv2.moveWindow('camera', -1, 0)
		self.color_dict = {'lines':'red', 'robot':'white'}

		self.spine_mode = cur_mode
		self.robot_color = self.color_dict['robot']
		self.lines_color = self.color_dict['lines']
		self.min_area = 1e9
		self.noise_size = 1e1
		self.trg_vec = []
		self.cur_side = 0
		self.old_side = 0

	def cal_dis(self, p1, p2):
		return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

	def dot_product_angle(self, vec1, vec2):
		product_val = vec1[0]*vec2[0] + vec1[1]*vec2[1]
		vec1_len  = math.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1])
		vec2_len  = math.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1])
		angle = math.acos(product_val/(vec1_len*vec2_len))
		return angle

	def cross_product_side(self, vec1, vec2):
		product_val = vec1[0]*vec2[1] - vec1[1]*vec2[0]
		vec1_len  = math.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1])
		vec2_len  = math.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1])

		sin_val = product_val/(vec1_len*vec2_len)

		cur_val = math.sqrt(abs(sin_val))
		if self.spine_mode == 0:
			cur_val = math.sqrt(abs(cur_val))
		cur_val = -cur_val*np.sign(product_val)
		return cur_val
		return -sin_val
		

	def draw_arrowedLine(self, img, p1, p2, t_color):
		start_p = [int(p1[0]), int(p1[1])]
		end_p = [int(p2[0]), int(p2[1])]
		cv2.arrowedLine(img, start_p, end_p, t_color,2,0,0,0.2)

	def get_mean_point(self, p1, p2):
		px = (p1[0] + p2[0])/2
		py = (p1[1] + p2[1])/2
		return [px, py]

	def image_process(self, image, init_flag):
		frame = cv2.flip(image, 0)
		gs_frame = cv2.GaussianBlur(frame, (5, 5), 0) 
		hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
		erode_hsv = cv2.erode(hsv, None, iterations=1)

		robot_hsv = cv2.inRange(erode_hsv, self.color_dist[self.robot_color]['Lower'], self.color_dist[self.robot_color]['Upper'])
		robot_cnt = cv2.findContours(robot_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

		lines_hsv = cv2.inRange(erode_hsv, self.color_dist[self.lines_color]['Lower'], self.color_dist[self.lines_color]['Upper'])
		lines_cnt = cv2.findContours(lines_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		line_num = len(lines_cnt)

		if init_flag and len(robot_cnt):
			robot_area = max(robot_cnt, key=cv2.contourArea)
			rect = cv2.minAreaRect(robot_area)
			self.robot_point = rect[0]

			min_dis = 1e9
			line_count = 0
			for i in range(line_num):
				line_area_id = lines_cnt[i]
				line_rect = cv2.minAreaRect(line_area_id)
				line_center = line_rect[0]
				line_area = line_rect[1][0]*line_rect[1][1]

				if line_area < self.noise_size:
					continue

				cur_dis = self.cal_dis(self.robot_point, line_center)

				#line_count += 1
				#cur_box = cv2.boxPoints(line_rect)
				#cv2.drawContours(frame, [np.int0(cur_box)], -1, (255, 255, 255), 2)

				if cur_dis < min_dis:
					min_dis = cur_dis
					self.trg_vec = [line_center[0] - self.robot_point[0], line_center[1] - self.robot_point[1]]

				if line_area < self.min_area:
					self.min_area = line_area
			#print(line_count)

		elif len(robot_cnt):
			robot_area = max(robot_cnt, key=cv2.contourArea)
			rect = cv2.minAreaRect(robot_area)
			robot_center_point = rect[0]
			#robot_mov_vec = [robot_center_point[0] - self.robot_point[0], robot_center_point[1] - self.robot_point[1]]
			error_dis = self.cal_dis(robot_center_point, self.robot_point)
			box = cv2.boxPoints(rect)
			cv2.drawContours(frame, [np.int0(box)], -1, (0, 0, 0), 2)

			#self.draw_arrowedLine(frame, self.robot_point, robot_center_point, (255,0,0))

			center_dis = []
			for i in range(4):
				c_c_dis = self.cal_dis(box[i], self.robot_point)
				center_dis.append([c_c_dis, box[i]])
			center_dis.sort(key=lambda x:(x[0]))
			#print(center_dis)
			
			#start_vec_p = center_dis[0][1]
			#end_vec_p = center_dis[2][1]
			#robot_mov_vec = [end_vec_p[0] - start_vec_p[0], end_vec_p[1] - start_vec_p[1]]
			#self.draw_arrowedLine(frame, start_vec_p, end_vec_p, (255,0,0))
			tail_point = self.get_mean_point(center_dis[0][1], center_dis[1][1])
			head_point = self.get_mean_point(center_dis[2][1], center_dis[3][1])
			robot_mov_vec = [head_point[0] - tail_point[0], head_point[1] - tail_point[1]]
			
			self.draw_arrowedLine(frame, tail_point, head_point, (255,0,0))

			error_angle = self.dot_product_angle(self.trg_vec, robot_mov_vec)
			if error_angle > math.pi/2:
				cv2.imshow('camera', frame)
				cv2.waitKey(1)
				return self.cur_side

			if error_dis > 50:
				self.trg_vec = [robot_center_point[0] - self.robot_point[0], robot_center_point[1] - self.robot_point[1]]
				self.robot_point = self.get_mean_point(robot_center_point, self.robot_point)
				#new_x = (robot_center_point[0] + self.robot_point[0])/2
				#new_y = (robot_center_point[1] + self.robot_point[1])/2
				#self.robot_point = [new_x, new_y] #robot_center_point

			line_list = []
			for i in range(line_num):
				line_area_id = lines_cnt[i]
				line_rect = cv2.minAreaRect(line_area_id)
				line_area = line_rect[1][0]*line_rect[1][1]
				if line_area < self.min_area/2:
					continue
				line_center = line_rect[0]
				cur_dis = self.cal_dis(line_center, head_point)
				line_list.append([cur_dis, line_center, line_rect])

			line_list.sort(key=lambda x:(x[0]))
			#print(line_list)
			trg_point =  line_list[0][1]
			vec_p_0 = head_point#robot_center_point
			vec_p = head_point
			cur_trg_vec = [trg_point[0]-vec_p[0], trg_point[1]-vec_p[1]]

			self.cur_side = self.cross_product_side(robot_mov_vec, cur_trg_vec)
			cur_box = cv2.boxPoints(line_list[0][2])
			cv2.drawContours(frame, [np.int0(cur_box)], -1, (255, 255, 255), 2)
			
			#cal_vec_angle ---> sel

		cv2.imshow('camera', frame)
		cv2.waitKey(1)
		return self.cur_side

	def shutdown(self):
		cv2.waitKey(0)
		cv2.destroyAllWindows()

