from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from time import time

detection_graph, sess = detector_utils.load_inference_graph()
score_thresh = 0.5
num_hands_detect = 20

def detect_image(input_img):
	image_np = cv2.imread(input_img)
	im_width = np.size(image_np, 1)
	im_height = np.size(image_np, 0)
	try:
		image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
	except:
		print("Error converting to RGB")
	start = time()
	boxes, scores = detector_utils.detect_objects(image_np,detection_graph,sess)
	t = time() - start
	print(t, "sec")
	detector_utils.draw_box_on_image(num_hands_detect, score_thresh,\
		scores, boxes, im_width, im_height,image_np)

	cv2.namedWindow('Single-Threaded Detection',cv2.WINDOW_NORMAL)
	cv2.imshow('Single-Threaded Detection',cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
	if cv2.waitKey() & 0xFF == ord('q'):
		cv2.destroyAllWindows()

def detect_video(input_vid,output_vid):
	cap = cv2.VideoCapture(input_vid)
	im_width, im_height = (cap.get(3), cap.get(4))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_vid,fourcc, 20.0, (int(im_width), int(im_height)))
	count = 0
	while (cap.isOpened()):
		ret,image_np = cap.read()
		if ret==True:
			count +=1
			try:
				image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
			except:
				print("Error converting to RGB")
			if (count%3==0):
				boxes, scores = detector_utils.detect_objects(image_np,detection_graph, sess)

				# draw bounding boxes on frame
				detector_utils.draw_box_on_image(num_hands_detect,score_thresh,scores, boxes, im_width, im_height,image_np)
				image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
				out.write(image_np)
				if (count%30 == 0):
					print("save "+str(count),end="\r", flush=True)
				cv2.namedWindow('Single-Threaded Detection',cv2.WINDOW_NORMAL)
				cv2.imshow('Single-Threaded Detection',image_np)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		else:
			break
	cap.release()
	out.release()
	cv2.destroyAllWindows()


detect_image('images/validation/Movie_4weds_174.jpg')
#detect_video('images/Now_You_See_Me_2.mp4','images/output.mp4')

#detect_image('egohands_data/_LABELLED_SAMPLES/CARDS_COURTYARD_B_T/frame_0113.jpg')
#detect_video('test_image/videoplayback-2.mp4','test_image/testvid-2-ssdmobilenet.avi')
