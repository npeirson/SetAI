import os
import cv2
import numpy as np
import pandas as pd
import argparse
import imutils

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	return (cnts, boundingBoxes)

def box_extraction(img_for_box_extraction_path, cropped_dir_path):
	idx = 0
	for file in os.listdir(img_for_box_extraction_path):
		orig_img = cv2.imread(os.path.join(img_for_box_extraction_path,file), 1)  # Read the image
		img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
		(thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
		img_bin = 255-img_bin  # Invert the image
		cv2.imwrite("Image_bin.jpg",img_bin)
		kernel_length = np.array(img).shape[1]//40
		verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
		hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
		verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
		cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
		img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
		horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
		cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
		alpha = 0.6
		beta = 0.7
		img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
		img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
		(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		cv2.imwrite("img_final_bin.jpg",img_final_bin)
		contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)
			if (((w/h > 1.4) and (w/h < 1.9)) or ((w/h < 0.7) and (w/h > 0.5))) and ((w>50) and (h>50) and (w<200) and (h<200)):
				new_img = orig_img[y:y+h, x:x+w]
				idx+=1
				cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
				'''
				cv2.imshow('image',new_img)
				k = cv2.waitKey(0)
				if k == 27:         # wait for ESC key to exit
					cv2.destroyAllWindows()
				elif k == ord('s'): # wait for 's' key to save and exit
					idx += 1
					cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
					cv2.destroyAllWindows()
				'''
			'''
			print(x,y,w,h)
			if ((w/h > 1.6) or (h/w > 1.6)) and ((w>60) or (h>60)): # and h > 80) and w > 3*h:
				idx += 1
				new_img = orig_img[y:y+h, x:x+w]
				#cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
			'''

box_extraction("images/", "clipped/")