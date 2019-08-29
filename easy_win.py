import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import imutils
import itertools

model = load_model('SetNet_0827.h5')

def apply_brightness_contrast(input_img, brightness = -10, contrast = 64):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def check_rotation(temp_img):
    if (temp_img.shape[0] > temp_img.shape[1]):
        temp_img = rotate_image(temp_img,90)
        temp_img = apply_brightness_contrast(temp_img)
    return temp_img

def translate_results(results):
    colors,nums,shades,shapes = [],[],[],[]
    for result in results:
        colors.append(['blue','green','red'][np.argmax(result[0][:])])
        nums.append(['one','two','three'][np.argmax(result[1][:])])
        shades.append(['empty','partial','full'][np.argmax(result[2][:])])
        shapes.append(['diamond','oval','squiggle'][np.argmax(result[3][:])])
    this_panda = pd.DataFrame()
    this_panda['colors'] = colors
    this_panda['nums'] = nums
    this_panda['shades'] = shades
    this_panda['shapes'] = shapes
    return this_panda

def sort_contours(cnts, method="left-to-right"):
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

def box_extraction(orig_img, cropped_dir_path):
    idx = 0
    #orig_img = cv2.imread(os.path.join(img_for_box_extraction_path,file), 1)  # Read the image
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255-img_bin
    kernel_length = np.array(img).shape[1]//40
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    alpha = 0.6
    beta = 0.7
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    coords = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h) # for debugging
        if (((w/h > 1.4) and (w/h < 1.9)) or ((w/h < 0.7) and (w/h > 0.5))) and ((w>50) and (h>50) and (w<200) and (h<200)):
            coords.append([x,y,w,h])
    return coords
        
cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    returned_coords = box_extraction(frame, "clipped/")
    subframes = []
    for coord in returned_coords:
        frame = cv2.rectangle(frame,(coord[0],coord[1]),(coord[0]+coord[2],coord[1]+coord[3]),(255,0,0),3)
        new_img = frame[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        subframe = check_rotation(new_img)
        subframes.append(subframe)
    if (len(subframes) == 12):
        results = []
        for sf in subframes:
            results.append(model.predict(np.expand_dims(cv2.resize(sf,(100,64)),0)))
        results = translate_results(results)
        for i,coord in enumerate(returned_coords):
        	frame = cv2.putText(frame," ".join(results.iloc[i].values),(coord[0],coord[1]+coord[3]), 
        						cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        						(255,255,255),1,cv2.LINE_AA)

        column_permutations = list(itertools.permutations(['colors','counts','shades','shapes']))
        '''
        for permutation in column_permutations:
            if (results.duplicated(permutation[:-1]).value_counts()[1] >= 3):
                print('WIN')
                break
                '''

    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing...")
        break
 
cam.release()
cv2.destroyAllWindows()
