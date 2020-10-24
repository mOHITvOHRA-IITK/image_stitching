import cv2
import numpy as np
import os
from os import path



def get_panaroma_image(base_img, target_img, crop_ROI):


	h1,w1 = base_img.shape[0:2]
	h2,w2 = target_img.shape[0:2]

	h = h1
	if h2 > h1:
		h = h2

	w = w1
	if w2 > w1:
		w = w2


	img1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)


	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,crop_ROI)
	kp2, des2 = sift.detectAndCompute(img2,None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	good = []
	for m in matches:
		if m[0].distance < 0.75*m[1].distance:
			good.append(m)
			
	matches = np.asarray(good)


	# print ('kp1.shape', np.asarray(kp1).shape, 'kp2.shape', np.asarray(kp2).shape, 'matches.shape', matches.shape)
	


	query_pixels = []
	train_pixels = []

	target_pixels = []

	for m in matches:
		
		local_train_index = m[0].queryIdx
		local_query_index = m[0].trainIdx

		
		train_pixels.append( [ (np.float32(kp1[local_train_index].pt[0]), np.float32(kp1[local_train_index].pt[1])) ] )
		query_pixels.append( [ (np.float32(kp2[local_query_index].pt[0]), np.float32(kp2[local_query_index].pt[1])) ] )

		x = 3*w/4 + 0.5*kp1[local_train_index].pt[0] + 0.5*kp2[local_query_index].pt[0]
		y = 3*h/4 + 0.5*kp1[local_train_index].pt[1] + 0.5*kp2[local_query_index].pt[1]

		target_pixels.append( [(x, y)] )


	H1, _ = cv2.findHomography(np.array(query_pixels), np.array(target_pixels), cv2.RANSAC, 10.0)
	warpped_target_img = cv2.warpPerspective(target_img, H1, (3*w, 3*h))

	H2, _ = cv2.findHomography(np.array(train_pixels), np.array(target_pixels), cv2.RANSAC, 10.0)
	warpped_base_img = cv2.warpPerspective(base_img, H2, (3*w, 3*h))


	final_img = 0*warpped_target_img
	ROI_img = np.zeros(shape = final_img.shape[0:2], dtype=np.uint8)

	

	final_img_base = 0*ROI_img
	final_img_base[(warpped_base_img[:,:,0]/255.0 + warpped_base_img[:,:,1]/255.0 + warpped_base_img[:,:,2]/255.0) > 0] = 1


	final_img_target = 0*ROI_img
	final_img_target[(warpped_target_img[:,:,0]/255.0 + warpped_target_img[:,:,1]/255.0 + warpped_target_img[:,:,2]/255.0) > 0] = 2
	ROI_img[warpped_target_img[:,:,0] + warpped_target_img[:,:,1] + warpped_target_img[:,:,2] > 0] = 255


	final_img_combined = final_img_base + final_img_target


	final_img [final_img_combined == 1] = warpped_base_img[final_img_combined == 1]
	final_img [final_img_combined == 2] = warpped_target_img[final_img_combined == 2]
	final_img [final_img_combined == 3] = 0.5*np.array(warpped_base_img[final_img_combined == 3]) + 0.5*np.array(warpped_target_img[final_img_combined == 3])



	final_mask = np.zeros((3*h, 3*w), np.uint8)
	final_mask[(final_img[:,:,0] + final_img[:,:,1] + final_img[:,:,2]) > 0]  = 255

	_, contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	max_index = 0
	max_cntr_length = 0

	for i in range(len(contours)):
		lcl_cntr = contours[i]

		if (cv2.arcLength(lcl_cntr,True) > max_cntr_length):
			max_cntr_length = len(lcl_cntr)
			max_index = i


	x,y,w,h = cv2.boundingRect(contours[max_index])

	crop_img = final_img[y:y+h, x:x+w, :]
	crop_ROI = ROI_img[y:y+h, x:x+w]



	return base_img, target_img, crop_img, crop_ROI





def resize_images(base_dir, scale, save_image):

	images_list = os.listdir(base_dir)

	resize_images_list = []

	for i in range(1, 1+len(images_list)):

		if (path.exists(base_dir + str(i) + '.jpg')):
			base_img = cv2.imread(base_dir + str(i) + '.jpg')
			h,w = base_img.shape[0:2]
			base_img = cv2.resize(base_img, (np.int(w/scale), np.int(h/scale)) )
			if save_image:
				cv2.imwrite(base_dir + str(i) + '_resized.jpg', base_img)
			resize_images_list.append(base_img)



	return resize_images_list



	