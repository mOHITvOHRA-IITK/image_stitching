import cv2
import numpy as np
import os




def get_panaroma_image(base_img, target_img):


	h1,w1 = base_img.shape[0:2]
	h2,w2 = base_img.shape[0:2]

	h = h1
	if h2 > h1:
		h = h2

	w = w1
	if w2 > w1:
		w = w2


	img1 = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)


	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)


	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)


	good = []
	for m in matches:
		if m[0].distance < 0.75*m[1].distance:
			good.append(m)
			
	matches = np.asarray(good)


	query_pixels = []
	train_pixels = []

	target_pixels = []

	for m in matches:
		
		local_query_index = m[0].queryIdx
		local_train_index = m[0].trainIdx

		query_pixels.append( [ (np.float32(kp1[local_query_index].pt[0]), np.float32(kp1[local_query_index].pt[1])) ] )
		train_pixels.append( [ (np.float32(kp2[local_train_index].pt[0]), np.float32(kp2[local_train_index].pt[1])) ] )

		x = 3*w/4 + 0.5*kp1[local_query_index].pt[0] + 0.5*kp2[local_train_index].pt[0]
		y = 3*h/4 + 0.5*kp1[local_query_index].pt[1] + 0.5*kp2[local_train_index].pt[1]

		target_pixels.append( [(x, y)] )


	H1, _ = cv2.findHomography(np.array(query_pixels), np.array(target_pixels), cv2.RANSAC, 10.0)
	warpped_target_img = cv2.warpPerspective(target_img, H1, (3*w, 3*h))

	H2, _ = cv2.findHomography(np.array(train_pixels), np.array(target_pixels), cv2.RANSAC, 10.0)
	warpped_base_img = cv2.warpPerspective(base_img, H2, (3*w, 3*h))


	final_img = 0*warpped_target_img

	for i in range(3*h):
		for j in range(3*w):

			org_data = 0
			wrapped_data = 0

			if ( warpped_base_img[i,j,0] > 0 or warpped_base_img[i,j,1] > 0 or warpped_base_img[i,j,2] > 0):
				org_data = 1

			if ( warpped_target_img[i,j,0] > 0 or warpped_target_img[i,j,1] > 0 or warpped_target_img[i,j,2] > 0):
				wrapped_data = 1


			if (org_data and wrapped_data):
				final_img[i,j,:] = 0.5*warpped_base_img[i,j,:] + 0.5*warpped_target_img[i,j,:]

			else:
				if (org_data):
					final_img[i,j,:] = warpped_base_img[i,j,:]

				else:

					if (wrapped_data):
						final_img[i,j,:] = warpped_target_img[i,j,:]


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


	return base_img, target_img, crop_img





def resize_images(base_dir, scale):

	images_list = os.listdir(base_dir)

	for i in range(1, 1+len(images_list)):
		base_img = cv2.imread(base_dir + str(i) + '.jpg')
		h,w = base_img.shape[0:2]
		base_img = cv2.resize(base_img, (np.int(w/scale), np.int(h/scale)) )
		cv2.imwrite(base_dir + str(i) + '.jpg', base_img)



	