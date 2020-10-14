import cv2
import numpy as np
from extra_functions import get_panaroma_image, resize_images
import os



set_num = 2

base_dir = './sample_images/set' + str(set_num) + '/'
images_list = os.listdir(base_dir)


# resize_images(base_dir, scale=8)



if len(images_list) < 2:
	print ('minimum two images are required')
	exit()



i = 1
img1 = cv2.imread(base_dir + str(i) + '.jpg')
img2 = cv2.imread(base_dir + str(i+1) + '.jpg')
_, _, final_img = get_panaroma_image(img1, img2)

for i in range(3, 1+len(images_list)):

	img2 = cv2.imread(base_dir + str(i) + '.jpg')
	_, _, final_img = get_panaroma_image(final_img, img2)




cv2.imshow('final_img', final_img)
cv2.imwrite(base_dir + 'final_img.jpg', final_img)
cv2.waitKey(0)






