import cv2
import numpy as np
from extra_functions import get_panaroma_image, resize_images
import os
import argparse
from termcolor import colored


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--set_num', help='Set number for the source images', type=int, default=1)
parser.add_argument('-r', '--image_scale', help='Scale down the image size', type=int, default=1)
args = parser.parse_args()



set_num = args.set_num

base_dir = './sample_images/set' + str(set_num) + '/'
images_list = os.listdir(base_dir)



resize_images(base_dir, scale=args.image_scale)


if len(images_list) < 2:
	print (colored('##################################', 'cyan'))
	print (colored('##################################', 'cyan'))
	print(colored('minimum two images are required', 'red'))
	print (colored('##################################', 'cyan'))
	print (colored('##################################', 'cyan'))
	exit()



i = 1
img1 = cv2.imread(base_dir + str(i) + '.jpg')
img2 = cv2.imread(base_dir + str(i+1) + '.jpg')
_, _, final_img, crop_ROI = get_panaroma_image(img1, img2, None)



for i in range(3, 1+len(images_list)):

	img2 = cv2.imread(base_dir + str(i) + '.jpg')
	_, _, final_img, crop_ROI = get_panaroma_image(final_img, img2, None)




cv2.imshow('final_img', final_img)
cv2.imwrite(base_dir + 'final_img.jpg', final_img)
cv2.waitKey(0)
