import cv2
import numpy as np
from extra_functions import get_panaroma_image, resize_images
import os
import argparse
from termcolor import colored


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--set_num', help='Set number for the source images', type=int, default=1)
parser.add_argument('-r', '--image_scale', help='Scale down the image size', type=int, default=1)
parser.add_argument('-w', '--save_resized_image', help='save_resized_image', type=bool, default=False)
args = parser.parse_args()



set_num = args.set_num

base_dir = './sample_images/set' + str(set_num) + '/'
images_list = os.listdir(base_dir)



resize_images_list = resize_images(base_dir, args.image_scale, args.save_resized_image)


if len(resize_images_list) < 2:
	print (colored('##################################', 'cyan'))
	print (colored('##################################', 'cyan'))
	print(colored('minimum two images are required', 'red'))
	print (colored('##################################', 'cyan'))
	print (colored('##################################', 'cyan'))
	exit()




img1 = resize_images_list[0]
img2 = resize_images_list[1]
_, _, final_img, crop_ROI = get_panaroma_image(img1, img2, None)



for i in range(2, len(resize_images_list)):

	img2 = resize_images_list[i]
	_, _, final_img, crop_ROI = get_panaroma_image(final_img, img2, None)




cv2.imshow('final_img', final_img)
cv2.imwrite(base_dir + 'final_img.jpg', final_img)
cv2.waitKey(0)
