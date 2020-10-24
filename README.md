# Image Stitching

**INTRODUCTION**
This repository has some codes for stiching multiple images to form a panoramic view of the scenes. 

<p align="center">
  <img src="/sample_images/set1/final_img.jpg" />
</p>

To use this repo, capture some images from the mobile and store the images in folder `</sample_images/set{number}>` with the image extension `.jpg` only. The sequence of images is very important and it should start with `1.jpg`, `2.jpg` and so on. The final panoramic view `final_img.jpg` will be saved in the folder `</sample_images/set{number}>`. 



**STEPS TO USE THIS REPO**
1. Create a folder `</sample_images/set3>`.
2. Capture images and save in the above folder with name should be `1.jpg`, `2.jpg` and so on.
3. In terminal type `cd /path/to/the/repository` and `python main.py -s 3 -r 1`.
4. To reduce the size of the image, set `-r` to higher values. 
For example  `python main.py -s 3 -r 2` means taking images from foder `</sample_images/set3>` and each image is resized to `(h/2, w/2)`.
5. To save the resized images in the same folder use the additional argument `-w true` with the command `python main.py`.




**TO DO**
1. Execution time is large because of multiple image wrapping operations.


