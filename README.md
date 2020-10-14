**INTRODUCTION**
This repository has some codes for stiching multiple images to form a panoramic view of the scenes. To use this repo, capture some images from the mobile and store the images in folder /sample_images/set{number} with the image extension .jpg only. The sequence of image is very important and it should start with 1.jpg, 2.jpg ans so on. The final panoramic view 'final_img.jpg' will be saved in the folder /sample_images/set{number}. 



**STEPS TO USE THIS REPO**
1. Create a folder /sample_images/set3
2. Capture images and save in the above folder with name should be 1.jpg, 2.jpg and so on.
3. In terminal type 
cd /path/to/the/repository
python main.py -s 3 -r 1


4. To reduce the size of the image set -r to higher values. 
For example  python main.py -s 3 -r 2 means taking images from foder '/sample_images/set3' and each image is resized to (h/2, w/2).





**TO DO**
1. Execution time should increase linearly with the number of the images.


