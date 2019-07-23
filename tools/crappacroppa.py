

''' Failed attempt at using PIL :()'''

# from PIL import Image
#     def crop(image_path, coords, saved_location:
#         image_obj = Image.open("Path of the image to be cropped")
#             cropped_image = image_obj.crop(coords)
#             cropped_image.save(saved_location)
#             cropped_image.show()


# if __name__ == '__main__':
#     image = "image.jpg"
#     crop(image, (100, 210, 710,380 ), 'cropped.jpg')

import numpy as np
import cv2
import glob
import os
import argparse

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='The directory containing images to be cropped')
    parser.add_argument('--rep', type=str, help='replace the original image with cropped (y or n or new)',default = 'n')
    print('pass')
    return parser.parse_args()


def crop(image_path):
	img = cv2.imread(image_path)
	h=len(img)
	w=len(img[0])
	cropped_image = img[:,int(w/2):w]
	filename, file_extension = os.path.splitext(image_path)
	new_image_path =filename+'_cropped'+file_extension
	cv2.imwrite(new_image_path,cropped_image)
	print('image cropped and saved at: '+new_image_path)
	return

def crop_replace(image_path):
	img = cv2.imread(image_path)
	h=len(img)
	w=len(img[0])
	cropped_image = img[:,int(w/2):w]
	cv2.imwrite(image_path,cropped_image)
	print('image cropped and replaced at: '+image_path)
	return



# init args
args = init_args()

image_dir = args.dir
replace_flag = args.rep

image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                  glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                  glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

if replace_flag == 'y':
	for image_path in image_path_list:
		crop_replace(image_path)
	print('\nAll images have been cropped and replaced\n')

else:
	for image_path in image_path_list:
		crop(image_path)
	print('\nAll images have been cropped and saved under new name\n')
