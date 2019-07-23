import cv2
import os
import argparse
import glob

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='The directory containing images to be cropped')
    parser.add_argument('--frame', type=float, help='The directory containing images to be cropped',default=0.5)
    print('pass')
    return parser.parse_args()

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    file_name = os.path.basename(file_path)
    if hasFrames:
        # cv2.imwrite("image"+str(count)+".jpg", image)
        cv2.imwrite(file_path+"/"+file_name+"_"+str(count)+".jpg", image)
        print('Image saved at: '+file_path+"/"+file_name+"_"+str(count)+".jpg")
    return hasFrames



# init args
args = init_args() 

vid_folder = args.dir
frameRate = args.frame
# frameRate = 0.5 #//captures image for every 0.5s

vid_path_list = glob.glob('{:s}/*.mp4'.format(vid_folder), recursive=True) + \
                glob.glob('{:s}/*.h264'.format(vid_folder), recursive=True)
# import pdb; pdb.set_trace()
for vid_path in vid_path_list:
	file_path, file_extension = os.path.splitext(vid_path)
	os.mkdir(file_path)
	print('Created folder '+file_path)

	print('Starting conversion for'+vid_path+'...\n')
	vidcap = cv2.VideoCapture(vid_path)
	sec = 0
	count=1
	success = getFrame(sec)
	while success:
	    count = count + 1
	    sec = sec + frameRate
	    sec = round(sec, 2)
	    success = getFrame(sec)