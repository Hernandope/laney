import json
import numpy as np
import cv2
import matplotlib.pyplot as plt



gt_path = 'label_data.json'
json_gt = [json.loads(line) for line in open(gt_path)]

gt = json_gt[0]

gt_lanes = gt['lanes']
y_samples = gt['h_samples']
raw_file = gt['raw_file']



###frequency count for elements in an array
unique, counts = np.unique(binary_seg_image, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(mask_image, return_counts=True)
##temporary

for x in range(0,len(binary_seg_image)):
    for y in range(0,len(binary_seg_image[x])):
        if binary_seg_image[x][y] == 1:
            binary_seg_image[x][y] = 255

with open('binary_data_list.json', 'r') as f:
    json_pr = json.load(f)

binary_mask_list = binary_mask.tolist()
data = { 'file_name' : image_path_epoch[0] , 'binary_mask' : binary_mask_list}

with open('test.json','w') as f: json.dump(data,f)

img = np.array(json_pr[0]['binary_mask'])


### == Identify all the lanes ==
def coloredLaneIdentifier(colored_lane_image):
    lane_colours = []
    for x in range(0,len(colored_lane_image)):
        for y in range(0,len(colored_lane_image[x])):
            if (len(lane_colours) == 0):
                lane_colours.append(colored_lane_image[x][y])
            else:
                check = any(np.array_equal(colored_lane_image[x][y],recorded) for recorded in lane_colours)
                if check == False:
                    lane_colours.append(colored_lane_image[x][y])
    for recorded in lane_colours:
        if np.array_equal(recorded,[0,0,0]):
            lane_colours.remove(recorded)
    return lane_colours


### == EXTRACT DEM LANNEY GOODNESS ==
def laneExtractor(colored_lane_image):
    lane_colours = coloredLaneIdentifier(colored_lane_image)
    lanes_dict = {}
    lanes = []

    h_samples = []

    #initialize lanes_dict
    for i in range(0,len(lane_colours)):
        lanes_dict[i] = []

    for x in range(0,len(colored_lane_image)):
        for y in range(0,len(colored_lane_image[x])):
            for i in range(0,len(lane_colours)):
                recorded = lane_colours[i]
                if np.array_equal(colored_lane_image[x][y],recorded):
                    lanes_dict[i].append(y)
                if x not in h_samples:
                    h_samples.append(x)

    for lane in lanes_dict.values():
        lanes.append(lane)

    return lanes, h_samples

json_data = { 'raw_file' : image_path_epoch[0] , 'lanes' : lanes, 'h_samples' : h_samples}
json_data_list.append(json_data)
with open('json_data_list.json','w') as f: json.dump(json_data_list,f)
