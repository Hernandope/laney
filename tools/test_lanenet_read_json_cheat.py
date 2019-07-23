#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

import json

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir. To replace original image, use replace param', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, use_gpu):
    """

    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('Start reading image data and pre-processing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image - VGG_MEAN
    log.info('Image is read, time consuming: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        log.info('Single image lane line prediction time consuming: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()

    return

### == convert final mask_image to binary_mask ==
def maskToBinaryConvert(mask_image):
    binary_mask = mask_image
    for x in range(0,len(mask_image)):
        for y in range(0,len(mask_image[x])):
            if np.array_equal(mask_image[x][y],[0,0,0]) == False:
                binary_mask[x][y] = [255,255,255]
    return binary_mask

### == Identify all the lanes ==
def coloredLaneIdentifierRaw(colored_lane_image):
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

### == EXTRACT DEM LANNEY GOODNESS for tuSimple test images==
def laneExtractorTuTest(colored_lane_image):
    lane_colours = coloredLaneIdentifierRaw(colored_lane_image)
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

### == EXTRACT DEM LANNEY GOODNESS for tuSimple test images according to preset h_samples (cheat)==
def laneExtractorTuTestCheat(colored_lane_image):
    lane_colours = coloredLaneIdentifierRaw(colored_lane_image)
    lanes_dict = {}
    lanes = []

    h_samples=[160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710]
    append_indicator = []
    #initialize lanes_dict and append_indicator
    for i in range(0,len(lane_colours)):
        lanes_dict[i] = []
        append_indicator.append(False)

    for x in h_samples:
        ##resets the append indicator after every row iteration
        for ind in append_indicator:
            ind = False
        for y in range(0,len(colored_lane_image[x])):
            for i in range(0,len(lane_colours)):
                recorded = lane_colours[i]
                if np.array_equal(colored_lane_image[x][y],recorded):
                    lanes_dict[i].append(y)
                    append_indicator[i] = True
        ##append -2 to lanes colour not appended with anything
        for indicator in append_indicator:
            if indicator == False:
                i = append_indicator.index(indicator)
                lanes_dict[i].append(-2)
                # if np.array_equal(colored_lane_image[x][y],[0,0,0]):
                #     lanes_dict[i].append(-2)

    for lane in lanes_dict.values():
        lanes.append(lane)

    return lanes, h_samples

def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """

    json_data_list = []

    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('Getting the image file path...')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        for epoch in range(epoch_nums):

            run_time=[]

            log.info('[Epoch:{:d}] Start image reading and preprocessing...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch

            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] Pretreatment{:d}image, total time consuming: {:.5f}s, Average time per sheet: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            run_time.append(t_cost)
            log.info('[Epoch:{:d}] Prediction{:d}image lane line, total time consuming: {:.5f}s, Average time per sheet: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('mask_image_after.jpg',mask_image)

                lanes, h_samples = laneExtractorTuTestCheat(mask_image)

                # binary_mask = maskToBinaryConvert(mask_image)
                # cv2.imwrite('bm_'+image_path_epoch[0],binary_mask)
                # binary_mask_list = binary_mask.tolist()
                # json_data = { 'file_name' : image_path_epoch[0] , 'binary_mask' : binary_mask_list}
                json_data = { 'raw_file' : image_path_epoch[0] , 'lanes' : lanes, 'h_samples' : h_samples, 'run_time' : run_time}
                json_data_list.append(json_data)
                with open('cheat_json_data_list.json','w') as f: json.dump(json_data_list,f)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir == 'replace':
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_save_path = image_path_epoch[index]
                    cv2.imwrite(image_save_path, mask_image)

                elif save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)

            log.info('[Epoch:{:d}] Execute{:d}Image lane line clustering, total time consuming: {:.5f}s, Average time per sheet: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir) and args.save_dir != 'replace':
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
