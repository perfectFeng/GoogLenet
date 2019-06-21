#!/usr/bin/python
#-*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import GoogleNet


event_dic = {'shot': 0,
             'corner': 1,
             'free-kick': 2,
             'yellow-card': 3,
             'foul': 4,
             'goal': 11,
             'offside': 5,
             'background': 6,
             'overhead-kick': 7,
             'solo-drive': 8,
             'penalty-kick': 9,
             'red-card': 10,
             } 
             

def readTestFile(batch_size, num_frames):
    f = open("../event_test.txt", 'r')
    lines = list(f)
    random.shuffle(lines)
    events = []
    labels = []
    for l in range(batch_size):
        frames = []
        event = lines[l].strip('\n').split(' ')
        event_type = event[0]
        start = int(event[1])
        video = event[3]
        labels.append(int(event_dic[str(event_type)]))
        cap = cv2.VideoCapture("../video/" + video)
        for n in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (GoogleNet.width, GoogleNet.height), interpolation=cv2.INTER_CUBIC)
            # n_b = np.zeros(b.shape, np.float32)
            # cv2.normalize(b, n_b, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            b = per_image_standard(b, GoogleNet.width * GoogleNet.height)
            frames.append(b)

        events.append(frames)
    f.close()
    events = np.array(events).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    return events, labels


def readFile():
    f = open("../event_train.txt", 'r')
    lines = list(f)
    random.shuffle(lines)
    f.close()

    return lines


def readTrainData(batch, lines, batch_size, num_frames):

    events = []
    labels = []

    for b in range(batch*batch_size, batch*batch_size + batch_size):

        frames = []

        event = lines[b].strip('\n').split(' ')
        event_type = event[0]
        start = int(event[1])
        end = int(event[2])
        video = event[3]
        labels.append(int(event_dic[str(event_type)]))
        cap = cv2.VideoCapture("../video/" + video)

        if (end - start) <= 32:
            first_frame = start
            skip_frame = 1
        elif (end - start) < 64:
            first_frame = random.randint(start, end-32)
            skip_frame = 1
        elif (end - start) < 128:
            first_frame = random.randint(start, end-64)
            skip_frame = 2
        else:
            first_frame = random.randint(start, end-128)
            skip_frame = 4
        for n in range(num_frames):

            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + skip_frame*n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (GoogleNet.width, GoogleNet.height), interpolation=cv2.INTER_CUBIC)
            # n_b = np.zeros(b.shape, np.float32)
            # cv2.normalize(b, n_b, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            b = per_image_standard(b, GoogleNet.width * GoogleNet.height)
            frames.append(b)

        events.append(frames)
    events = np.array(events).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    return events, labels


def per_image_standard(image, num_rgb):
    mean = np.mean(image)
    stddev = np.std(image)
    image = (image - mean)/(max(stddev, 1.0/np.sqrt(num_rgb)))
    return image
