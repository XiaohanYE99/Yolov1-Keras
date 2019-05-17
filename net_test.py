# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:16:32 2019

@author: yexiaohan
"""

import tensorflow as tf
import keras
from keras.models import Model,load_model
from keras.layers import Reshape, Activation, Convolution2D,Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda,Dropout
from keras.layers.advanced_activations import LeakyReLU
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from net_train import get_loss,load_train_proposals
from PIL import Image
import matplotlib.pyplot as plt
S=7
B=2
C=2
classes=["tulip","pancy"]
"""
images,p=load_train_proposals('refine_list.txt',7,2)
labels=p[7]
label=np.zeros([7,7,12])
label[...,0:2]=labels[...,5:]
label[...,2]=labels[...,0]
label[...,3]=labels[...,0]
label[...,4:8]=labels[...,1:5]
label[...,8:12]=labels[...,1:5]
label=label.astype(np.float32)
"""
def build_detector(predicts,threshold=0.2,iou_threshold=0.4,max_output_size=10):
    idx1=S*S*C
    idx2=idx1+S*S*B
    width=224
    height=224
    
    class_probs=tf.reshape(predicts[0,:idx1],[S,S,C])
    confs=tf.reshape(predicts[0,idx1:idx2],[S,S,B])
    boxes=tf.reshape(predicts[0,idx2:],[S,S,B,4])
    """
    class_probs=tf.reshape(predicts[...,0:2],[S,S,C])
    confs=tf.reshape(predicts[...,2:4],[S,S,B])
    boxes=tf.reshape(predicts[...,4:12],[S,S,B,4])
    """
    x_offset = np.transpose(np.reshape(np.array([np.arange(S)] * S * B),
												[B, S, S]), [1, 2, 0])
    y_offset = np.transpose(x_offset, [1, 0, 2])
    boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(x_offset, dtype=tf.float32)) / S * width,
                      (boxes[:, :, :, 1] + tf.constant(y_offset, dtype=tf.float32)) / S * height,
                      tf.square(boxes[:, :, :, 2]) * width,
                      tf.square(boxes[:, :, :, 3]) * height], axis=3)

    scores = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)
    scores = tf.reshape(scores, [-1, C])  # [S*S*B, C]
    boxes = tf.reshape(boxes, [-1, 4])  # [S*S*B, 4]

    #只选择confidence最大的值作为box的类别、分数
    box_classes = tf.argmax(scores, axis=1) #边界框box的类别
    box_class_scores = tf.reduce_max(scores, axis=1) #边界框box的分数
    #print(tf.Session().run(boxes))

    filter_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    box_classes = tf.boolean_mask(box_classes, filter_mask)
    #NMS
    _boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
                       boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)

    nms_indices = tf.image.non_max_suppression(_boxes, scores,
                                               max_output_size, iou_threshold)
    scores = tf.Session().run(tf.gather(scores, nms_indices))
    boxes = tf.Session().run(tf.gather(boxes, nms_indices))
    box_classes = tf.Session().run(tf.gather(box_classes, nms_indices))

    return scores,boxes,box_classes

def detect_from_file(image_file,imshow=True,
                     detected_image_file="detected_image.jpg"):

		# read image
    image = cv2.imread(image_file)
    img_h, img_w, _ = image.shape
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image = (image / 255.0) * 2.0 - 1.0
    image=np.reshape(image,[1,224,224,3])
    model=load_model('./logs2/model.h5',custom_objects={'get_loss': get_loss})
    predicts=model.predict(image)
    scores, boxes, box_classes = build_detector(predicts)
    #print(scores, boxes, box_classes)
    predict_boxes = []
    for i in range(len(scores)):
    # 预测框数据为：[概率,x,y,w,h,类别置信度]
        predict_boxes.append((classes[box_classes[i]], boxes[i, 0],
								  boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
    #print(predict_boxes)
    show_results(image, predict_boxes, imshow, detected_image_file)
    
def show_results(image, results, imshow=True,
					 detected_image_file=None):

    image=np.reshape(image,[224,224,3])
    img_cp = image.copy()

    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3]/2)
        h = int(results[i][4]/2)


        cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)

				# 在边界框上显示类别、分数(类别置信度)
        cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1) # puttext函数的背景
        cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if imshow:
        cv2.imshow('YOLO_small detection', img_cp)
        cv2.waitKey()
        cv2.destroyAllWindows()
        if detected_image_file:
            img_cp=(img_cp+1.0)/2.0*255.0
            cv2.imwrite(detected_image_file, img_cp)
if __name__ == '__main__':
    detect_from_file('./image1.jpg',imshow=True,detected_image_file="detected_image1.jpg")
						 
