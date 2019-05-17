# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:41:53 2019

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


S=7
B=2
C=2

def load_train_proposals(datafile,S,C):
    train_list = open(datafile,'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        image = cv2.imread(tmp[0])
        W,H,_=image.shape
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        box = tmp[2].split(',')
        boxes = [int(i) for i in box]
        boxes[0]+=boxes[2]/2
        boxes[1]+=boxes[3]/2
        index = int(tmp[1])
        label=np.zeros([S,S,5+C])
        label_box=np.zeros(4)
        cell_w=W/S
        cell_h=H/S
        cell_x=int(boxes[0]/cell_w)
        cell_y=int(boxes[1]/cell_h)
        label_box[0]=(boxes[0]-cell_x*cell_w)/cell_w
        label_box[1]=(boxes[1]-cell_y*cell_h)/cell_h
        label_box[2]=np.sqrt(boxes[2]/W)
        label_box[3]=np.sqrt(boxes[3]/H)
        one_label=np.zeros(5+C)
        one_label[0]=1
        one_label[1:5]=label_box
        one_label[5+index]=1
        label[cell_x,cell_y,:]=one_label
        images.append(image)
        labels.append(label)
        #labels_size=[S,S,5+C]
    images=np.array(images)
    labels=np.array(labels)
    #return tf.constant(images),tf.constant(labels)
    return images,labels
def calc_iou(boxes1, boxes2, scope='iou'):
    with tf.variable_scope(scope):
        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
def get_loss(y_true,y_pred):
    predicts=y_pred
    labels=y_true
    #N=labels.get_shape().as_list()[0]
    idx1=S*S*C
    idx2=idx1+S*S*B
    #N=labels.shape[0]
    lamda1=5
    lamda2=0.2
    predict_classes = tf.reshape(predicts[..., :idx1],(-1,S,S,C))
    predict_scales = tf.reshape(predicts[...,idx1:idx2],(-1,S,S,B))
    predict_boxes = tf.reshape(predicts[...,idx2:],(-1,S,S,B,4))
    label_classes=labels[...,5:]
    label_scales=labels[...,0]
    label_boxes=labels[...,1:5]
    label_boxes=tf.stack([label_boxes[...,0],
                         label_boxes[...,1],
                         tf.sqrt(label_boxes[...,2]),
                         tf.sqrt(label_boxes[...,3])],axis=-1)
    boxes=tf.reshape(label_boxes,(-1,S,S,1,4))
    boxes = tf.tile(
                boxes, [1, 1, 1, B, 1])

    iou_predict_truth = calc_iou(predict_boxes,boxes)
    response = tf.reshape(
                label_scales,
                [-1,S,S,1])

    obj_1 = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
    obj_1 = tf.cast(
        (iou_predict_truth >= obj_1), tf.float32) * response

    obj_2 = tf.ones_like(
        obj_1, dtype=tf.float32) - obj_1
    """
    loss=tf.reduce_sum(tf.square(label_classes*obj_1-predict_classes*obj_1))
    #object_loss
    max_predict_scales=tf.reduce_max(predict_scales,axis=3)
    min_predict_scales=tf.reduce_min(predict_scales,axis=3)
    loss+=tf.reduce_sum(tf.square(label_scales*obj_1-max_predict_scales*obj_1))
    #noobject_loss
    loss+=lamda2*tf.reduce_sum(tf.square(label_scales*obj_1-min_predict_scales*obj_1))
    loss+=lamda2*tf.reduce_sum(tf.square(label_scales*obj_2-max_predict_scales*obj_2))
    loss+=lamda2*tf.reduce_sum(tf.square(label_scales*obj_2-min_predict_scales*obj_2))
    #boxes_loss
    max_predict_boxes=tf.reduce_max(predict_boxes,axis=3)
    loss+=lamda1*tf.reduce_sum(tf.square(label_boxes[...,0]*obj_1-max_predict_boxes[...,0]*obj_1)+
                        tf.square(label_boxes[...,1]*obj_1-max_predict_boxes[...,1]*obj_1))
    loss+=lamda1*tf.reduce_sum(tf.square(tf.sqrt(label_boxes[...,2]*obj_1)-tf.sqrt(max_predict_boxes[...,2]*obj_1))+
                        tf.square(tf.sqrt(label_boxes[...,3]*obj_1)-tf.sqrt(max_predict_boxes[...,3]*obj_1)))
    """
    #class loss
    class_delta=response*(predict_classes-label_classes)
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(class_delta),axis=[1,2,3]))
    #object loss
    object_delta=obj_1*(predict_scales-iou_predict_truth)
    loss+=tf.reduce_mean(tf.reduce_sum(tf.square(object_delta),axis=[1,2,3]))
    #noobject loss
    noobject_delta=obj_2*predict_scales
    loss+=tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta),axis=[1,2,3]))*lamda2
    #boxes loss
    obj=tf.expand_dims(obj_1,4)
    box_delta=obj*(boxes-predict_boxes)
    loss+=tf.reduce_mean(tf.reduce_sum(tf.square(box_delta),axis=[1,2,3,4]))*lamda1
    return loss  
def net():
    
    model=load_model('./logs1/model.h5')
    for i in range(6):
        model.pop()
    for layer in model.layers[:-10]:
        layer.trainable = False
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1,name='leak_1'))
    model.add(Dropout(0.5,name='drop_1'))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1,name='leak_2'))
    model.add(Dropout(0.5,name='drop_2'))
    model.add(Dense(S*S*(5*B+C)))
    return model
"""
def cal_acc(predicts,labels):
    idx1=S*S*C
    idx2=idx1+S*S*B
    label_scales=labels[...,0]
    predict_scales = tf.reshape(predicts[...,idx1:idx2],(-1,S,S,B))
    ans=label_scales*predict_scales[...,0]+label_scales*predict_scales[...,1]
    return tf.reduce_sum(ans)
"""
if __name__ == '__main__':
    images,labels=load_train_proposals('refine_list.txt',7,2)

    
    #model=net()
    model=load_model('./logs2/model.h5',custom_objects={'get_loss': get_loss})
    
    model.summary()
    
    adam = keras.optimizers.Adam(lr=1e-5)
    model.compile(loss=get_loss,optimizer=adam)

    for i in range(30):
        predicts=model.predict(images)

        model.fit(images,labels,nb_epoch=10,batch_size=16)
        
        mp = "./logs2/model.h5"
        model.save(mp)
