from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.visualization.summary_utils import visualize_input
import glob
from libs.datasets import coco

import libs.preprocessings.coco_v1 as coco_preprocess

def get_dataset(dataset_name, split_name, dataset_dir, 
        im_batch=1, is_training=False, file_pattern=None, reader=None):
    """"""
    if file_pattern is None:
        file_pattern = dataset_name + '_' + split_name + '*.tfrecord' 

    tfrecords = glob.glob('D:\迅雷下载\\records\\coco_train2014_00000-of-00033.tfrecord')
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = coco.read(tfrecords) # 一次只返回一张图片的信息
    # img_id是每个像素点的 类别。但是这个值有问题
    image, gt_boxes, gt_masks = coco_preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training)
    #visualize_input(gt_boxes, image, tf.expand_dims(gt_masks, axis=3))

    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id
    # num_instances表示一个图像中有多少个 实例被标注
if __name__ == '__main__':
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        get_dataset(dataset_name= 'coco',
                             split_name='train2014',
                             dataset_dir='D:\迅雷下载',
                             is_training=True)
    sess=tf.Session()
    package = sess.run([image,ih,iw,gt_boxes,gt_masks,num_instances,img_id])