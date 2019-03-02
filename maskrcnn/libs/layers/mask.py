# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config_v1 as cfg
from libs.logs.log import LOG
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

_DEBUG = False 
def encode(gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width):
    # gt_boxes是 真实标记的边长大小,其mask长宽跟原图一样,rois是 rpn网络中经过dxdydwdh提出来的anchors，gt_masks 表示原图大小中每个像素都被标记为0或者1，也就是
    # 背景或者前景，而且一个图片中的 一个样本 就有一个gt_masks。 比如一个图形中有6个物体被标记，所以就会有6个gt_boxes和6个gt_masks
    # 注意的是, gt_mask跟gt_boxes的数量是一致的。
  """Encode masks groundtruth into learnable targets
  Sample some exmaples
  
  Params
  ------
  gt_masks: image_height x image_width {0, 1} matrix, of shape (G, imh, imw)
  # 映射在原图上面每个点的种类,G表示每个groundtruth
  gt_boxes: ground-truth boxes of shape (G, 5), each raw is [x1, y1, x2, y2, class]
  rois:     the bounding boxes of shape (N, 4),
  ## scores:   scores of shape (N, 1)
  num_classes; K
  mask_height, mask_width: height and width of output masks
  
  Returns
  -------
  # rois: boxes sampled for cropping masks, of shape (M, 4)
  labels: class-ids of shape (M, 1)
  mask_targets: learning targets of shape (M, pooled_height, pooled_width, K) in {0, 1} values
  mask_inside_weights: of shape (M, pooled_height, pooled_width, K) in {0, 1}Í indicating which mask is sampled
  """
  total_masks = rois.shape[0]
  if gt_boxes.size > 0: 
      # B x G
      overlaps = cython_bbox.bbox_overlaps(
          np.ascontiguousarray(rois[:, 0:4], dtype=np.float),
          np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
      print("gt_box")
      print(gt_boxes)
      print('rois')
      print(rois)
      print('overlaps:%d')
      print(overlaps)
      gt_assignment = overlaps.argmax(axis=1)  # shape is N (proposal anchors)
      max_overlaps = overlaps[np.arange(len(gt_assignment)), gt_assignment] #N(proposal anchors)存储的是跟最大gt的overlaps
      # note: this will assign every rois with a positive label 
      # labels = gt_boxes[gt_assignment, 4] # N
      labels = np.zeros((total_masks, ), np.float32) #labels的长度是 proposal anchors的数量, labels背景是-1 前景具体的分类ID
      labels[:] = -1

      # sample positive rois which intersection is more than 0.5
      keep_inds = np.where(max_overlaps >= cfg.FLAGS.mask_threshold)[0] # 从proposal anchors中选择合适的anchors的坐标
      num_masks = int(min(keep_inds.size, cfg.FLAGS.masks_per_image))
      if keep_inds.size > 0 and num_masks < keep_inds.size:
        keep_inds = np.random.choice(keep_inds, size=num_masks, replace=False)
        LOG('Masks: %d of %d rois are considered positive mask. Number of masks %d'\
                     %(num_masks, rois.shape[0], gt_masks.shape[0]))

      labels[keep_inds] = gt_boxes[gt_assignment[keep_inds], -1]
        
      # rois = rois[inds]
      # labels = labels[inds].astype(np.int32)
      # gt_assignment = gt_assignment[inds]

      # ignore rois with overlaps between fg_threshold and bg_threshold 
      # mask are only defined on positive rois
      ignore_inds = np.where((max_overlaps < cfg.FLAGS.fg_threshold))[0]
      labels[ignore_inds] = -1 
      # 不是前景的都给 -1
      mask_targets = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.int32)
      #total_masks属于propsoal个数，输出的mask的长度和宽度，种类数量，最后mask分支输出的内容

      mask_inside_weights = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.float32)

      rois [rois < 0] =  0 # 为了满足鲁棒性
      
      # TODO: speed bottleneck?

      print(gt_assignment)
      for i in keep_inds:  # 每个i都是符合 mask阔值的proposal anchor
        roi = rois[i, :4]
        cropped = gt_masks[gt_assignment[i], int(roi[1]):int(roi[3])+1, int(roi[0]):int(roi[2])+1]

        #取得原图中的groundtruth的masks, 抠出来的是proposal anchor的大小矩阵，但是矩阵每个像素点里面有具体的mask 0或者1
        cropped = cv2.resize(cropped, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
        # 把原图大小的mask 变成输出的大小的mask,mask_width跟mask_height都是28*28
        
        mask_targets[i, :, :, int(labels[i])] = cropped
        mask_inside_weights[i, :, :, int(labels[i])] = 1
  else:
      # there is no gt
      labels = np.zeros((total_masks, ), np.float32)
      labels[:] = -1
      mask_targets = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.int32)
      mask_inside_weights = np.zeros((total_masks, mask_height, mask_height, num_classes), dtype=np.float32)
  return labels, mask_targets, mask_inside_weights
# labels是从rpn中anchors跟gt_boxes的overlap中大于mask阈值的 rpn_anchors的坐标来从gt_boxes中寻找labels，gt_boxes的labels,
#  mask_targets也是 取上面大于阈值的坐标，具体是从gt_mask里面取,取得大小是相对应的原图大小。并且进行双线性插值缩放需要的大小28*28
#  mask_inside_weights [  总共rpn里面提取出来的anchors, mask高28,mask宽28,numclasses ]


def decode(mask_targets, rois, classes, ih, iw):
  """Decode outputs into final masks
  Params
  ------
  mask_targets: of shape (N, h, w, K)
  rois: of shape (N, 4) [x1, y1, x2, y2]
  classes: of shape (N, 1) the class-id of each roi
  height: image height
  width:  image width
  classes 就是每个 rois对应的物体ID
  Returns
  ------
  M: a painted image with all masks, of shape (height, width), in [0, K]
  """
  Mask = np.zeros((ih, iw), dtype=np.float32)
  assert rois.shape[0] == mask_targets.shape[0], \
    '%s rois vs %d masks' %(rois.shape[0], mask_targets.shape[0])
  num = rois.shape[0]
  rois = clip_boxes(rois, (ih, iw))
  for i in np.arange(num):
    k = classes[i]
    k = int(k)
    mask = mask_targets[i,:,:,k]

    h, w = rois[i, 3] - rois[i, 1] + 1, rois[i, 2] - rois[i, 0] + 1
    x, y = rois[i, 0], rois[i, 1]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask *= k

    # paint
    Mask[y:y+h, x:x+w] += mask
  return Mask

if __name__ == '__main__':
  import time
  import matplotlib.pyplot as plt
  t = time.time()
  np.set_printoptions(threshold=np.inf)
  for i in range(1):
    cfg.FLAGS.mask_threshold = 0.2

    W, H = 100,100

    gt_masks = np.zeros((2, H, W), dtype=np.int32)
    gt_masks[0, 20:50, 30:60] = 1
    gt_masks[1, 20:30, 10:70] = 1
    gt_boxes = np.asarray(
      [
        [30, 20,60, 50, 1],
        [10, 20,70, 30, 2]
      ])
    rois = gt_boxes[:, :4]
    print (rois)
    labels, mask_targets, mask_inside_weights = encode(gt_masks, gt_boxes, rois, 3, 10,10)

    print (rois)
    Mask = decode(mask_targets, rois, labels, H, W)
    if True:
      plt.figure(1)
      plt.imshow(Mask)
      plt.show()
      time.sleep(2)
  print(labels)
  print('average time: %f' % ((time.time() - t) / 10.0))

# 给定 gt_boxes和rois(proposal anchors) 都是对应的原图像的坐标  算出overlaps,并且通过gt_boxes的label 赋值给
# 每个proposal anchors，赋值规则是 anchors对应gt 中overlaps最大的那个。把 proposal anchor在 gt_mask上面截取得到实际的mask
# 其大小是 roi大小 然后 缩小到mask 分支 需要输出的大小。 也就是roi从 原图中截取相应的mask，其大小是rois大小，每个rois不同
# ，然后缩放到 具体的相同大小。注意这里
# 每个anchor有一维度，每个anchor中又有N个class，所以其值是0，1 代表 前景或者背景，前面N跟你分类分好了
#
# decode部分, 把得到的mask部分 缩放到 原始roi大小。然后分别 乘以 不同种类,传进来的mask值都是0，1
# 在加到 Mask上面。 Mask是原图大小。只加 roi面积的部分。所有proposal anchor都加到上面
# 注意 原始数据中gt_boxes 是  G,iw,ih   其G是 groundtruth。也就是人工标记的目标个数。 其 mask是分开存储的。