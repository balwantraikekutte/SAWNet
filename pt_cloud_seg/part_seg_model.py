import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util
from transform_nets import input_transform_net

def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 20

  adj = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3, is_dist=True)
  point_cloud_transformed = tf.matmul(point_cloud, transform)
  
  input_image = tf.expand_dims(point_cloud_transformed, -1)
  adj = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)



  adj = tf_util.pairwise_distance(net_1)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util.conv2d(out3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
  
  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
  
  

  adj = tf_util.pairwise_distance(net_2)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  # out6 = tf_util.conv2d(out5, 64, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training, weight_decay=weight_decay,
  #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

  net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)



  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')


  one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
  one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
  out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     net_1,
                                     net_2,
                                     net_3])

  net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
  net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
  net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

  net2 = tf.reshape(net2, [batch_size, num_point, part_num])

  return net2

def get_model_my_model_1(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 30

  adj = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3, is_dist=True)
  point_cloud_transformed = tf.matmul(point_cloud, transform)

#############################################################################################
  
  input_image = tf.expand_dims(point_cloud_transformed, -1)
  adj = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
  
  net_local1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
    
  net_local1_intermediate = net_local1
    
  net_local1 = tf_util.conv2d(net_local1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='dgcnn1r', bn_decay=bn_decay, is_dist=True, activation_fn=None)
    
  net_local1 += net_local1_intermediate
  net_local1_ac = tf.nn.relu(net_local1)
  net_local1 = tf.reduce_max(net_local1_ac, axis=-2, keep_dims=True)
  
  net_global1 = tf_util.conv2d(input_image, 64, [1,3],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='conv1', bn_decay=bn_decay, is_dist=True)
    
  net_global1_intermediate = net_global1
    
  net_global1 = tf_util.conv2d(net_global1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='conv1r', bn_decay=bn_decay, is_dist=True, activation_fn=None)
    
  net_global1 += net_global1_intermediate 
  net_global1 = tf.nn.relu(net_global1)
    
  points_feat1_concat = tf.concat(axis=-1, values=[net_global1, net_local1])


##############################################################################################
  
  adj = tf_util.pairwise_distance(points_feat1_concat)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(points_feat1_concat, nn_idx=nn_idx, k=k)


  net_local2 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)
    
  net_local2 = tf_util.conv2d(net_local2, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='dgcnn2r', bn_decay=bn_decay, is_dist=True, activation_fn=None)
  
  net_local2 += net_local1_ac
  net_local2_ac = tf.nn.relu(net_local2)
  net_local2 = tf.reduce_max(net_local2_ac, axis=-2, keep_dims=True) 

  net_global2 = tf_util.conv2d(points_feat1_concat, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='conv2', bn_decay=bn_decay, is_dist=True)
    
  net_global2 = tf_util.conv2d(net_global2, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='conv2r', bn_decay=bn_decay, is_dist=True, activation_fn=None)
    
  net_global2 += net_global1
  net_global2 = tf.nn.relu(net_global2)
    
  points_feat2_concat = tf.concat(axis=-1, values=[net_global2, net_local2])
  
  
############################################################################################## 

  adj = tf_util.pairwise_distance(points_feat2_concat)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(points_feat2_concat, nn_idx=nn_idx, k=k)
  
  net_local3 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)
    
  net_local3 = tf_util.conv2d(net_local3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='dgcnn3r', bn_decay=bn_decay, is_dist=True, activation_fn=None)
    
  net_local3 += net_local2_ac
  net_local3_ac = tf.nn.relu(net_local3)
  net_local3 = tf.reduce_max(net_local3_ac, axis=-2, keep_dims=True)

  net_global3 = tf_util.conv2d(points_feat2_concat, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='conv3', bn_decay=bn_decay, is_dist=True)
    
  net_global3 = tf_util.conv2d(net_global3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='conv3r', bn_decay=bn_decay, is_dist=True, activation_fn=None)
    
  net_global3 += net_global2
  net_global3 = tf.nn.relu(net_global3)
    
  points_feat3_concat = tf.concat(axis=-1, values=[net_global3, net_local3])


##############################################################################################

  out7 = tf_util.conv2d(tf.concat([points_feat1_concat, points_feat2_concat, points_feat3_concat], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')


  one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
  one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
  out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     points_feat1_concat,
                                     points_feat2_concat,
                                     points_feat3_concat])

  net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
  net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
  net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

  net2 = tf.reshape(net2, [batch_size, num_point, part_num])

  return net2


def get_loss(seg_pred, seg):
  per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
  
  return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

