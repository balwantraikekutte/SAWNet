import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net


def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl

def get_model_my_model(point_cloud, is_training, bn_decay=None):
    
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20
    
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)
    
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    
    # Conv 1
    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
    
    net_local1 = tf_util.conv2d(edge_feature, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn1', bn_decay=bn_decay)
    net_local1 = tf.reduce_max(net_local1, axis=-2, keep_dims=True)
    
    net_local1_intermediate = net_local1
    
    net_local1 = tf_util.conv2d(net_local1, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn1', bn_decay=bn_decay, activation_fn=None)
    net_local1 = tf.reduce_max(net_local1, axis=-2, keep_dims=True)
    
    net_local1 += net_local1_intermediate
    net_local1 = tf.nn.relu(net_local1)
    
    #net1 = net_local1
    
    net_local_vector1 = tf_util.max_pool2d(net_local1, [num_point,1],
                         padding='VALID', scope='maxpool1')
    
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net_global1 = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    
    net_global_vector1 = tf_util.max_pool2d(net_global1, [num_point,1],
                           padding='VALID', scope='maxpool1')
    
    points_feat1_concat = tf.concat(axis=-1, values=[net_global_vector1, net_local_vector1])
    points_feat1_concat = tf.reduce_max(points_feat1_concat, axis=-2, keep_dims=True)
    
    # Conv 2
    adj_matrix = tf_util.pairwise_distance(points_feat1_concat)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(points_feat1_concat, nn_idx=nn_idx, k=k)
    
    net_local2 = tf_util.conv2d(edge_feature, 64, [1,1], padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training, scope='dgcnn2', bn_decay=bn_decay)
    net_local2 = tf.reduce_max(net_local2, axis=-2, keep_dims=True)
    #net2 = net_local2
    
    net_local_vector2 = tf_util.max_pool2d(net_local2, [num_point,1],
                         padding='VALID', scope='maxpool2')
    

    net_global2 = tf_util.conv2d(points_feat1_concat, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    
    net_global_vector2 = tf_util.max_pool2d(net_global2, [num_point,1],
                           padding='VALID', scope='maxpool2')
    
    points_feat2_concat = tf.concat(axis=-1, values=[net_global_vector2, net_local_vector2])
    
    # Conv 3
    adj_matrix = tf_util.pairwise_distance(points_feat2_concat)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(points_feat2_concat, nn_idx=nn_idx, k=k)
    
    net_local3 = tf_util.conv2d(edge_feature, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn3', bn_decay=bn_decay)
    net_local3 = tf.reduce_max(net_local3, axis=-2, keep_dims=True)
    #net3 = net_local3
    
    net_local_vector3 = tf_util.max_pool2d(net_local3, [num_point,1],
                         padding='VALID', scope='maxpool3')
    

    net_global3 = tf_util.conv2d(points_feat2_concat, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    
    net_global_vector3 = tf_util.max_pool2d(net_global3, [num_point,1],
                           padding='VALID', scope='maxpool3')
    
    points_feat3_concat = tf.concat(axis=-1, values=[net_global_vector3, net_local_vector3])
    
    # Conv 4
    adj_matrix = tf_util.pairwise_distance(points_feat3_concat)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(points_feat3_concat, nn_idx=nn_idx, k=k)
    
    net_local4 = tf_util.conv2d(edge_feature, 128, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn4', bn_decay=bn_decay)
    net_local4 = tf.reduce_max(net_local4, axis=-2, keep_dims=True)
    #net4 = net_local4
    
    net_local_vector4 = tf_util.max_pool2d(net_local4, [num_point,1],
                         padding='VALID', scope='maxpool4')
    

    net_global4 = tf_util.conv2d(points_feat3_concat, 128, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    
    net_global_vector4 = tf_util.max_pool2d(net_global4, [num_point,1],
                           padding='VALID', scope='maxpool4')
    
    points_feat4_concat = tf.concat(axis=-1, values=[net_global_vector4, net_local_vector4])
    
    # Conv 5
    net_concat = tf_util.conv2d(tf.concat([points_feat1_concat, points_feat2_concat, points_feat3_concat, points_feat4_concat], axis=-1), 1024, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='conv5', bn_decay=bn_decay)
    
    # Symmetry Aggregation
    net_agg = tf_util.max_pool2d(net_concat, [num_point,1],
                         padding='VALID', scope='maxpool_agg')
    
    net = tf.reshape(net_agg, [batch_size, -1])
    #net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
    #                              scope='fc1', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp1')
    #net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
    #                              scope='fc2', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    
    return net, end_points


def get_model_my_model_1(point_cloud, is_training, bn_decay=None):
    
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20
    
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)
    
    point_cloud_transformed = tf.matmul(point_cloud, transform)
#############################    
    # Conv 1
    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
    
    net_local1 = tf_util.conv2d(edge_feature, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn1', bn_decay=bn_decay)
    
    net_local1_intermediate = net_local1
    
    net_local1 = tf_util.conv2d(net_local1, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn1r', bn_decay=bn_decay, activation_fn=None)
    
    net_local1 += net_local1_intermediate
    net_local1_ac = tf.nn.relu(net_local1)
    net_local1 = tf.reduce_max(net_local1_ac, axis=-2, keep_dims=True)
    #net1 = net_local1
    
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net_global1 = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    
    net_global1_intermediate = net_global1
    
    net_global1 = tf_util.conv2d(net_global1, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='conv1r', bn_decay=bn_decay, activation_fn=None)
    
    net_global1 += net_global1_intermediate 
    net_global1 = tf.nn.relu(net_global1)
    
    points_feat1_concat = tf.concat(axis=-1, values=[net_global1, net_local1])
##########################   
    # Conv 2
    adj_matrix = tf_util.pairwise_distance(points_feat1_concat)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(points_feat1_concat, nn_idx=nn_idx, k=k)
    
    net_local2 = tf_util.conv2d(edge_feature, 64, [1,1], padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training, scope='dgcnn2', bn_decay=bn_decay)
    
    net_local2 = tf_util.conv2d(net_local2, 64, [1,1], padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training, scope='dgcnn2r', bn_decay=bn_decay, activation_fn=None)
    net_local2 += net_local1_ac
    net_local2_ac = tf.nn.relu(net_local2)
    net_local2 = tf.reduce_max(net_local2_ac, axis=-2, keep_dims=True)
    #net2 = net_local2   

    net_global2 = tf_util.conv2d(points_feat1_concat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    
    net_global2 = tf_util.conv2d(net_global2, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2r', bn_decay=bn_decay, activation_fn=None)
    
    net_global2 += net_global1
    net_global2 = tf.nn.relu(net_global2)
    
    points_feat2_concat = tf.concat(axis=-1, values=[net_global2, net_local2])
###########################   
    # Conv 3
    adj_matrix = tf_util.pairwise_distance(points_feat2_concat)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(points_feat2_concat, nn_idx=nn_idx, k=k)
    
    net_local3 = tf_util.conv2d(edge_feature, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn3', bn_decay=bn_decay)
    
    net_local3 = tf_util.conv2d(net_local3, 64, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn3r', bn_decay=bn_decay, activation_fn=None)
    
    net_local3 += net_local2_ac
    net_local3_ac = tf.nn.relu(net_local3)
    net_local3_ac_128 = tf_util.conv2d(net_local3_ac, 128, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn3r128', bn_decay=bn_decay)
    net_local3 = tf.reduce_max(net_local3_ac, axis=-2, keep_dims=True)
    #net3 = net_local3

    net_global3 = tf_util.conv2d(points_feat2_concat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    
    net_global3 = tf_util.conv2d(net_global3, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3r', bn_decay=bn_decay, activation_fn=None)
    
    net_global3 += net_global2
    net_global3 = tf.nn.relu(net_global3)
    net_global3_128 = tf_util.conv2d(net_global3, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3r128', bn_decay=bn_decay)
    
    points_feat3_concat = tf.concat(axis=-1, values=[net_global3, net_local3])
###########################    
    # Conv 4
    adj_matrix = tf_util.pairwise_distance(points_feat3_concat)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(points_feat3_concat, nn_idx=nn_idx, k=k)
    
    net_local4 = tf_util.conv2d(edge_feature, 128, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn4', bn_decay=bn_decay)
    
    net_local4 = tf_util.conv2d(net_local4, 128, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn4r', bn_decay=bn_decay, activation_fn=None)
    net_local4 += net_local3_ac_128
    net_local4_ac = tf.nn.relu(net_local4)
    net_local4_ac_1024 = tf_util.conv2d(net_local4_ac, 1024, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='dgcnn3r1024', bn_decay=bn_decay)
    net_local4 = tf.reduce_max(net_local4_ac, axis=-2, keep_dims=True)
    #net4 = net_local4    

    net_global4 = tf_util.conv2d(points_feat3_concat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    
    net_global4 = tf_util.conv2d(net_global4, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4r', bn_decay=bn_decay, activation_fn=None)
    
    net_global4 += net_global3_128
    net_global4 = tf.nn.relu(net_global4)
    net_global4_1024 = tf_util.conv2d(net_global4, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3r1024', bn_decay=bn_decay)

    points_feat4_concat = tf.concat(axis=-1, values=[net_global4, net_local4])
#############################   
    # Conv 5
    net_concat = tf_util.conv2d(tf.concat([points_feat1_concat, points_feat2_concat, points_feat3_concat, points_feat4_concat], axis=-1), 1024, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='conv5', bn_decay=bn_decay)
    
    net_concat = tf_util.conv2d(net_concat, 1024, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=is_training,
                     scope='conv5r', bn_decay=bn_decay, activation_fn=None)
    
    net_concat =  net_concat + net_local4_ac_1024 + net_global4_1024
    net_concat = tf.nn.relu(net_concat)
    
    # Symmetry Aggregation
    net_agg = tf_util.max_pool2d(net_concat, [num_point,1],
                         padding='VALID', scope='maxpool_agg')
    
    net = tf.reshape(net_agg, [batch_size, -1])
    #net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
     #                             scope='fc1', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
     #                     scope='dp1')
    #net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
     #                             scope='fc2', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
     #                     scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    
    return net, end_points

def get_model(point_cloud, is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  adj_matrix = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

  point_cloud_transformed = tf.matmul(point_cloud, transform)
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net
 
  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
  
  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg', bn_decay=bn_decay)

  net = tf_util.conv2d(net, 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg2', bn_decay=bn_decay)

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg3', bn_decay=bn_decay)
 
  net = tf.reduce_max(net, axis=1, keep_dims=True) 

  # MLP on global point cloud vector
  net = tf.reshape(net, [batch_size, -1]) 
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

  return net, end_points


def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print(res1.shape)
      print(res1)

      print(res2.shape)
      print(res2)












