# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inception_utils
import time
from datetime import datetime
import math
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception model from http://arxiv.org/abs/1512.00567.

  Constructs an Inception v3 network from inputs to the given final endpoint.
  This method can construct the network up to the final inception block
  Mixed_7c.

  Note that the names of the layers in the paper do not correspond to the names
  of the endpoints registered by this function although they build the same
  network.

  Here is a mapping from the old_names to the new names:
  Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):
      # 299 x 299 x 3
      end_point = 'Conv2d_1a_3x3'
      net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 149 x 149 x 32
      end_point = 'Conv2d_2a_3x3'
      net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 147 x 147 x 32
      end_point = 'Conv2d_2b_3x3'
      net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 147 x 147 x 64
      end_point = 'MaxPool_3a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 73 x 73 x 64
      end_point = 'Conv2d_3b_1x1'
      net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 73 x 73 x 80.
      end_point = 'Conv2d_4a_3x3'
      net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 71 x 71 x 192.
      end_point = 'MaxPool_5a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 35 x 35 x 192.

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # mixed: 35 x 35 x 256.
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_1: 35 x 35 x 288.
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                 scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_2: 35 x 35 x 288.
      end_point = 'Mixed_5d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_3: 17 x 17 x 768.
      end_point = 'Mixed_6a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed4: 17 x 17 x 768.
      end_point = 'Mixed_6b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_5: 17 x 17 x 768.
      end_point = 'Mixed_6c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # mixed_6: 17 x 17 x 768.
      end_point = 'Mixed_6d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_7: 17 x 17 x 768.
      end_point = 'Mixed_6e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_8: 8 x 8 x 1280.
      end_point = 'Mixed_7a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # mixed_9: 8 x 8 x 2048.
      end_point = 'Mixed_7b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_10: 8 x 8 x 2048.
      end_point = 'Mixed_7c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
  """Inception model from http://arxiv.org/abs/1512.00567.

  "Rethinking the Inception Architecture for Computer Vision"

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna.

  With the default arguments this method constructs the exact model defined in
  the paper. However, one can experiment with variations of the inception_v3
  network by changing arguments dropout_keep_prob, min_depth and
  depth_multiplier.

  The default image size used to train this network is 299x299.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v3_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      # Auxiliary Head logits
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        aux_logits = end_points['Mixed_6e']
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
              aux_logits, [5, 5], stride=3, padding='VALID',
              scope='AvgPool_1a_5x5')
          aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                   scope='Conv2d_1b_1x1')

          # Shape of feature map before the final layer.
          kernel_size = _reduced_kernel_size_for_small_input(
              aux_logits, [5, 5])
          aux_logits = slim.conv2d(
              aux_logits, depth(768), kernel_size,
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')
          if spatial_squeeze:
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

inception_v3.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.pack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out

inception_v3_arg_scope = inception_utils.inception_arg_scope

batch_size = 10
height, width = 299, 299
istart = 0
iend = 0
isize = batch_size


# 显示网络每一层结构的函数，输出名称和尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# 权重初始化函数
def weight_variable(shape):
    # 添加一些随机噪声来避免完全对称，使用截断正态分布，标准差为0.1
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)


# 偏置量初始化函数
def bias_variable(shape):
    # 为偏置量增加一个很小的正值(0.1)，避免死亡节点
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


def randommask(img, size, num, centerX, centerY, radius):
    imgHeight = img.shape[0]
    imgWidth = img.shape[0]
    mask = np.ones([imgHeight, imgWidth])
    for i in range(num):
        minX, minY, maxX, maxY = makemask(size, centerX, centerY, imgWidth, imgHeight, radius)
        for x_i in range(minX, maxX):
            for y_i in range(minY, maxY):
                mask[y_i, x_i] = 0
    return img * mask

def makemask(size, centerX, centerY, imgWidth, imgHeight ,radius):
    startX = 0
    startY = 0
    maxX = imgWidth - size
    maxY = imgHeight - size
    startX = np.random.randint(low=0, high=maxX)
    startY = np.random.randint(low=0, high=maxY)
    node_min_x = int(centerX - radius)
    node_min_y = int(centerY - radius)
    node_max_x = int(centerX + radius)
    node_max_y = int(centerY + radius)

    # 判断节点是否在mask内部
    if node_min_x > startX and node_min_y > startY and node_max_x < (startX + size) and node_max_y < (startY + size):
        # 如果在的话递归调用再随机产生一个
        return makemask(size, centerX, centerY, imgWidth, imgHeight, radius)
    else:
        # 不在的话返回mask左上角和右下角坐标
        return startX, startY, startX+size, startY+size


def randomnodemask(img, size, num, centerX, centerY, radius):
    imgHeight = img.shape[0]
    imgWidth = img.shape[0]
    mask = np.ones([imgHeight, imgWidth])
    for i in range(num):
        minX, minY, maxX, maxY = makenodemask(size, centerX, centerY, imgWidth, imgHeight, radius)
        for x_i in range(minX, maxX):
            for y_i in range(minY, maxY):
                mask[y_i, x_i] = 0
    return img * mask


def makenodemask(size, centerX, centerY, imgWidth, imgHeight ,radius):
    startX = centerX - size - radius
    startY = centerY - size - radius
    maxX = centerX + size + radius
    maxY = centerY + size + radius
    startX = np.random.randint(low=startX, high=maxX)
    startY = np.random.randint(low=startY, high=maxY)

    return startX, startY, startX+size, startY+size


def cutimg(img, centerX, centerY):
    if centerX < 106:
        startX = int(centerX) - 20
    elif centerX > 405:
        startX = int(centerX) + 20 - 299
    else:
        startX = 106

    if centerY < 106:
        startY = int(centerY) - 20
    elif centerY > 405:
        startY = int(centerY) + 20 - 299
    else:
        startY = 106

    maxX = startX + 299
    maxY = startY + 299

    # 中心点太靠上
    if centerX - startX < 20:
        startX -= 20
        maxX -= 20
    # 中心点太靠左
    if centerY - startY < 20:
        startY -= 20
        maxY -= 20
    # 中心点太靠下
    if maxX - centerX < 20:
        startX += 20
        maxX += 20
    # 中心点太靠右
    if maxY - centerY < 20:
        startY += 20
        maxY += 20
    return img[startX:maxX, startY:maxY], startX, startY


# 输入
# 先在二维空间上测试，299*299，1个通道(单色图片)
x_ = tf.placeholder(dtype=tf.float32, shape=[None, height, width])
# label: 结节的中心坐标[, 0] = x [, 1] = y
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
inputs = tf.reshape(x_, (-1, height, width, 1))

with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(inputs, depth_multiplier=1, is_training=False)

# print_activations(logits)
# 输出层
W_out = weight_variable([1000, 2])
b_out = bias_variable([2])
# 输出([None, 2])
y_conv = tf.matmul(logits, W_out) + b_out
# 损失([])
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
# 优化器
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

train_file_path = "traindata\\"
train_file_list = glob(train_file_path+"images_????_????.npy")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for runs in range(200):
    train_x = np.array(np.zeros(shape=[isize*3, 299, 299], dtype=np.float32))
    train_y = np.array(np.zeros(shape=[isize*3, 2], dtype=np.float32))

    print("Runs " + str(runs) + " start")

    for i in range(0, 100):
        istart = i*isize
        iend = istart + isize - 1
        iIndex = 0
        all_loss = 0.0
        for i_ in range(istart, iend):
            imgfilename = train_file_list[i_]
            v_centerfilename = imgfilename.replace("images", "v_center")
            imgs = np.load(imgfilename)
            v_center = np.load(v_centerfilename)

            train_x[iIndex], startX, startY = cutimg(imgs[0], v_center[0], v_center[1])
            train_x[iIndex + 1], _, _ = cutimg(imgs[0], v_center[0], v_center[1])
            train_x[iIndex + 2], _, _ = cutimg(imgs[0], v_center[0], v_center[1])

            train_y[iIndex][0] = train_y[iIndex + 1][0] = train_y[iIndex + 2][0] = v_center[0] - startX
            train_y[iIndex][1] = train_y[iIndex + 1][1] = train_y[iIndex + 2][1] = v_center[1] - startY

            train_x[iIndex] = randommask(train_x[iIndex], 32, 5, train_y[iIndex][0], train_y[iIndex][1], 20)
            train_x[iIndex] = randomnodemask(train_x[iIndex], 3, 3, train_y[iIndex][0], train_y[iIndex][1], 10)
            train_x[iIndex + 1] = randommask(train_x[iIndex + 1], 32, 3, train_y[iIndex][0], train_y[iIndex][1], 20)
            train_x[iIndex + 1] = randomnodemask(train_x[iIndex + 1], 3, 3, train_y[iIndex][0], train_y[iIndex][1], 10)
            train_x[iIndex + 2] = randommask(train_x[iIndex + 2], 32, 5, train_y[iIndex][0], train_y[iIndex][1], 20)
            train_x[iIndex + 2] = randomnodemask(train_x[iIndex + 2], 3, 3, train_y[iIndex][0], train_y[iIndex][1], 10)

            # plt.imshow(train_x[iIndex])
            # plt.plot(int(train_y[iIndex][0]), int(train_y[iIndex][1]), 'ro', color='red')
            # plt.show()

            iIndex += 1

        _, loss_ = sess.run([optimizer, loss], feed_dict={x_: train_x, y_: train_y})

        if i % 10 == 0:
            print("Step ", str(i), "Loss = ", str(loss_))


# 测试
test_imgfilename = train_file_list[1063]
test_v_centerfilename = test_imgfilename.replace("images", "v_center")
test_img = np.load(test_imgfilename)[1]
test_v_center = np.load(test_v_centerfilename)
test_x = np.array(np.zeros(shape=[1, 299, 299], dtype=np.float32))
test_y = np.array(np.zeros(shape=[1, 2], dtype=np.float32))
test_x[0] = test_img[106:405, 106:405]
test_y[0][0] = test_v_center[0] - 106
test_y[0][1] = test_v_center[1] - 106

y_e, loss_ = sess.run([y_conv, loss], feed_dict={x_: test_x, y_: test_y})

print(test_y)
print(y_e)
print(loss_)

plt.imshow(test_x[0])
plt.plot(int(test_y[0][0]), int(test_y[0][1]), 'ro', color='yellow')
plt.plot(int(y_e[0][0]), int(y_e[0][1]), 'ro', color='red')
plt.show()

plt.imshow(test_x[0])
plt.show()
