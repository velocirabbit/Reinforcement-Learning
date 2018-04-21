'''
Code for setting up and training A3C networks.
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


'''
Creates a single instance of an Advantageous AC network
'''
class ACNetwork:
    def __init__(self, s_size, a_size, img_sz, scope, trainer):
        # Used to initialize weights for policy and value output layers
        # Returns a function handle to the initializer
        def _norm_col_initializer(std = 1.0):
            def _initializer(shape, dtype = None, partition_info = None):
                out = np.random.randn(*shape).astype(np.float32)
                out *= std / np.sqrt(np.square(out).sum(axis = 0, keepdims = True))
                return tf.constant(out)
            return _initializer
    
        with tf.variable_scope(scope):
            # Input layers
            self.inputs = tf.placeholder(shape = [None, s_size], dtype = tf.float32)
            self.image_in = tf.reshape(self.inputs, shape = [-1, img_sz, img_sz, 1])
            # Convolution layers
            self.conv1 = layers.convolution2d(
                inputs = self.image_in, num_outputs = 16, padding = 'valid',
                kernel_size = [8, 8], stride = [4, 4], activation_fn = tf.nn.elu
            )
            self.conv2 = layers.convolution2d(
                inputs = self.conv1, num_outputs = 32, padding = 'valid',
                kernel_size = [4, 4], stride = [2, 2], activation_fn = tf.nn.elu
            )
            hidden = layers.fully_connected(
                tf.contrib.layers.flatten(self.conv2), 256, activation_fn = tf.nn.elu
            )
            
            # Recurrent subnet for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple = True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])  # Insert a dimension of 1 at axis 0
            step_size = tf.shape(self.image_in)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state = state_in,
                sequence_length = step_size, time_major = False
            )
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            
            # Output layers for policy and value estimations
            self.policy = layers.fully_connected(
                rnn_out, a_size, activation_fn = tf.nn.softmax,
                weights_initializer = _norm_col_initializer(1.0),
                biases_initializer = None
            )
            self.value = layers.fully_connected(
                rnn_out, 1, activation_fn = None,
                weights_initializer = _norm_col_initializer(1.0),
                biases_initializer = None
            )
