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
    # trainer = instance of a Tensorflow optimizer
    def __init__(self, s_size, a_size, img_sz, scope, trainer, clip_norm = 40.0):
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
            
            # Only the Worker network needs ops for the loss functions and 
            # updating the gradients
            if scope != 'global':
                self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
                self.actions_OHV = tf.one_hot(self.actions, a_size, dtype = tf.float32)
                self.target_v = tf.placeholder(shape = [None], dtype = tf.float32)
                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)
                
                # Policy values for the target actions
                self.predict_policy = tf.reduce_sum(self.policy*self.actions_OHV, [1])
                
                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(
                    self.target_v - tf.reshape(self.value, [-1])
                ))
                
                self.entropy = tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.predict_policy) * self.advantages
                )
                
                self.loss = 0.5*self.value_loss + self.policy_loss - self.entropy*0.01
                
                # Get local network gradients using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, clip_norm)
                
                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

            
'''
Creates a Worker instance that contains a copy of the model network, the
environment, and logic for interacting with the environment and updating the
global network.
'''
class Worker:
    # game = instance of the environment
    # name = sometime to distinguish this Worker instance from others
    # s_size = number of states
    # a_size = number of actions
    # trainer = instance of a Tensorflow optimizer
    # model_path = 
    # global_episodes = 
    def __init__(self, game, name, s_size, a_size, buffer_size, trainer, model_path, global_episodes):
        # Copies values from one set of variables to another
        # Used to set the worker networks' weights to those of the global network's
        def update_target_graph(from_scope, to_scope):
            from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
            to_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
            # We'll return the graph operations that will do the variable assignment in a list
            op_holder = []
            for from_var, to_var in zip(from_vars, to_vars):
                op_holder.append(to_var.assign(from_var))
            return op_holder
        
        self.name = '_'.join(['worker', str(name)])
        self.number = name
        self.buffer_size = buffer_size
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter('_'.join(['train', str(self.number)]))
        
        # Create a local copy of the network and the Tensorflow ops to copy the
        # global parameters to the local network
        self.local_AC = ACNetwork(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        
        # Set up the Doom environment
        game.set_doom_scenario_path('basic.wad')
        game.set_doom_map('map01')
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = game
        
    def train(self, rollout, sess, gamma, bootstrap_value):
        # Calculates and returns discounted rewards
        def discount(x, gamma):
            return lfilter([1], [1, -gamma], x[::,-1], axis = 0)[::-1]
    
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Generate the advantage and discounted rewards using the rewards and
        # values from the rollout. The advantage function uses "Generalized
        # Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.values_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma*self.values_plus[1:] - self.values_plus[:-1]
        advantages = discount(advantages, gamma)
        
        # Update the global network using gradients from the loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_AC.target_v   : discounted_rewards,
            self.local_AC.inputs     : np.vstack(observations),
            self.local_AC.actions    : actions,
            self.local_AC.state_in[0]: self.batch_rnn_state[0],
            self.local_AC.state_in[1]: self.batch_rnn_state[1],
        }
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([
            self.local_AC.value_loss, self.local_AC.policy_loss,
            self.local_AC.entropy, self.local_AC.grad_norms,
            self.local_AC.var_norms, self.local_AC.state_out,
            self.local_AC.apply_grads
        ], feed_dict = feed_dict)
        v_l /= len(rollout)
        p_l /= len(rollout)
        e_l /= len(rollout)
        return v_l, p_l, e_l, g_n, v_n
        
    def work(self, max_episode_len, gamma, sess, coord, saver):
        # Processes an image to produce a cropped and resized image
        def process_frame(frame):
            s = frame[10:-10, 30:-30]
            s = resize(s, [84, 84])
            s = np.reshape(s, [np.prod(s.shape)]) / 255.0
            return s
    
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print('Starting worker %d' % self.number)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_steps  = 0
                d = False
                
                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while not self.env.is_episode_finished():
                    # Take an action using probabilities from policy network output
                    a_dist, v, rnn_state = sess.run(
                        [
                            self.local_AC.policy, self.local_AC.value,
                            self.local_AC.state_out
                        ],
                        feed_dict = {
                            self.local_AC.inputs     : [s],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]
                        }
                    )
                    a = np.random.choice(a_dist[0], p = a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    r = self.env.make_action(self.actions[a]) / 100.0
                    d = self.env.is_episode_finished()
                    if not d:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s, a, r, s1, d, v[0,0]])
                    episode_values.append(v[0,0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_steps += 1
                    
                    # If the episode hasn't ended but the experience replay
                    # buffer is full, then we make an update step using that
                    # experience rollout
                    if len(episode_buffer) == self.buffer_size and not d and episode_steps != max_episode_length - 1:
                        # Since we don't know what the true final return is, we
                        # bootstrap from our current value estimation
                        v1 = sess.run(self.local_AC.value, feed_dict = {
                            self.local_AC.inputs     : [s],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1],
                        })[0,0]
                        v_l, p_l, e_l, g_n, v_n = self.train(
                            episode_buffer, sess, gamma, v1
                        )
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d:
                        break
                    # Episode finished
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_steps)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)
                    
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
        
        