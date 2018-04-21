# Helper functions for training DQNs
import numpy as np
import tensorflow as tf

# Helper function to resize our game frames
def processState(states, img_sz):
    return np.reshape(states, [img_sz**2 * 3])

# These functions allow us to update the parameters of our target network with
# those of the primary network. `tau` determines how much to update the target
# network's parameters by with the primary network's. If `tau = 1`, then this
# will set `targetDQN`'s parameters equal to `mainDQN`'s parameters, while
# `tau = 0` will keep `targetDQN`'s parameters constantly the same (no updates).
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    # TensorFlow graph has both networks initialized, so `tfVars[0:total_vars//2]`
    # should get the `mainDQN`'s parameters, while `tfVars[total_vars//2:]` should
    # get the `targetDQN`'s parameters
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(
            tfVars[idx+total_vars//2].assign(
                (tau*var.value()) + ((1-tau)*tfVars[idx+total_vars//2].value())
            )
        )
    return op_holder

# When updating the target DQN (that we'll use to learn the Q-values of
# actions), we simply do a weighted average of the target DQN's parameters
# with those of the main DQN's parameters.
def updateTarget(op_holder, sess, log = False):
    for op in op_holder:
        sess.run(op)
    if log:
        main = tf.trainable_variables()[0].eval(session = sess)
        target = tf.trainable_variables()[total_vars//2].eval(session = sess)
        if main.all() == target.all():
            print("Target set success")
        else:
            print("Target set failed")