from replay_buffer import ReplayBuffer,PrioritizedReplayBuffer,LinearSchedule
import gym
import itertools
import numpy as np
import os
import random
import tensorflow as tf
from collections import deque, namedtuple
import wrsn2 as wr
from wrsn2 import *
import matplotlib
import plotting
from plotting import *
import sys
if "../" not in sys.path:
  sys.path.append("../")

env=wrsn(sensor_node)

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        self.X_pl = tf.placeholder(shape=[None, sensor_node*3], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        # importance weights
        self.importance_weights_ph = tf.placeholder(shape=[None],dtype=tf.float32, name="weight")

        scalex=([battle_level,queue_len,distance_level])*sensor_node
        X = tf.to_float(self.X_pl)/scalex
        batch_size = tf.shape(self.X_pl)[0]

        # Fully connected layers
        
        fc1 = tf.contrib.layers.fully_connected(X, 48)
        fc2 = tf.contrib.layers.fully_connected(fc1, 32)

        self.adv= tf.contrib.layers.fully_connected(fc2, env.action_space.n,activation_fn=None) 
        self.val= tf.contrib.layers.fully_connected(fc2, 1,activation_fn=None) 
        self.predictions = self.val+self.adv- tf.reduce_mean(self.adv,axis=1,keep_dims=True)
        #self.predictions = tf.contrib.layers.fully_connected(fc2, env.action_space.n,activation_fn=None)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.td_error=self.y_pl-self.action_predictions
        # def huber_loss(x, delta=1.0):
        #     return tf.where(tf.abs(x) < delta,tf.square(x) * 0.5,delta * (tf.abs(x) - 0.5 * delta))        
        # huber_loss
        self.losses = tf.where(tf.abs(self.td_error) < 1.0,tf.square(self.td_error) * 0.5,1.0 * (tf.abs(self.td_error) - 0.5),name="losses")
        # self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses*self.importance_weights_ph)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00010, 0.95, 0.01)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, sensornode*3]

        Returns:
          Tensor of shape [batch_size, env.action_space.n] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y, weights):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, sensornode*3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a, self.importance_weights_ph: weights }
        summaries, global_step, _, loss, td_error = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss, self.td_error],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss ,td_error

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_buffer_size=500000,
                    replay_buffer_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    prioritized_replay_alpha=0.6,
                    prioritized_replay_beta0=0.4,
                    prioritized_replay_beta_iters=500000,
                    prioritized_replay_eps=1e-6):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing 
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the 
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    

    # The replay buffer
    
    replay_buffer = PrioritizedReplayBuffer(replay_buffer_size,alpha=prioritized_replay_alpha)    
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,initial_p=prioritized_replay_beta0,final_p=1.0)
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_transbag=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    # monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        copy_model_parameters(sess, q_estimator, target_estimator)
        print("\nCopied model parameters to target network.")
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        env.action_space.n)

    # Populate the replay buffer with initial experience
    print("Populating replay buffer...")
    state = env.reset_test()
    for i in range(replay_buffer_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        
        if i%1000==0:
            print("\r{} in {} ".format(i,replay_buffer_init_size),end="")
            sys.stdout.flush()
            state = env.reset_test()
        else:
            state = next_state

    # Record videos
    # Use the gym env Monitor wrapper
    # env = Monitor(env,
    #               directory=monitor_path,
    #               resume=True,
    #               video_callable=lambda count: count % record_video_every ==0)

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset_test()
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, data_overflow = env.step(action)

            # Save transition to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            stats.episode_transbag[i_episode]+=data_overflow
            # Sample a minibatch from the replay buffer
            experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(total_t))
            (states_batch, action_batch, reward_batch, next_states_batch, done_batch, weights_batch, batch_idxes) = experience

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            loss,td_error = q_estimator.update(sess, states_batch, action_batch, targets_batch, weights_batch)

            new_priorities = np.abs(td_error) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

            # if done:
            #     break
            if t>=1000:
                break
            state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        episode_summary.value.add(simple_value=stats.episode_transbag[i_episode],node_name="episode_transbag", tag="episode_transbag")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1],
            episode_transbag=stats.episode_transbag[:i_episode+1])

    #env.monitor.close()
    q_estimator.summary_writer.add_graph(sess.graph)
    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/wrsn2")

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    experiment_dir=experiment_dir,
                                    num_episodes=850,
                                    replay_buffer_size=3000000,
                                    replay_buffer_init_size=1500000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=800000,
                                    discount_factor=0.99,
                                    batch_size=32,
                                    prioritized_replay_alpha=0.6,
                                    prioritized_replay_beta0=0.4,
                                    prioritized_replay_beta_iters=800000,
                                    prioritized_replay_eps=1e-6):

        print("\nEpisode Reward: {},Episode Transbag:{}".format(stats.episode_rewards[-1],stats.episode_transbag[-1]))
    plotting.plot_episode_stats(stats)