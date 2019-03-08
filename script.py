import gym
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers, regularizers
import numpy as np
import threading
from functools import reduce
import time
import os
from collections import deque
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Creates Folder to Store models variables
checkpoint_path = "./models_checkpoints"
try:

    os.mkdir(checkpoint_path)
except FileExistsError:
    pass


class StateTrasitionRecorder:

    def __init__(self):
        self.recorder_memory = deque()

    def save_state_transition(self, transition):
        """Save transition
        Inputs:
        trasition - list [s, a, r, s', done]"""

        self.recorder_memory.append(transition)

    def flush_recorder_memory(self):
        self.recorder_memory = deque()


class RolloutBuffer(StateTrasitionRecorder):

    def __init__(self, policy_net_args):
        super().__init__()
        """
        rollout_memory -- deque object, that stores other deque objects that represent episodes: 
        deque([deque_1 ([(state transition 1, state transition 2)]), deque_2([...])])
        """
        self.rollout_memory = deque()
        self.gamma = policy_net_args["gamma"]

    def save_rollout(self, episode):
        """ Saves rollout, computes Qsa and flushs the state transition memory of the current episode 

        Inputs:
        episode --  deque object which contains list of state transitions
        """

        complete_episode = self.compute_total_rewards(episode, self.gamma)
        # print(complete_episode)
        self.rollout_memory.append(complete_episode)
        self.flush_recorder_memory()

    def compute_total_rewards(self, episode_transitions, gamma):
        """ Computes Qsa discounted by gamma values for every state transition of the episode (Rollout)
        Inputs:
        episode --  deque object which contains list of state transitions deque([state transition1], [state transition2])

        Outputs:

        episode-- deque object where each state transition tuple has a new element Qsa : deque([(state transition 1, Qsa1), (state transition 1, Qsa2)])
        """

        states, actions, rewards, nex_states, dones = zip(*episode_transitions)
        Q_s_a = []

        # computationaly intensive operation O(n^2)
        for i in range(len(rewards)):
            Q_i = 0
            for j in range(i, len(rewards)):
                Q_i += rewards[j] * self.gamma ** (j - i)

            Q_s_a.append(Q_i)

        episode = deque(zip(states, actions, rewards,
                            nex_states, dones, Q_s_a))

        return(episode)

    def unroll_state_transitions(self):
        """
        unrroll state transitions for training value function estimator

        Outputs:
        states -- an array of states shape (sum of all state transitions over every rollout in memory , state_space)
        actions -- an array of actions with shape (||, action_space)
        next_states --an array of next_states shape (sum of all state transitions over every rollout in memory , state_space)
        rewards --an array of rewards shape (sum of all state transitions over every rollout in memory , 1)
        dones --  and array of dones shape (sum of all state transitions over every rollout in memory , 1)
        Q(s, a)-- and array of Q(s, a) shape (sum of all state transitions over every rollout in memory , 1)

        """

        states = ()
        actions = ()
        next_states = ()
        rewards = ()
        dones = ()
        Q_sa = ()

        for episode in self.rollout_memory:
            ep_states, ep_actions, ep_next_states, ep_rewards, ep_dones, ep_Q_s_a = zip(
                *episode)

            states += ep_states
            actions += ep_actions
            next_states += ep_next_states
            rewards += ep_rewards
            dones += ep_dones
            Q_sa += ep_Q_s_a

        states = np.asarray(states)
        actions = np.asarray(actions)
        next_states = np.asarray(next_states)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones, dtype=int)
        Q_sa = np.asarray(Q_sa).reshape(-1, 1)

        return states, actions, next_states, rewards, dones, Q_sa

    def flush_rollout_memory(self):
        """Deletes memory of rollouts : to be performed after a step of gradient descent"""

        self.rollout_memory = deque()


def build_networks(network_name, num_Hlayers, activations_Hlayers, Hlayer_sizes, n_output_units, output_layer_activation, regularization_constant, network_type, input_features,):
    """
    Creates network and performs forward propagation

    Inputs:
    network_name-- string that will be used in the tensor flow graph as the name of the ANN
    num_Hlayers -- scalar, input layer not included

    input_features -- tensor_flow placeholder: will contain the training examples to fed into the ANN
    Hlayer_sizes -- list with the size of each hidden layer
    activations_Hlayers -- list with the name of the activation function for each hidden layer
    Hlayer_sizes -- list with size of each hidden layer
    n_output_units -- scalar
    output_layer_activation -- activation object for output layer
    network_type -- string , Actor or Critic are the parameters accepted 

    Outputs:
    if network_type == "Actor"
        mu-- tensor with the averages values for the actions at a given state
        covariance -- tensor with the values for the covariance matrix
        params -- list with tf.Variable objects created in the network: weights and biases

    if network_type == "Critic"
        critic-- tensor with the expected reward for a given state
        params -- list with tf.Variable objects created in the network: weights and biases
    """

    assert(num_Hlayers == (len(activations_Hlayers)) and num_Hlayers ==
           len(Hlayer_sizes)),  "Check the number of activations and layer_sizes provided "

    with tf.variable_scope(network_type):

        network = tf.layers.Dense(Hlayer_sizes[0], activation=activations_Hlayers[0], kernel_initializer=tf.initializers.glorot_normal(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_constant), name="Layer_1")(input_features)

        for layer in range(1, num_Hlayers):

            network = tf.layers.Dense(units=Hlayer_sizes[layer], kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_constant), activation=activations_Hlayers[layer], name=(
                "Layer_" + str(layer + 1)))(network)

        if network_type == "Actor":
            mu = tf.layers.Dense(units=n_output_units, kernel_initializer=tf.initializers.glorot_normal(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_constant), activation=output_layer_activation, name="mu")(network)

            covariance = tf.layers.Dense(
                units=n_output_units, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_constant), activation=tf.nn.softplus, name="covariance")(network)
            # finds the parameters for the covariance matrix

            # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_type)

            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name + "/" + network_type)

            return mu, covariance, params

        else:

            critic = tf.layers.Dense(units=n_output_units, kernel_initializer=tf.initializers.glorot_normal(
            ), activation=output_layer_activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_constant), name="V")(network)
            # params = tf.trainable_variables(scope=network_name + "/" + network_type)
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name + "/" + network_type)

            return critic, params


class ComputationGraph:
    """ Creates the main components of the graph: Creates operation for the optimization of both actor and critic networks"""

    def __init__(self, name, policy_network_args, value_function_network_args):
        """
        Inputs : 
        name -- string with the name of the Agent the created graph will belong to  
        policy_network_args -- dictionary of parameters needed to train the Actor network
        value_function_network_args -- dictionary of parameters needed to train the Critic network 
        """
        super().__init__(policy_network_args)

        with tf.variable_scope(name):
            self.actor_optimizer = policy_network_args["optimizer"]
            self.critic_optimizer = value_function_network_args["optimizer"]

            self.st_placeholder = tf.placeholder(dtype=tf.float32, shape=[
                None, policy_network_args["state_space_size"]], name="State")

            self.rewards_placeholder = tf.placeholder(
                tf.float32, shape=[None, 1], name="rewards")
            self.actions_placeholder = tf.placeholder(
                tf.float32, shape=[None, policy_network_args["action_space_size"]], name="actions")
            self.dones_placeholder = tf.placeholder(
                tf.float32, shape=[None, 1], name="dones")

            self.Qsa_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name="Q_sa")

            self.mu, self.covariance, self.actor_params = build_networks(name, policy_network_args["num_Hlayers"], policy_network_args["activations_Hlayers"], policy_network_args[
                "Hlayer_sizes"], policy_network_args["n_output_units"], policy_network_args["output_layer_activation"], policy_network_args["regularization_constant"], "Actor", self.st_placeholder)

            self.critic, self.critic_params = build_networks(name, value_function_network_args["num_Hlayers"], value_function_network_args["activations_Hlayers"], value_function_network_args[
                "Hlayer_sizes"], value_function_network_args["n_output_units"], value_function_network_args["output_layer_activation"], value_function_network_args["regularization_constant"], "Critic", self.st_placeholder)

            with tf.variable_scope("Train_value_function_estimator"):

                self.value_function_net_cost = tf.losses.mean_squared_error(
                    self.Qsa_placeholder, self.critic) + tf.losses.get_regularization_loss(scope=name + "/" + "Critic")

                tf.summary.scalar("Critic_Cost", self.value_function_net_cost)

                # self.value_function_net_opt = tf.train.AdamOptimizer().minimize(
                #     self.value_function_net_cost, var_list=self.critic_params)

            if name == "Global_Agent":

                # get summary for Global Agent

                for variable in self.actor_params:
                    var_name = "Actor_" + variable.name.replace("kernel:0", "w").replace("bias:0", "b")
                    tf.summary.histogram(var_name, variable)

                for variable in self.critic_params:
                    var_name = "Critic_" + variable.name.replace("kernel:0", "w").replace("bias:0", "b")
                    tf.summary.histogram(var_name, variable)

            with tf.variable_scope("Train_policy_network"):

                self.advantage_funtion = tf.math.subtract(
                    self.Qsa_placeholder, self.critic)

                # Find the pdf
                self.probability_density_func = tf.distributions.Normal(
                    self.mu, self.covariance)

                self.log_prob_a = self.probability_density_func.log_prob(
                    self.actions_placeholder)

                auxiliary = tf.multiply(
                    self.log_prob_a, self.advantage_funtion)

                entropy = self.probability_density_func.entropy()

                self.auxiliary = policy_network_args["Entropy"] * \
                    entropy + auxiliary

                self.policy_net_cost = tf.reduce_sum(-self.auxiliary) + tf.losses.get_regularization_loss(scope=name + "/" + "Actor")

                # Store policy cost
                self.summary_policy_cost = tf.summary.scalar("Policy_Cost", self.policy_net_cost)

            with tf.name_scope("choose_a"):

                self.action = tf.clip_by_value(self.probability_density_func.sample(
                    1), policy_network_args["action_space_lower_bound"], policy_network_args["action_space_upper_bound"])

            with tf.name_scope("get_grad"):
                self.actor_grads = tf.gradients(self.policy_net_cost, self.actor_params)
                self.critic_grads = tf.gradients(self.value_function_net_cost, self.critic_params)

                # Save information to examine the gradients for problems like vanishing or exploding gradients

                for act_grad, critic_grad in zip(self.actor_grads, self.critic_grads):
                    var_name_actor = "Actor_" + act_grad.name.replace("Addn", "w")
                    var_name_critic = "Critic_" + critic_grad.name.replace("Addn", "w")
                    tf.summary.histogram(var_name_actor, act_grad)
                    tf.summary.histogram(var_name_critic, critic_grad)

            self.summaries = tf.summary.merge_all()


class RLAgent(ComputationGraph, RolloutBuffer):

    """Creates the backbone of every RL Agent
    An Agent in A3C will operate individually, but be a part of system of Agents all connected to a Global agent, to which all child agents will periodically syncronize with. 
    """

    def __init__(self, name,  policy_network_args, value_function_network_args, session, summary_writer, Global_Agent=None):
        """Inputs: 
           name --  string with the name of the Agent the created graph will belong to  
           session -- a tensorflow session every operation will be run in 
           summary_writer -- FileWriter object that will save summaries to then be presented in Tensorboard
           Global_Agent -- A Global_Agent object, which is very similar to a simple RL Agent but with some 
           specific methods for keep track of the improvement of the overall group of child agents

        """
        super().__init__(name, policy_network_args, value_function_network_args)

        self.current_num_epi = 0
        self.env = gym.make('LunarLanderContinuous-v2')
        self.total_number_episodes = policy_network_args["total_number_episodes"]
        self.num_episodes_before_update = policy_network_args["number_of_episodes_before_update"]
        self.Global_Agent = Global_Agent
        self.ep_rewards = []
        self.frequency_printing_statistics = policy_network_args["frequency_of_printing_statistics"]
        self.episodes_back = policy_network_args["episodes_back"]
        self.rendering_frequency = policy_network_args["frequency_of_rendering_episode"]
        self.max_steps = policy_network_args["max_steps_per_episode"]
        self.summary_writer = summary_writer
        self.name = name

        self.session = session
        if Global_Agent is not None:
            with tf.name_scope(name):

                with tf.name_scope('sync'):  # update global network and local network
                    with tf.name_scope('pull_from_global'):
                        self.pull_actor_params_op = [local_params.assign(
                            global_params) for local_params, global_params in zip(self.actor_params, Global_Agent.actor_params)]
                        self.pull_critic_params_op = [local_params.assign(
                            global_params) for local_params, global_params in zip(self.critic_params, Global_Agent.critic_params)]
                    with tf.name_scope("push_to_global"):
                        self.push_actor_params_op = self.actor_optimizer.apply_gradients(zip(self.actor_grads, self.Global_Agent.actor_params))
                        self.push_critic_params_op = self.critic_optimizer.apply_gradients(zip(self.critic_grads, Global_Agent.critic_params))

    def update_Global_Agent(self, feed_dict):
        """updates the Global Agent with the parameters find locally by running gradient descent"""
        _, _, = self.session.run([self.push_actor_params_op,
                                  self.push_critic_params_op], feed_dict)

    def save_summary(self, feed_dict):
        """Collects summary operations and writes with the FileWriter"""
        summary = self.session.run(self.Global_Agent.summaries, feed_dict)
        self.summary_writer.add_summary(summary, self.Global_Agent.current_num_epi)

    def pull_from_global(self):
        """ Updates the parameters of the local networks with the parameters coming from the Global network
            Because the Global Network suffers asychronous updates from variuous child agents we need to pull from Global to 
            provide all the childs with the information gained from other child agents as the only link between them is the Global network """
        self.session.run([self.pull_actor_params_op,
                          self.pull_critic_params_op])

    def take_action(self, state):
        """Computes the action to take at a given state s 
        Input:
        state-- an array with shape (state_space_size,)
        Returns:
        action -- an array with with schape (action_space)


        """
        state = state.reshape(-1, 8)
        # v = self.session.run([self.critic], feed_dict={self.st_placeholder: state})
        # print(f"Expected Reward {v}")
        # mu = self.session.run([self.mu], feed_dict={self.st_placeholder: state})
        # covariance = self.session.run([self.covariance], feed_dict={self.st_placeholder: state})
        # print(f"Mean {mu} \n")
        # print(f"Covariance {covariance} \n")

        action = self.session.run([self.action], feed_dict={
                                  self.st_placeholder: state})
        # print(action)

        return action[0].reshape(2,)

    def collect_rollouts(self, n_rolls, max_steps, render=False):
        """ Runs a specified number of episodes and collects the sate transition information

        Inputs:
        n_rolls -- int number of episodes to run 
        max_steps -- maximum number of decisions allowed per episode
        prevents the episode of never ending, which can happen, if the agent never crashes nor lands 
        render-- boolean: True if we want to render the episode, False otherwise
        """
        for i in range(n_rolls):
            n_steps = 0
            state = self.env.reset()
            done = False
            sum_rewards = 0
            while not done and n_steps <= max_steps:
                if render:
                    self.env.render()

                action = self.take_action(state)
                # print(action)
                next_state, reward, done, info = self.env.step(action)

                if not done and n_steps == max_steps:
                    # print("here")
                    state_feed = next_state.reshape(-1, 8)
                    reward = reward + float(self.session.run([self.critic], feed_dict={self.st_placeholder: state_feed})[0])

                self.save_state_transition(
                    [state, action, reward, next_state, done])

                # print(f"""Observation: {next_state.shape} \n
                #         Reward:  {reward} \n
                #         Done: {done}
                #         """)

                sum_rewards += reward
                state = next_state
                n_steps += 1

            if self.name == "Global_Agent":
                print(f"Episode Reward: {sum_rewards}")

            self.ep_rewards.append(sum_rewards)
            self.save_rollout(self.recorder_memory)

    def training_loop(self):
        """Runs episodes in a loop and performs steps of gradient descent after every episode"""

        while not coord.should_stop() and self.Global_Agent.current_num_epi <= self.total_number_episodes:
            self.collect_rollouts(
                self.num_episodes_before_update, self.max_steps, render=False)

            states, actions, next_states, rewards, dones, Q_sa = self.unroll_state_transitions()

            feed_dict = {self.st_placeholder: states,
                         self.actions_placeholder: actions,
                         self.Qsa_placeholder: Q_sa}

            self.update_Global_Agent(feed_dict)
            self.Global_Agent.current_num_epi += self.num_episodes_before_update

            feed_dict_global_summary = {self.Global_Agent.st_placeholder: states,
                                        self.Global_Agent.actions_placeholder: actions,
                                        self.Global_Agent.Qsa_placeholder: Q_sa}

            self.save_summary(feed_dict_global_summary)

            self.flush_rollout_memory()
            self.pull_from_global()

            if self.Global_Agent.current_num_epi % self.frequency_printing_statistics == 0:

                average_reward = self.Global_Agent.compute_average_rewards(self.episodes_back)
                print(
                    f"Global ep number {self.Global_Agent.current_num_epi}: Reward = {average_reward}")

            # if self.Global_Agent.current_num_epi % self.rendering_frequency == 0:
            #     self.Global_Agent.collect_rollouts(1, render=True)


class Global_Agent(RLAgent):
    def __init__(self, name, policy_network_args, value_function_network_args, session, summary_writer, child_agents=[]):
        super().__init__(name, policy_network_args, value_function_network_args, session, summary_writer)
        self.child_agents = child_agents
        self.num_childs = len(child_agents)

    def compute_average_rewards(self, episodes_back):
        """Computes the average reward of each child agent going n episodes back, and returnes the average of those average rewards"""
        reward = 0
        for agent in self.child_agents:
            agent_average_reward = reduce(
                lambda x, y: x + y, agent.ep_rewards[-episodes_back:]) / episodes_back
            reward += agent_average_reward

        reward /= self.num_childs

        return reward


if __name__ == "__main__":

    env = gym.make('LunarLanderContinuous-v2')
    action_space_upper_bound = env.action_space.high
    action_space_lower_bound = env.action_space.low
    subdir = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    logdir = "./summary/" + subdir
    writer = tf.summary.FileWriter(logdir)
    sess = tf.Session()

    policy_net_args = {"num_Hlayers": 2,
                       "activations_Hlayers": ["relu", "relu"],
                       "Hlayer_sizes": [100, 100],
                       "n_output_units": 2,
                       "output_layer_activation": tf.nn.tanh,
                       "state_space_size": 8,
                       "action_space_size": 2,
                       "Entropy": 0.01,
                       "action_space_upper_bound": action_space_upper_bound,
                       "action_space_lower_bound": action_space_lower_bound,
                       "optimizer": tf.train.RMSPropOptimizer(0.0001),
                       "total_number_episodes": 5000,
                       "number_of_episodes_before_update": 1,
                       "frequency_of_printing_statistics": 100,
                       "frequency_of_rendering_episode": 1000,
                       "number_child_agents": 8,
                       "episodes_back": 20,
                       "gamma": 0.99,
                       "regularization_constant": 0.01,
                       "max_steps_per_episode": 2000

                       }

    valuefunction_net_args = {"num_Hlayers": 2,
                              "activations_Hlayers": ["relu", "relu"],
                              "Hlayer_sizes": [100, 64],
                              "n_output_units": 1,
                              "output_layer_activation": "linear",
                              "state_space_size": 8,
                              "action_space_size": 2,
                              "optimizer": tf.train.RMSPropOptimizer(0.01),
                              "regularization_constant": 0.01}

    global_agent = Global_Agent("Global_Agent", policy_net_args, valuefunction_net_args, sess, writer)

    child_agents = []

    for i in range(policy_net_args["number_child_agents"]):
        i_name = f"ChildAgent_{i}"
        child_agents.append(RLAgent(i_name, policy_net_args, valuefunction_net_args, sess, writer, global_agent))

    global_agent.child_agents = child_agents
    global_agent.num_childs = len(child_agents)

    saver = tf.train.Saver()

    coord = tf.train.Coordinator()

    # Check if there is a model checkpoint, if there is load variables in the model
    if len(os.listdir(checkpoint_path)) == 0:

        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, checkpoint_path + "/variables.ckpt")

    # Creates a saver to hold trained weights

    child_agents_threads = []

    subdir = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    logdir = "./summary/" + subdir
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)

    for child_agent in child_agents:
        def job(): return child_agent.training_loop()
        t = threading.Thread(target=job)
        t.start()
        child_agents_threads.append(t)

    coord.join(child_agents_threads)
    saver.save(sess, checkpoint_path + "/variables.ckpt")


    # Test the model at the end of trainning

    for i in range(1):

        global_agent.collect_rollouts(10, 2000, render=True)
        global_agent.collect_rollouts(90, 2000)

    rewards = global_agent.ep_rewards
    average = sum(rewards)/len(rewards)
    average = [average] * 100

    fig, ax = plt.subplots()

    ax.plot(rewards, label="Episode Reward")
    ax.plot(average, label="Average")
    ax.legend(loc="best")
    plt.show()
