# A3C-algorithm-implementation-for-continous-and-multidimensional-actions
Using as a test the OpenAI LunarLanderContinuous-v2 environment

This is an implementation of the [A3C](https://arxiv.org/abs/1602.01783) (Asynchronous Advantage Actor-Critic) algorithm introduced by the
deepmind team in 2016.

The implementation is in script.py file

#### Requirements
+ tensorflow ver. 1.12.0
+ python ver 3.x

##### Hyperparameters can be changed  for the Actor and critic networks by changing the two dictionaries:
 
 ```python
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

````
  
  
#### Performance 

![Performance](https://github.com/TomeASilva/A3C-algorithm-implementation-for-continous-and-multidimensional-actions/blob/master/supporting_images/Figure_1.png "Performance")

The performance is quite noisy, but still respects the conditions for it to be considered solved. The noisy solution may be due to early
stopping, but it seems that the reward structure does not translate itself into landings that would be considered safe. For example, landing with just a leg inside the landing zone seems to be as good as landing with both legs inside. Landing in very few time steps is also considered better, encouraging maneuvers performed with high acceleration. 

  
