import scipy
import scipy.io
from datetime import date, time, datetime as Date, time, datetime
from scipy import optimize
import networkx as nx
import ast
import random
import os
import re
import numpy as np
import time as Time
from operator import itemgetter
import math as m
import copy as cpy
import matplotlib.pyplot as plt
import os
import itertools
import pickle


from TabularTspEdgeSwapENV import PolytopeENV as Env
from Q_learning import Q_learning, EdgeSwap_Q_learning
from optimal_policy_extraction import edge_swap_policy_evaluation


from draw_fiber_graph import draw_graph_animation
from reward_functions import reward_cost
from helper_functions import create_tsp_polytope_graph, extract_distance_matrix, create_state_graph, create_state_edges

from create_initial_solutions import create_exact_solution




def run_q_learning_instance(seed,
                            available_actions, 
                            initial_state, 
                            distance_matrix, 
                            objective_table, 
                            reward_list,
                            episode_num,
                            max_path_length,
                            path_num,
                            node_num,
                            discount_factor,
                            epsilon,
                            lr,
                            save_plots,
                            save_data,
                            n_step_lookup,
                            H_paths,
                            job_num):

    random.seed(seed)
    np.random.seed(seed)
    
    
    table_size = available_actions[0].shape[0] # the size of each state and action vector.
    combinations = list(itertools.combinations(range(node_num), 2))
    # Convert each tuple to a list
    action_space_values = [list(pair) for pair in combinations]
    action_space_size = node_num-1
    best_states_size = 10
    best_states = {0: (initial_state, reward_cost(reward_list, initial_state))}

    visited_states = [np.array(initial_state)]
    visited_states = np.stack(visited_states)

    #Initialize the environment.
    env = Env(initial_state, # initial_state
             reward_list, # edge_weights
             episode_num, # total_episodes
             max_path_length,
             50, # show_path_num
             visited_states,  # visited_states
             available_actions, # basis_moves
             node_num, # node_num
             0, # P
             best_states,
             best_states_size,
             objective_table,
             False,
             discount_factor,
             reward_function = reward_cost
             )
    
    
    Q, agent_paths, ave_episode_reward = EdgeSwap_Q_learning(epsilon, 
                                                               episode_num, 
                                                               path_num, 
                                                               table_size, 
                                                               max_path_length, 
                                                               discount_factor, 
                                                               env, 
                                                               lr, 
                                                               save_plots, 
                                                               node_num,
                                                               action_space_values,
                                                               action_space_size,
                                                               n_step_lookup)
    
#     if save_data:
#         time = Time.localtime()
#         current_time = Time.strftime("%H-%M-%S", time)
#         date = datetime.now()
#         d = date.isoformat()[0:10]
#         data_save = [Q]
#         data_save = np.array(data_save, dtype=object)
#         print(d[0:10])
#         np.save('Models/'
#                 +'Q_EP_'
#                 +str(episode_num)+'_P_'+str(path_num)
#                 +'_PL_'+ str(max_path_length)+'_Date_'+d+'_.npy',data_save)
    
    
    result_dict = {"Q":Q, "agent_paths": agent_paths, "ave_episode_reward": ave_episode_reward}
    return result_dict





def initialize_enviornment(initial_states,
                            reward_list,
                            episode_num,
                            max_path_length,
                            available_actions,
                            node_num,
                            best_states,
                            best_states_size,
                            objective_table,
                            discount_factor):
    
    
    # Convert dictionary values to a list of arrays
    visited_states = [np.array(initial_states[0])]
    visited_states = np.stack(visited_states)

    #Initialize the environment.
    env = Env(initial_states[0], # initial_state
             reward_list, # edge_weights
             episode_num, # total_episodes
             max_path_length,
             50, # show_path_num
             visited_states,  # visited_states
             available_actions, # basis_moves
             node_num, # node_num
             0, # P
             best_states,
             best_states_size,
             objective_table,
             False,
             discount_factor,
             reward_function = reward_cost
             )
    
    return env










