import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
from operator import add
from gym import spaces, Env
import os
import math as m
import networkx as nx
import pickle
import sys
import copy

from helper_functions import reconnect_graph_generalized_version, reconnect_graph_universal, swap_and_reconnect_if_needed

class PolytopeENV(Env):
    def __init__(self, 
                 initial_state, 
                 edge_weights, 
                 total_episodes, 
                 max_path_length,
                 show_path_num, 
                 visited_states, 
                 basis_moves, 
                 node_num, 
                 P, 
                 best_states, 
                 best_states_size,
                 objective_table,
                 testing,
                 discount_factor,
                 reward_function=None):
        """
        Custom environment for Polytope problem, based on a multi-discrete action space.
        """
        super(PolytopeENV, self).__init__()

        # Environment settings
        self.initial_state = initial_state
        self.basis_move = basis_moves
        self.node_num = node_num
        self.path_num = [1]
        self.P = P
        self.edge_weights = edge_weights
        self.visited_states = visited_states
        self.show_path_num = show_path_num
        self.total_episodes = total_episodes
        self.max_path_length = max_path_length
        self.reward_func = reward_function
        self.episode = -1
        self.best_states = best_states
        self.best_states_size = best_states_size
        self.objective_table = objective_table
        self.discount_factor = discount_factor
        
        self.state =  self.initial_state
        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        self._iteration = 0
        self._total_reward = 0
        self.path = 0
        self.episode += 1
        
        
        # Start from a random visited state
        state_indx = 0
        keys_list = list(self.visited_states) # the key is an integer. 
        if len(keys_list) > 1:
            state_indx = random.randint(0, self.visited_states.shape[0]-1)
        state = self.visited_states[state_indx,:].tolist()
        self.state = state
#         self.state = self.initial_state
        return self.state  # Return the initial state self.initial_state#

    def step(self, action):
        """Take a step in the environment."""
        
       
        done = False
        found_solution = False
        info = {}
        
         
        # Compute reward components
        reward_double_edge = 0
        reward_direction = 0
        reward_adjecent_edges = 0 # essentially zero move. 
        connectivity_reward = 0
        
        if action[0] == action[1]:
            reward_double_edge = -1000
            print("Double Action: ", action)
            sys.exit()
        else:
            state_g = self.create_state_graph(self.node_num, self.state)
            state_edges = list(state_g.edges())
            state_edges = [(min(e), max(e)) for e in state_edges]
            state_edges = sorted(state_edges)
           
            if set(state_edges[action[0]]) & set(state_edges[action[1]]):
                reward_adjecent_edges = -1000
                print("The agent picked adjecent edges!: ", state_edges[action[0]], state_edges[action[1]])
                next_state = self.state
#                 sys.exit()
            else:
#                 state_edges = self.swap_edges_no_self_loops_and_check_connectivity(action, state_edges, self.node_num)
                state_edges, valid = swap_and_reconnect_if_needed(edge_list, e1, e2, edge_weights=None, structure="cycle")
                state_g = nx.Graph()
                state_g.add_edges_from(state_edges)
                next_state = self.create_state_vector(state_g)
#                 if self._iteration < self.max_path_length - 1:
#                     reward_direction  = -max((0, self.reward_func(self.edge_weights, next_state)-self.reward_func(self.edge_weights, self.state)))
#                     print(f"Compute directional reward: {reward_direction}")
#                 else:
#                     reward_direction = -self.reward_func(self.edge_weights, next_state)
#                     print(f"Compute the last reward: {reward_direction}")

#                 reward_direction  = -max((0, self.reward_func(self.edge_weights, next_state)-self.reward_func(self.edge_weights, self.state)))
                print(state_edges)
                print(action)
                reward_direction = -self.reward_func(self.edge_weights, next_state)
                print(f"Compute directional reward: {reward_direction} for state: {next_state}, {state_edges}, ",action,  state_edges[action[0]], state_edges[action[1]], flush=True)
 
                self.best_states = self.update_best_states(self.best_states, self.best_states_size, next_state, reward_direction)
                if next_state.tolist() not in self.visited_states.tolist():  
                    self.visited_states = np.concatenate((self.visited_states,[next_state]),axis=0)


        reward = reward_direction + reward_double_edge + reward_adjecent_edges 
        self._total_reward += self.discount_factor**self._iteration * reward #reward_normalized
        self._iteration += 1
        
        # Define a done condition (e.g., maximum iterations)
        if self._iteration % self.max_path_length == 0:  # You can define a suitable condition based on your problem
            print(f'Episode: {self.episode} ||| Reward: {self._total_reward} ||| Discovered States: {len(self.visited_states)}')
            done = True

        if self.episode >= self.total_episodes:
            with open('Models/best_states.pkl', 'wb') as f: # save the N best states we found. 
                pickle.dump(self.best_states, f)
            with open('Models/visited_states.pkl', 'wb') as f: # save the N best states we found. 
                pickle.dump(self.visited_states, f)    
            
            
        self.state = next_state
        return self.state, reward, done, info
    
    def _handle_close(self, evt):
        self._closed_plot = True

    # Given a state graph, construct a corresponding vector.
    def create_state_vector(self, g):
        adjacency = nx.to_numpy_array(g, sorted(list(g.nodes())), dtype=int)
        n_rows, n_cols = adjacency.shape
        upper_mask = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=1)
        upper_diagonal = adjacency[upper_mask]
        flattened_vector = upper_diagonal.flatten()
        init_sol = np.array(flattened_vector)
        return init_sol
    
    # Given a state vector, construct a corresponding graph.
    def create_state_graph(self, num_nodes, state):
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        index = 0
        for n in range(num_nodes-1,-1,-1):
            for i in range(n+1):
                if num_nodes-1-n+i > num_nodes-1-n:
                    if state[index] != 0:
                        adj_matrix[num_nodes-1-n, num_nodes-1-n+i] = state[index]
                    #print(num_nodes-1-n, num_nodes-1-n+i)
                    index += 1
                    
        adj_matrix += np.triu(adj_matrix, 1).T
        MG = nx.MultiGraph()
        for i in range(num_nodes):
            for j in range(i, num_nodes):  # Iterate over upper triangular part including diagonal
                for _ in range(adj_matrix[i, j]):
                    MG.add_edge(i, j)
        return MG 
    
    # Check whether the state grah has the appropriate number of connected components. 
    def check_connectivity(self, g, p_num):
        conn_components = list(nx.connected_components(g))
        if len(conn_components) == p_num:
            return True
        else:
            return False
        
        
    def update_best_states(self, best_states, dict_size, state, state_directional_rew):
        
        keys = list(best_states.keys())
        states = [s[0].tolist() for s in best_states.values()]
        if state.tolist() in states: # do not reapet the same states. 
            return best_states
        #find the worst state in the dicitonary.
        key_with_min_float = min(best_states, key=lambda k: best_states[k][1])
        min_value = best_states[key_with_min_float][1]
        # Check for uniqness of states.
        if len(keys) >= dict_size:
            if min_value < state_directional_rew:  
                print("The BEST STATES DICT is full, and we found a better state, remove the worst state ", len(keys))
                best_states.pop(key_with_min_float)
                best_states[key_with_min_float] = (state, state_directional_rew)
        else:
            print("The BEST STATES DICT is not full, add state at key ", max(keys)+1)
            best_states[max(keys)+1] = (state, state_directional_rew)
        return best_states
    
    
    
    
    def swap_edges_no_self_loops_and_check_connectivity(self, action, state_edges, num_nodes):
        # Get the edges based on the action indices
        e1 = state_edges[action[0]]
        e2 = state_edges[action[1]]

        # Remove the chosen edges from the list
        state_edges.remove(e1)
        state_edges.remove(e2)

        # Swap the nodes between the two edges
        new_e_1 = (e1[0], e2[1])
        new_e_2 = (e1[1], e2[0])

        # Ensure the edges are in the (min, max) format
        new_e_1 = (min(new_e_1), max(new_e_1))
        new_e_2 = (min(new_e_2), max(new_e_2))

        # Check for self-loops
        if new_e_1[0] != new_e_1[1] and new_e_2[0] != new_e_2[1]:
            # Temporarily create a new graph with the swapped edges
            temp_edges = state_edges + [new_e_1, new_e_2]
            temp_g = nx.MultiGraph()
            temp_g.add_edges_from(temp_edges)
            # Check for connectivity
            if not self.check_connectivity(temp_g, 1): #self.path_num[self.P]
                # Fix the state graph in a minimal way.
                new_state_g = reconnect_graph_generalized_version(temp_g, self.objective_table)
                state_edges = sorted(list(new_state_g.edges()))
            else:
                state_edges.append(new_e_1)
                state_edges.append(new_e_2)
        else:
            # If self-loops are detected, pick the second swap
            new_e_1 = (e1[0], e2[0])
            new_e_2 = (e1[1], e2[1])
              
            # Temporarily create a new graph with the swapped edges
            temp_edges = state_edges + [new_e_1, new_e_2]
            temp_g = nx.MultiGraph()
            temp_g.add_edges_from(temp_edges)
            
            if not self.check_connectivity(temp_g, self.path_num[self.P]):
                new_state_g = reconnect_graph_generalized_version(temp_g, self.objective_table)
                state_edges = sorted(list(new_state_g.edges()))

            else:
                state_edges.append(new_e_1)
                state_edges.append(new_e_2)
            
        # Sort the edges to maintain consistent ordering
        state_edges_ = [(min(e), max(e)) for e in state_edges]
        state_edges = sorted(state_edges_)

        return state_edges
    
    
    




  