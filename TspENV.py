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




class PolytopeENV(Env):
    """Class for a the Polytope environment.
    
            Given the current move A and current state table S, this
            environment produces the user defined reward for taking action A in state S 
            and evolves the environment into the next table state S' via the dynamics
            
                                    S' = S + A 
    """
    
    def __init__(self, 
                 initial_state,
                 edge_weights, 
                 total_episodes, 
                 show_path_num, 
                 visited_states, 
                 basis_moves,
                 node_num,
                 P,
                 lb,
                 reward_function = None):
        
        
        """Initialisation function

        Args:
            objective (list): Target table S* that we are interested in finding. 
                              We know the places in which to place the 0 values. 
                              We do not know the other entries.
            episode_length (int): number of steps to play the game for
        """
     
    
        self.initial_state = initial_state
        self.basis_move = basis_moves
        self.node_num = node_num
        self.path_num = [1]
        self.P = P
        self.lb = lb # lower bound on action coefficients.
        self.edge_weights = edge_weights
        self.visited_states = visited_states
        self.show_path_num = show_path_num
        self.total_episodes = total_episodes
        self.path = -1
        self.reward_func = reward_function
        
        
        self.reset()
        
        
        
    def reset(self):
        """Reset the environment.

        Returns:
            Give a randomly generated feasable table S as a starting point for the next episode.
        """
        
        if self.path % self.show_path_num == 0:
            #print("#####################################  Episode: ", self.episode, "  #####################################")
            print("After ", self.path, " paths, the agent found ", len(self.visited_states), " unique points in the fiber.")
            #print(self.visited_states)
        
        
        self.path += 1
        self._iteration = 0
        self._total_reward = 0
        self._closed_plot = False

        state_indx = 0
        if len(self.visited_states) > 1:
            state_indx = random.randint(0,len(self.visited_states)-1)
        state_str = self.visited_states[state_indx]
        state = np.frombuffer(state_str,dtype=int)
        self.state = state
        self.action = 0
        
        return self.initial_state #state
    

#     def reward_control(self,eps,decrement,slope):
#         eps -= decrement
#         return eps, m.exp(-slope*(1-eps))*eps
    
                
   
    def step(self, action):
        """Takes an action and the state and computes the immediate reward and evolves the
           environment into another table S'.

        Args:
            action (np.array): control variable 

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the environment that has evolved.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """
   
        action_rounded = np.array(np.round(action),dtype=int) # round off the coefficients
        all_actions = [ np.multiply(action_rounded[i],self.basis_move[i]) for i in range(len(action_rounded))] # create linear combination
        all_actions = np.stack(all_actions) # stack the scaled basis vectors
        self.action = np.sum(all_actions,0)  # compute the full action by adding the vectors in the stack

        
        self._iteration += 1  # update the iteration number
        done = False # True if the soultion is found
        found_solution = False # True if the soultion is found
        info = {}
        
        next_state = np.add(self.state, self.action) # update the state
        
        
        reward_feasibility = 0 # initialize the feasability reward.
        reward_non_zero_action = 0 # initialize the non-zero move reward.
        reward_diconnection = 0 # initialize connectivity reward.
        reward_direction = 0 # initialize the directional reward for cost optimization.
        
        
        
        # If action is the trivial zero action, penalize the agent and end the episode.
        if any(self.action) == False: 
            print("Action is a zero vector!")
            reward_non_zero_action = -1000
            next_state = self.state
        # Else if the action is not trivial.
        else:
            
            reward_non_zero_action = 10
            
            # Compute the feasability reward.
            if all(coord >= 0 for coord in next_state):  
                print("Next State is feasible!")
                state_g = self.create_state_graph(self.node_num, next_state)
                
                if self.check_connectivity(state_g, self.path_num[self.P]) == False:
                    print("Graph is disconnected")
                    reward_diconnection = -1000
                    next_state = self.state
                else:
                    print("Graph is connected")
                    reward_diconnection = 10
                    
                    if next_state.tobytes() not in self.visited_states:
                        self.visited_states.append(next_state.tobytes())
                        
                    reward_direction  = self.reward_func(self.edge_weights, next_state)  # compute the directional reward
            else:
                
                for coord in next_state:
                    if coord < 0:
                        reward_feasibility += 30*coord
                #print("Next State is unfeasible and reward is: ", reward_feasibility)
                next_state = self.state
                
          
        reward = reward_direction + reward_feasibility + reward_diconnection + reward_non_zero_action # combine the reward
        if reward > 0:
            print("WE HAVE A POSITIVE REWARD: ", self.state, self.action, next_state)
        self._total_reward += reward  # collect the reward
        
        
        self.state = next_state
            
        return self.state, reward, done, done, info
    
    def _handle_close(self, evt):
        self._closed_plot = True


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