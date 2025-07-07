import numpy as np
from operator import itemgetter

from EdgeSwap_Q_table_functions import getBestActionES
from helper_functions import create_state_edges
from reward_functions import reward_cost

def edge_swap_policy_evaluation(Q,env,reward_list,max_path_length,action_space_values,action_space_size, node_num):
                                
    path_reward = []
    step = 0
    state = env.reset()
    done = False
    path = [state]
    path_reward = [(reward_cost(reward_list, state),state,0)]
    
    for i in range(max_path_length):
        
        state_edges = create_state_edges(node_num, state)
        action = getBestActionES(Q, state, state_edges)
        print("--|||-- Current state cost: ", reward_cost(reward_list, state), " add action coeffs: ", action)
        
        next_state, reward, done, info = env.step(action)
        path_reward.append((reward_cost(reward_list, next_state),next_state,i))
        path.append(next_state)
        state = next_state
        
    min_pair = min(path_reward, key = itemgetter(0))
    print("Min Pair: ", min_pair)
    
    return path_reward
        
   
