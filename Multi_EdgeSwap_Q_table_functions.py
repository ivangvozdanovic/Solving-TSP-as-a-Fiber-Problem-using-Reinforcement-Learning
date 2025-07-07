import numpy as np
import random
from helper_functions import share_element, ToBytes, FromBytes

'''
Get a Q value given state and action. 
If the state has not been seen before, the Q value is set to zero.
If the action for the given state has not been used before, return Q value = 0.
'''
def getQvalueMultiES(Q, state, action):
    state_keys = list(Q.keys())
    state_str = ToBytes(state)
    action_str = ToBytes(action)
    if state_str not in state_keys:
        return 0
    else:
        action_keys = list(Q[state_str].keys())
        if action_str not in action_keys:
            return 0
        else:
            return Q[state_str][action_str]
        

        
'''
Get action which produces maximum Q value given a state.
If the given state has not been seen before, return a random action.
'''
def getBestActionMultiES(Q, state, state_edges, k_swaps):
    state_keys = list(Q.keys())
    state_str = ToBytes(state)
    if state_str not in state_keys:
        action = []
        e = (100,101)
        a_list = [100]
        for k in range(k_swaps):
            valid_indices = [i for i, tup in enumerate(state_edges) if i not in a_list and share_element(e, tup) == False]
            a = random.choice(valid_indices)
            a_list.append(a)
            e = state_edges[a]
            action.append(a)
        action = sorted(action)
       
    else:
        
        action_keys = list(Q[state_str].keys())
        max_action_str = action_keys[0]
        max_q_value = Q[state_str][max_action_str]
        for q in range(len(action_keys)):
            if share_element(state_edges[FromBytes(max_action_str)[0]], state_edges[FromBytes(max_action_str)[1]], state_edges[FromBytes(max_action_str)[0]]) == False:
                if max_q_value < Q[state_str][action_keys[q]]:
                    max_action_str = action_keys[q]
                    max_q_value =  Q[state_str][action_keys[q]]
        action = FromBytes(max_action_str)
    return action

'''Update the Q table at the point given by (state,action) by value update_value.'''
def updateQdictMultiES(Q,state,action,update_value):
    state_keys = list(Q.keys())
    state_str = ToBytes(state)
    action_str = ToBytes(action)
    if state_str not in state_keys:
        Q[state_str] = {action_str : update_value}
    else:
        action_keys = list(Q[state_str].keys())
        if action_str not in action_keys:
            Q[state_str][action_str] = update_value
        else:
            Q[state_str][action_str] += update_value





'''Convert the Q dicitonary with bytes into a dictionary with vectors'''
def Qbytes2vectors(Q):
    print("hello")