import numpy as np
import networkx as nx

from helper_functions import share_element, ToBytes, FromBytes
from Q_table_functions import getQvalue, getBestAction, updateQdict, updateQpath, checkAction, createNewAction

import random

def edge_swap_policy(Q, state, state_edges, epsilon, node_num):
    prob = np.random.rand()
    if prob < epsilon:
        action1 = random.randint(0,len(state_edges)-1)
        edge1 = state_edges[action1]
        valid_indices = [i for i, tup in enumerate(state_edges) if i != action1 and share_element(edge1, tup) == False]
#         valid_indices = [i for i, tup in enumerate(state_edges) if i != action1]
        action2 = random.choice(valid_indices)
        action = [action1, action2]
        action = sorted(action)
    else:
        state_str = ToBytes(state)
        state_keys = list(Q.keys())
        if state_str in state_keys:
            action_keys = list(Q[state_str].keys())

            max_action_str = action_keys[0]
            max_q_value = Q[state_str][max_action_str]
            for q in range(len(action_keys)):
                if share_element(state_edges[FromBytes(max_action_str)[0]], state_edges[FromBytes(max_action_str)[1]]) == False:
                    if max_q_value < Q[state_str][action_keys[q]]:
                        max_action_str = action_keys[q]
                        max_q_value =  Q[state_str][action_keys[q]]
            action = FromBytes(max_action_str)
            
        else:
            action1 = random.randint(0,len(state_edges)-1)
            edge1 = state_edges[action1]
            valid_indices = [i for i, tup in enumerate(state_edges) if i != action1 and share_element(edge1, tup) == False]
            action2 = random.choice(valid_indices)
            action = [action1, action2]
            action = sorted(action)
    return action




# def edge_swap_policy(Q, state, state_edges, epsilon, node_num, mode="cycle"):

#     prob = np.random.rand()

#     if prob < epsilon:
#         # --- Exploration: randomly try valid swaps ---
#         while True:
#             action1 = random.randint(0, len(state_edges) - 1)
#             action2 = random.randint(0, len(state_edges) - 1)

#             if action1 == action2:
# #                 print("Action not valid - double edge")
#                 continue

#             e1 = state_edges[action1]
#             e2 = state_edges[action2]
            
#             if set(e1) & set(e2):
# #                 print("Action not valid - adjecent")
#                 continue

#             if is_valid_swap(state_edges, e1, e2, node_num, mode=mode):
# #                 print(f"Valid Random Action: Action: {action1},{action2} Swap {e1} and {e2} from state edges: {state_edges}")
#                 return [action1, action2]
#             else:
#                 print("Not a valid swap: ", e1, e2, state_edges)
#     else:
#         # --- Exploitation: choose best valid action from Q-table ---
#         state_str = ToBytes(state)

#         if state_str in Q:
#             action_keys = list(Q[state_str].keys())

#             best_action = None
#             best_q = -float('inf')

#             for a_str in action_keys:
#                 action = FromBytes(a_str)
#                 if len(action) != 2:
#                     continue
#                 a1, a2 = action

#                 if a1 >= len(state_edges) or a2 >= len(state_edges):
#                     continue
                
#                 if a1 == a2:
#                     print("Greedy Action not valid - double edge")

                
#                 e1 = state_edges[a1]
#                 e2 = state_edges[a2]
                   
#                 if is_valid_swap(state_edges, e1, e2, node_num, mode=mode):
#                     q_val = Q[state_str][a_str]
#                     if q_val > best_q:
#                         best_q = q_val
#                         best_action = action

#             if best_action is not None:
# #                 print(f"Valid Greedy Action: Action: {a1},{a2} Swap {e1} and {e2} from state edges: {state_edges}")
#                 return sorted(best_action)

#         # --- Fallback: random valid swap ---
#         while True:
#             print("Falling back to a random action")
#             action1 = random.randint(0, len(state_edges) - 1)
#             action2 = random.randint(0, len(state_edges) - 1)

#             if action1 == action2:
#                 continue

#             e1 = state_edges[action1]
#             e2 = state_edges[action2]
            
#             if set(e1) & set(e2):
# #                 print("Action not valid - adjecent")
#                 continue

#             if is_valid_swap(state_edges, e1, e2, node_num, mode=mode):
# #                 print(f"Second Valid Random Action: Action: {action1},{action2} Swap {e1} and {e2} from state edges: {state_edges}")
#                 return [action1, action2]
#             else:
#                 print("Not a valid swap: ", e1, e2, state_edges)


            
            
def is_valid_hamiltonian_path(edges, node_num):
    G = nx.Graph()
    G.add_edges_from(edges)

    if not nx.is_connected(G):
        return False

    degrees = dict(G.degree())
    degree_counts = list(degrees.values())
    
    return (
        degree_counts.count(1) == 2 and  # exactly 2 endpoints
        degree_counts.count(2) == node_num - 2
    )


def is_valid_two_paths(edges, node_num):
    G = nx.Graph()
    G.add_edges_from(edges)

    if not nx.is_forest(G):  # no cycles allowed
        return False

    components = list(nx.connected_components(G))
    if len(components) != 2:
        return False

    for comp in components:
        subgraph = G.subgraph(comp)
        degs = list(dict(subgraph.degree()).values())
        if degs.count(1) != 2 or all(d <= 2 for d in degs) == False:
            return False

    return True


# def is_valid_swap(state_edges, e1, e2, node_num, mode="path"):
#     G = nx.Graph()
#     G.add_edges_from(state_edges)

#     if e1 not in G.edges or e2 not in G.edges:
#         return False

#     G.remove_edge(*e1)
#     G.remove_edge(*e2)

#     new1 = (e1[0], e2[1])
#     new2 = (e2[0], e1[1])
#     for new_e in [new1, new2]:
#         if new_e[0] == new_e[1] or G.has_edge(*new_e):
#             return False

#     G.add_edge(*new1)
#     G.add_edge(*new2)

#     new_edges = list(G.edges())

#     if mode == "cycle":
#         return nx.is_connected(G) and all(deg == 2 for _, deg in G.degree())
#     elif mode == "path":
#         return is_valid_hamiltonian_path(new_edges, node_num)
#     elif mode == "2-paths":
#         return is_valid_two_paths(new_edges, node_num)
#     else:
#         raise ValueError(f"Unknown mode: {mode}")    


def is_valid_swap(state_edges, e1, e2, node_num, mode="path"):
    def try_swap(new_e1, new_e2):
        G = nx.Graph()
        G.add_edges_from(state_edges)

        if e1 not in G.edges or e2 not in G.edges:
            return False

        G.remove_edge(*e1)
        G.remove_edge(*e2)

        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            return False

        if G.has_edge(*new_e1) or G.has_edge(*new_e2):
            return False

        G.add_edge(*new_e1)
        G.add_edge(*new_e2)

        if mode == "cycle":
            return nx.is_connected(G) and all(deg == 2 for _, deg in G.degree())
        elif mode == "path":
            return is_valid_hamiltonian_path(G.edges(), node_num)
        elif mode == "2-paths":
            return is_valid_two_paths(G.edges(), node_num)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Two possible rewiring options for 2-opt
    swap_options = [
        ((e1[0], e2[1]), (e2[0], e1[1])),
        ((e1[0], e2[0]), (e1[1], e2[1])),
    ]

    return any(try_swap(new1, new2) for new1, new2 in swap_options)



            
            

def multi_edge_swap_policy(Q, state, state_edges, epsilon, k_swaps):
    prob = np.random.rand()
    if prob < epsilon:
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
        
        state_str = ToBytes(state)
        state_keys = list(Q.keys())
        if state_str in state_keys:
            action_keys = list(Q[state_str].keys())
            max_action_str = action_keys[0]
            max_q_value = Q[state_str][max_action_str]
            for q in range(len(action_keys)):
                if share_element(state_edges[FromBytes(max_action_str)[0]], state_edges[FromBytes(max_action_str)[1]], state_edges[FromBytes(max_action_str)[2]]) == False:
                    if max_q_value < Q[state_str][action_keys[q]]:
                        max_action_str = action_keys[q]
                        max_q_value =  Q[state_str][action_keys[q]]
            action = FromBytes(max_action_str)
            
        else:
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
    return action


'''
Time-independent epsilon-greedy policy:
    Pick a random action with probability epsilon and the best current action with probability 1-epsilon.
'''
def time_independent_policy(Q, epsilon, state, available_actions, lb, ub):
    reset_game = False
    prob = np.random.rand()
    action = None
    if prob < epsilon:
        action_coeffs = np.random.randint(lb, ub, len(available_actions)) 
    else:
        action_coeffs = getBestAction(Q,state,available_actions, lb, ub) #Best Action
    return action_coeffs




'''Given a state table, only consider the feasable actions which won't produce negative values in the next state table.'''
def filter_actions(state, available_actions):
    good_actions = []
    for a in available_actions:
        next_state = state + a
        min_entry = min(next_state)
        max_entry = max(next_state)
        if min_entry < 0:
            next_state = None
            continue
        else:
            good_actions.append(a)
    return good_actions


''' True if the given action is feasable for the given state. '''
def check_feasability(state,action):
    for i in range(len(state)):
        if abs(state[i]) >= abs(action[i]):
            continue
        else:
            return False
    return True


