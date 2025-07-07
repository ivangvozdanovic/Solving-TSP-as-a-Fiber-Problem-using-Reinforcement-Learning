import numpy as np
from numpy.random import default_rng
import random
import os
import shutil
import re
import sys
from operator import itemgetter
import time as Time
import networkx as nx
from datetime import datetime
import ast
import matplotlib.pyplot as plt
import scipy
import copy
import math
import itertools
from itertools import permutations, combinations
from more_itertools import distinct_permutations as idp
from collections import Counter
import pandas as pd
import pickle
from sympy import Matrix, eye, ImmutableSparseMatrix
import itertools
from itertools import permutations, combinations
from bitarray import bitarray
import struct


def ToBytes(data):
    format_string = f'{len(data)}I'  # 'I' means unsigned 4-byte integer
    byte_data = struct.pack(format_string, *data)
    return byte_data


def FromBytes(byte_data):
    num_integers = len(byte_data) // 4  # Each integer is 4 bytes
    format_string = f'{num_integers}I'
    return list(struct.unpack(format_string, byte_data))
    
    
# Create a fully connected graph with randomly assigned weights.
def create_fully_connected_graph(num_nodes,positive_w, p, weight_min, weight_max):
#     # Create a complete graph with the specified number of nodes
#     G = nx.complete_graph(num_nodes)

#     # Assign random weights to each edge
#     if positive_w == True:
#         for (u, v) in G.edges():
#             G.edges[u, v]['weight'] = random.uniform(weight_min, weight_max)  # Random weight between 1 and 10
#     else:
#         for (u, v) in G.edges():
#             G.edges[u, v]['weight'] = -1*random.uniform(weight_min, weight_max)  # Random weight between 1 and 10
#     return G
        # Create a complete graph with the specified number of nodes
    G = nx.complete_graph(num_nodes)
    
    # Calculate the total number of edges in the complete graph
    num_edges = G.number_of_edges()
    
    # Generate unique integer weights for all edges
    if positive_w:
        weights = random.sample(range(weight_min, weight_max + 1), num_edges)  # Unique positive integer weights
    else:
        weights = random.sample(range(-weight_max, -weight_min + 1), num_edges)  # Unique negative integer weights

    # Shuffle the weights to randomize the assignment
    random.shuffle(weights)
    
    # Assign the unique weights to each edge
    for i, (u, v) in enumerate(G.edges()):
        G.edges[u, v]['weight'] = weights[i]

    return G


# Compute the sublist of rewards corresponding to the edges which can be obtained in a subproblem.
def compute_reward_map(rewards, nodes):
    edges = list(combinations(nodes, 2))
    print("Edges for reward list: ", edges)
    reward_list = [rewards[e] for e in edges]
    return reward_list


# Given the fully connected graph with random weights, extract the weights and edges into a dict.
def extract_weights_and_edges(G):
    edge_weights = {(u, v): w['weight'] for u, v, w in G.edges(data=True)}
    return edge_weights

# Given the dicitonary of edges and their weights, extract the weights into a distance matrix.
def extract_distance_matrix(edges):
    nodes = set()
    for edge in edges:
        nodes.update(edge)
    num_nodes = max(nodes) + 1
    adj_matrix = np.zeros((num_nodes, num_nodes)) # Initialize the numpy array with zeros
    for (u, v), weight in edges.items(): # Populate the numpy array with edge weights
        adj_matrix[u, v] = weight
        adj_matrix[v, u] = weight
    return adj_matrix

# Given a state graph, construct a corresponding vector.
def create_state_vector(g):
    adjacency = nx.to_numpy_array(g, sorted(list(g.nodes())), dtype=int)
    n_rows, n_cols = adjacency.shape
    upper_mask = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=1)
    upper_diagonal = adjacency[upper_mask]
    flattened_vector = upper_diagonal.flatten()
    init_sol = np.array(flattened_vector)
    return init_sol


def create_design_mat(nodes_num):
    nodes = [i for i in range(nodes_num)]
    G = nx.complete_graph(nodes_num)
    incidence_matrix = nx.incidence_matrix(G, oriented=False)
    design_mat = incidence_matrix.toarray()
    return design_mat


def init_solution(adjacency):
    #Get the vertex-incidence matrix for initial solutions.
    # Get the dimensions of the array
    adjacency = adjacency.toarray()
    n_rows, n_cols = adjacency.shape
    # Create a mask for the upper diagonal part
    upper_mask = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=1)
    # Apply the mask to the array to get the upper diagonal part
    upper_diagonal = adjacency[upper_mask]
    # Flatten the upper diagonal part into a vector
    flattened_vector = upper_diagonal.flatten()
    init_sol = np.array(flattened_vector)
    return init_sol, upper_diagonal


# Given a state vector, construct a corresponding graph.
def create_state_graph(num_nodes, state):
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

def create_state_edges(node_num, state):
    state_g = create_state_graph(node_num, state)
    edges = list(state_g.edges())
    edges_new = [(min(e), max(e)) for e in edges]
    state_edges = sorted(edges_new)
    return state_edges


def convert_sym_to_np(actions):
    action_list = []
    for action in actions:
        a = np.array(action).astype(np.int8)
        a = np.transpose(a)
        action_list.append(a[0])
    return action_list


def extract_lattice_basis_sparse(design_matrix):
    M = Matrix(design_matrix)
    return M.nullspace()


    
# def share_element(tuple1, tuple2):
#     # Check if any element is shared between two tuples
#     return bool(set(tuple1) & set(tuple2))

def share_element(*tuples):
    # Start with the set of the first tuple
    shared_elements = set(tuples[0])
    
    # Iterate over the rest of the tuples and intersect with the shared elements
    for tup in tuples[1:]:
        shared_elements &= set(tup)
        
        # If no shared elements are left, we can exit early
        if not shared_elements:
            return False
            
    # Return True if there are any shared elements, False otherwise
    return bool(shared_elements)


# Determins whether the state vector corresponding to a subgraph is connected.
def is_connected_from_incidence_vector(incidence_vector, num_nodes):
    # Initialize an adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Index for the incidence vector
    index = 0

    # Populate the adjacency matrix from the incidence vector
    for i, j in combinations(range(num_nodes), 2):
        if incidence_vector[index] != 0:
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
        index += 1

    # Initialize visited nodes list
    visited = [False] * num_nodes

    # Function for DFS
    def dfs(v):
        visited[v] = True
        for neighbor in range(num_nodes):
            if adj_matrix[v][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor)

    # Start DFS from the first node
    dfs(0)
    # Check if all nodes are visited
    return all(visited)


def find_cycle(start_node, neighbours, edge_weight_list, G):

    cycle_nodes = [start_node]
    cycle_edges = []

    sorted_neighbours = sorted(neighbours[start_node])
    restricted_edges = [(start_node,n) for n in sorted_neighbours]#combinations(sorted_restricted_neighbours, 2)
    restricted_edges = [(min(tup), max(tup)) for tup in restricted_edges]
    indx = np.random.randint(0, len(restricted_edges))
    edge = restricted_edges[indx]
    n1 = edge[0]
    n2 = edge[1]
    if n1 == start_node:
        current_node = n2
    elif n2 == start_node:
        current_node = n1
    prev_node = start_node

    cycle_nodes.append(current_node)
    cycle_edges.append((prev_node, current_node))

    while current_node != start_node:

        # Randomly pick next node in the cylce s.t it is not the previous node.
        current_neighbours = neighbours[current_node]
        restricted_neighbours = [n for n in current_neighbours if n not in cycle_nodes]

        if len(restricted_neighbours) == 0:
            cycle_edges.append((current_node, start_node))
            return cycle_nodes, cycle_edges

        sorted_restricted_neighbours = sorted(restricted_neighbours)
        restricted_edges = [(current_node,n) for n in sorted_restricted_neighbours]#combinations(sorted_restricted_neighbours, 2)
        restricted_edges = [(min(tup), max(tup)) for tup in restricted_edges]
        indx = np.random.randint(0, len(restricted_edges))
        edge = restricted_edges[indx]
        n1 = edge[0]
        n2 = edge[1]

        prev_node = current_node

        if n1 == current_node:
            current_node = n2
        elif n2 == current_node:
            current_node = n1

        cycle_nodes.append(current_node)
        cycle_edges.append((prev_node, current_node))

    return cycle_nodes, cycle_edges



def create_fiber_sampling_erdos_renyi_graph(file, initial_states, total_nodes, p, graph_num):
    
    for g in range(graph_num):
        G = nx.erdos_renyi_graph(total_nodes, p, seed=None, directed=False)

        node_neighbours = {n: list(G.neighbors(n)) for n in list(G.nodes())}
        sort_nodes = sorted(list(G.nodes()))

        adj = nx.adjacency_matrix(G, nodelist=sort_nodes) # adjecency matrix
        init_sol, upper_diagonal = init_solution(adj)
        dm = create_design_mat(len(sort_nodes))
        initial_states[g] = init_sol
        margin = np.dot(dm, init_sol)
        print("Initial solution: \n", len(init_sol), init_sol)
        print("Sufficient statistic: \n", margin)
    
    available_actions = extract_lattice_basis_sparse(dm) # get the lattice basis out of the design matrix.
    available_actions = convert_sym_to_np(available_actions) # convert to numpy.
    print(f'Number of actions is {len(available_actions)}')
    
    date = datetime.now()
    d = date.isoformat()[0:10]
    problem_name = file + "_Node_" + str(total_nodes)
    directory_name = problem_name + '_Date_' + d
    
    return available_actions, initial_states
    
def create_tsp_polytope_graph(nodes_per_patch, 
                          patches, 
                          initial_states, 
                          reward_lists,
                          file,
                          reward_cost,
                          weight_min,
                          weight_max):
    
    G = create_fully_connected_graph(nodes_per_patch, True, 0, weight_min, weight_max)
    
        
   
    #nx.draw(G, with_labels=True)
    #plt.show()

    # extract all edges and weights into a dict. Used for reward.
    edge_weights = extract_weights_and_edges(G)
    objective_table = edge_weights
    #print("Obj table: \n", objective_table)

    node_neighbours = {n: list(G.neighbors(n)) for n in list(G.nodes())}

    # compute the distance matrix for the integer program.
    distance_matrix = extract_distance_matrix(edge_weights)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j and distance_matrix[i, j] == 0:
                distance_matrix[i, j] = 10000

    path_nodes, path_edges = find_cycle(0, node_neighbours, edge_weights, G)

    print("Path nodes: \n", path_nodes)
    print("Path edges: \n", path_edges)

    sub_g = nx.Graph()
    sub_g.add_edges_from(path_edges)

    sort_nodes = sorted(list(sub_g.nodes()))
    print("sub nodes: \n", list(sub_g.nodes()))
    print("sub sorted nodes: \n", sort_nodes)
    print("sub edges: \n", list(sub_g.edges()))


    #subgraphs.append(sub_g)


    adj = nx.adjacency_matrix(sub_g, nodelist=sort_nodes) # adjecency matrix
    init_sol, upper_diagonal = init_solution(adj)
    dm = create_design_mat(len(sort_nodes))
    initial_states[0] = init_sol
    print("Initial solution: \n", len(init_sol), init_sol)

    is_connected = is_connected_from_incidence_vector(init_sol, len(sort_nodes))
    print("is connected: ", is_connected)
    print("Is graph connected: ", nx.is_connected(sub_g))


    reward_list = compute_reward_map(objective_table, sort_nodes)
    print("Sub reward list: \n", len(reward_list), reward_list)
    reward_lists.append(reward_list)
    print("Initial reward: ", reward_cost(reward_list, init_sol))


    margin = np.dot(dm, init_sol)
    print("Margin: \n", margin)
#     if adj.shape[0] not in list(all_size_basis_action.keys()):
    available_actions = extract_lattice_basis_sparse(dm) # get the lattice basis out of the design matrix.
    available_actions = convert_sym_to_np(available_actions) # convert to numpy.
#         all_size_basis_action[adj.shape[0]] = available_actions
    print(f'Number of actions is {len(available_actions)}')


    date = datetime.now()
    d = date.isoformat()[0:10]
    problem_name = file + "_Node_" + str(nodes_per_patch * patches)
    directory_name = problem_name + '_Date_' + d
    
    
    return available_actions, initial_states, distance_matrix, objective_table, reward_list






def generate_combinations_4(tuples, edge_cost):

        comb_reward_dict = {}
        # Create a set of original pairs for quick lookup
        original_pairs = {tuple(sorted(t)) for t in tuples}

        # Generate all unique elements from the tuples
        all_elements = set()
        for tup in tuples:
            all_elements.update(tup)

        all_elements = list(all_elements)
        n = len(tuples)

        unique_combinations = set()

        # Generate all permutations of elements
        for perm in permutations(all_elements, n * 2):
            # Split the permutation into chunks of 2
            chunks = [perm[i:i+2] for i in range(0, len(perm), 2)]

            # Ensure each chunk has unique elements
            if all(len(set(chunk)) == len(chunk) for chunk in chunks):
                # Ensure elements within chunks are unique and not from original pairs
                if all(tuple(sorted(chunk)) not in original_pairs for chunk in chunks):
                    # Ensure overall uniqueness of elements
                    flat_combination = [item for sublist in chunks for item in sublist]
                    if len(flat_combination) == len(set(flat_combination)):
                        sorted_combination = tuple(tuple(sorted(chunk)) for chunk in chunks)
                        unique_combinations.add(sorted_combination)
                        reward = 0
                        for comb in sorted_combination:
                            reward += edge_cost[comb]
                        #reward = edge_cost[sorted_combination[0]] + edge_cost[sorted_combination[1]] + edge_cost[sorted_combination[2]]
                        comb_reward_dict[sorted_combination] = reward
        return list(unique_combinations), comb_reward_dict




def reconnect_graph_generalized_version(g, objective_table):
 
    if nx.is_connected(g) == False:

        all_edges = list(g.edges())
        all_nodes = list(g.nodes())
        all_edges = [(min(e),max(e)) for e in all_edges]
        old_edge = all_edges.copy()

        conn_components = list(nx.connected_components(g))
        components = [g.subgraph(component).copy() for component in conn_components]
        components = sorted(components, key=lambda x: x.number_of_edges() )
        component_num = len(components)


        min_pair = None
        min_reward = 0
        min_reward_diff = 0
        edges_2_remove = None


        ranges = [len(list(components[i].edges())) for i in range(component_num)]
  
        loops = [range(r) for r in ranges]
        for comb in itertools.product(*loops):
            # for each edge index do the rest.
            tup = []
            for j in range(len(comb)):
                e = list(components[j].edges())[comb[j]]
                e = (e[0], e[1])
                e = (min(e), max(e))
                tup.append(e)
            
            original_tup_rew = 0
            for e in tup:
                original_tup_rew += objective_table[e]
            combinations, comb_reward_dict = generate_combinations_4(tup, objective_table)

            for i in range(len(combinations)):

                if min_pair == None:
                    if i == 0:
                        min_pair = combinations[i]
                        min_reward = comb_reward_dict[min_pair]
                        min_reward_diff = abs(original_tup_rew - min_reward)
                        edges_2_remove = tup
                else:
                    # if the (combination_reward - original comb reward.)
                    if abs(comb_reward_dict[combinations[i]] - original_tup_rew) < min_reward_diff:
                        min_pair = combinations[i]
                        min_reward = comb_reward_dict[combinations[i]]
                        edges_2_remove = tup

        edges_2_remove_ordered = [(min(e),max(e)) for e in edges_2_remove]
   
        for e in edges_2_remove_ordered:
            all_edges.remove(e)
        for e in list(min_pair):
            all_edges.append(e)
        all_edges = sorted(all_edges, key=lambda x: x[0] )
        new_state_g = nx.empty_graph()
        new_state_g.add_edges_from(all_edges)

        total_rew_new = 0
        for e in all_edges:
            total_rew_new += objective_table[e]

        total_rew_old = 0
        for e in old_edge:
            total_rew_old += objective_table[e]

        if nx.is_connected(new_state_g) == False:
            print("GRAPH IS STILL DISCONNECTED")
            sys.exit()
    else:
        new_state_g = g
            
    return new_state_g



def swap_and_reconnect_if_needed(edge_list, e1, e2, edge_weights=None, structure="cycle"):
    """
    Swap two edges in a graph and reconnect if the resulting graph becomes improperly disconnected.

    Parameters:
        edge_list (list of tuple): Existing edges of the graph.
        e1, e2 (tuple): Edges to be swapped. Must exist in edge_list.
        edge_weights (dict): Optional dict mapping edge tuples to weights.
        structure (str): One of {"cycle", "path", "2paths"} to define expected connectivity.

    Returns:
        new_edge_list (list of tuple): Possibly modified edge list.
        was_reconnected (bool): Whether reconnection was necessary.
    """
    assert e1 in edge_list and e2 in edge_list, "Edges to swap must exist in edge list"
    assert structure in {"cycle", "path", "2paths"}, "Unsupported structure type"

    # Remove the selected edges
    edge_list_copy = edge_list.copy()
    edge_list_copy.remove(e1)
    edge_list_copy.remove(e2)

    # Generate swap options
    swap_options = [
        ((e1[0], e2[1]), (e1[1], e2[0])),
        ((e1[0], e2[0]), (e1[1], e2[1]))
    ]
    
    # Try both swap options
    for new_e1, new_e2 in swap_options:
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue  # Skip self-loops

        candidate_edges = edge_list_copy + [new_e1, new_e2]
        G = nx.Graph()
        G.add_edges_from(candidate_edges)

        num_components = nx.number_connected_components(G)

        # Check if the graph still satisfies the required structure
        if (structure in {"cycle", "path"} and num_components == 1) or \
           (structure == "2paths" and num_components == 2):
            return candidate_edges, False  # No reconnection needed

    # Swap failed, need to reconnect the graph
    G = nx.Graph()
    G.add_edges_from(edge_list_copy)
    components = list(nx.connected_components(G))

    target_components = 1 if structure in {"cycle", "path"} else 2

    # Reconnect using minimal-weight edges
    while len(components) > target_components:
        min_weight = float('inf')
        best_edge = None

        for c1, c2 in itertools.combinations(components, 2):
            for u in c1:
                for v in c2:
                    if u == v:
                        continue
                    weight = (
                        edge_weights.get((u, v)) or
                        edge_weights.get((v, u)) or
                        1  # default weight
                    )
                    if weight < min_weight:
                        min_weight = weight
                        best_edge = (u, v)

        if best_edge:
            G.add_edge(*best_edge)
            components = list(nx.connected_components(G))
        else:
            raise ValueError("No valid edge found to reconnect components.")

    return list(G.edges()), True



def reconnect_graph_universal(g, objective_table, min_node, path_num, start_nodes, end_nodes, forbidden_connections):

    if path_num == 1:

        if nx.is_connected(g) == False:
            
            all_edges = list(g.edges())
            all_nodes = list(g.nodes())
            all_edges = [(min(e)+min_node,max(e)+min_node) for e in all_edges]
            old_edge = all_edges.copy()
        
            conn_components = list(nx.connected_components(g))
            components = [g.subgraph(component).copy() for component in conn_components]
            components = sorted(components, key=lambda x: x.number_of_edges() )
            component_num = len(components)
        
           
            min_pair = None
            min_reward = 0
            min_reward_diff = 0
            edges_2_remove = None
            
            
            ranges = [len(list(components[i].edges())) for i in range(component_num)]
            print("Ranges: ", ranges)
            loops = [range(r) for r in ranges]
            for comb in itertools.product(*loops):
                print("Combination: ", comb)
                # for each edge index do the rest.
                tup = []
                for j in range(len(comb)):
                    e = list(components[j].edges())[comb[j]]
                    e = (e[0]+min_node, e[1]+min_node)
                    e = (min(e), max(e))
                    tup.append(e)
                    
                print("original tup: ", tup)
                original_tup_rew = 0
                for e in tup:
                    original_tup_rew += objective_table[e]
                print("original tuple rew: ", original_tup_rew)
                combinations, comb_reward_dict = generate_combinations_4(tup, objective_table)
                
                print("Combinations: \n",combinations)
                print(comb_reward_dict)
        
                for i in range(len(combinations)):
                    
                    if min_pair == None:
                        if i == 0:
                            min_pair = combinations[i]
                            min_reward = comb_reward_dict[min_pair]
                            min_reward_diff = abs(original_tup_rew - min_reward)
                            edges_2_remove = tup
                    else:
                        # if the (combination_reward - original comb reward.)
                        if abs(comb_reward_dict[combinations[i]] - original_tup_rew) < min_reward_diff:
                        #if comb_reward_dict[combinations[i]] < min_reward:
                            min_pair = combinations[i]
                            min_reward = comb_reward_dict[combinations[i]]
                            # og_edge_2 = e2
                            # og_edge_3 = e3
                            edges_2_remove = tup
        
        
                print("Edges to remove: ", edges_2_remove)
                print("Min pair: ", min_pair)
                print("##############################")
        
            edges_2_remove_ordered = [(min(e),max(e)) for e in edges_2_remove]
            for e in edges_2_remove_ordered:
                all_edges.remove(e)
            for e in min_pair:
                all_edges.append(e)
            all_edges = sorted(all_edges, key=lambda x: x[0] )
            new_state_g = nx.empty_graph()
            new_state_g.add_edges_from(all_edges)
        
            total_rew_new = 0
            for e in all_edges:
                total_rew_new += objective_table[e]
        
            total_rew_old = 0
            for e in old_edge:
                total_rew_old += objective_table[e]
        
            print("Old edges: ", total_rew_old, old_edge)
            print("New edges: ", total_rew_new, all_edges)
            print("New graph connected: ", nx.is_connected(new_state_g))
            
            #print("Start node list element: ", start_nodes[0])
            if new_state_g.degree[start_nodes[0]] > 1 or new_state_g.degree[end_nodes[0]] > 1:
                print("PATH CONNECTOR NODES NOT ASSIGNED APPROPRIATLY!")
                print("Degrees: ", new_state_g.degree[start_nodes[0]], new_state_g.degree[end_nodes[0]])
                sys.exit()
            print("################################################")
            if nx.is_connected(new_state_g) == False:
                sys.exit()
        else:
            new_state_g = g
    
    elif path_num == 2:
        
        
        
        all_edges = list(g.edges())
        all_nodes = list(g.nodes())
        all_edges = [(min(e)+min_node,max(e)+min_node) for e in all_edges]
        old_edge = all_edges.copy()
    
        conn_components = list(nx.connected_components(g))
        components = [g.subgraph(component).copy() for component in conn_components]
        components = sorted(components, key=lambda x: x.number_of_edges() )
        component_num = len(components)
        
        
        min_pair = None
        min_reward = 0
        
        min_reward_diff = 0
        
        edges_2_remove = None
        ranges = [len(list(components[i].edges())) for i in range(component_num)]
        print("Ranges: ", ranges)
        loops = [range(r) for r in ranges]
        for comb in itertools.product(*loops):
            
            # for each edge index do the rest.
            tup = []
            for j in range(len(comb)):
                e = list(components[j].edges())[comb[j]]
                e = (e[0]+min_node, e[1]+min_node)
                e = (min(e), max(e))
                tup.append(e)
                
            print("original tup: ", tup)
            original_tup_rew = 0
            for e in tup:
                original_tup_rew += objective_table[e]
            print("original tuple rew: ", original_tup_rew)
            combinations, comb_reward_dict = generate_combinations_4(tup, objective_table)
                
            print(comb_reward_dict)
    
            for i in range(len(combinations)):
                
                if filter_edge_swap_combinations(g, combinations[i], tup, forbidden_connections, min_node):
                
                    if min_pair == None:
                        if i == 0:
                            min_pair = combinations[i]
                            min_reward = comb_reward_dict[min_pair]
                            min_reward_diff = abs(original_tup_rew - min_reward)
                            edges_2_remove = tup
                    else:
                        if abs(comb_reward_dict[combinations[i]] - original_tup_rew) < min_reward_diff:
                        #if comb_reward_dict[combinations[i]] < min_reward:
                            min_pair = combinations[i]
                            min_reward = comb_reward_dict[combinations[i]]
                            # og_edge_2 = e2
                            # og_edge_3 = e3
                            edges_2_remove = tup
    
    
            print("Edges to remove: ", edges_2_remove)
            print("Min pair: ", min_pair)
            print("##############################")
        if edges_2_remove != None:
            edges_2_remove_ordered = [(min(e),max(e)) for e in edges_2_remove]
            for e in edges_2_remove_ordered:
                all_edges.remove(e)
            for e in min_pair:
                all_edges.append(e)
            all_edges = sorted(all_edges, key=lambda x: x[0] )
        
        new_state_g = nx.empty_graph()
        new_state_g.add_edges_from(all_edges)
    
        total_rew_new = 0
        for e in all_edges:
            total_rew_new += objective_table[e]
    
        total_rew_old = 0
        for e in old_edge:
            total_rew_old += objective_table[e]
    
        print("Old edges: ", total_rew_old, old_edge)
        print("New edges: ", total_rew_new, all_edges)
        print("New graph connected: ", nx.is_connected(new_state_g))
    
    
        # else:
        #     new_state_g = g
        
    return new_state_g