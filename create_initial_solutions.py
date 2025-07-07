import numpy as np
from numpy.random import default_rng
from collections import Counter
import pandas as pd
import pickle
import random
import os
import networkx as nx
from networkx.algorithms.community import louvain_communities
import ast
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy
import copy
import math
import itertools
from itertools import permutations, combinations
from more_itertools import distinct_permutations as idp


from helper_functions import create_fully_connected_graph,\
                                extract_weights_and_edges,\
                                extract_distance_matrix,\
                                init_solution,\
                                create_design_mat,\
                                is_connected_from_incidence_vector,\
                                compute_reward_map,\
                                extract_lattice_basis_sparse,\
                                convert_sym_to_np,\
                                create_fully_connected_graph,\
                                create_tsp_polytope_graph

from reward_functions import reward_cost
from exact_solution import TSP_integer_program





def create_single_TSP_initial_solution(node_num, weight_min, weight_max):

    objective_table = [] # cost vector for each possible edge.
    initial_states = {} # dictionary holding the initial states.
    reward_lists = [] # list holding the rewrds for each edge in the subproblem. We read it off to compute the travel cost.
    patches = 1
    
    available_actions, initial_states, distance_matrix, objective_table, reward_list = create_tsp_polytope_graph(node_num, 
                                                                                                             patches, 
                                                                                                             initial_states, 
                                                                                                             reward_lists,
                                                                                                             reward_cost,
                                                                                                             weight_min,
                                                                                                             weight_max)
    
    return available_actions, initial_states, distance_matrix, objective_table, reward_list



def create_random_community_graph(N, 
                                  community_size_min=5, 
                                  community_size_max=10, 
                                  min_edges=1, 
                                  max_edges=3, 
                                  max_connections_between_communities=3, 
                                  inter_community_weight_factor=10.0,
                                  seed=None):
    if seed is not None:
        random.seed(seed)

    # Step 1: Create N fully connected subgraphs with weighted edges
    subgraphs = []
    for _ in range(N):
        size = random.randint(community_size_min, community_size_max)
        subgraph = nx.complete_graph(size)
        for u, v in subgraph.edges():
            weight = random.uniform(1.0, 10.0)
            subgraph[u][v]['weight'] = weight
        subgraphs.append(subgraph)
        

    # Step 2: Combine all subgraphs into a single graph
    G = nx.Graph()
    mapping = {}
    node_offset = 0
    community_sizes = []

    for i, subgraph in enumerate(subgraphs):
        mapping.update({node: node + node_offset for node in subgraph.nodes()})
        G = nx.disjoint_union(G, subgraph)
        community_sizes.append(len(subgraph))
        node_offset += len(subgraph)

    # Create a master multi-graph with one node for each community
    master_graph = nx.MultiGraph()
    #master_graph.add_nodes_from(range(N))

    # Track nodes that have already been connected to avoid multi-community connections
    used_nodes = {i: set() for i in range(N)}
    connector_nodes_dict = {i: [] for i in range(N)}  # Dictionary to store connector nodes
    master_graph_edges = []
    # List to store edges connecting different communities
    inter_community_edges = []

    # Step 3: Connect the subgraphs with a random number of edges, ensuring no two edges attach to the same node and no node connects to more than one other community
    connection_edges = []
    subgraph_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    for (i, j) in subgraph_pairs:
        # Determine the number of edges to add between these two communities, capped by max_connections_between_communities
        num_edges = min(random.randint(min_edges, max_edges), max_connections_between_communities)
        nodes_i = [node for node in subgraphs[i].nodes() if node not in used_nodes[i]]
        nodes_j = [node for node in subgraphs[j].nodes() if node not in used_nodes[j]]

        # Shuffle nodes to randomize the selection
        random.shuffle(nodes_i)
        random.shuffle(nodes_j)

        added_edges = 0
        
        for _ in range(num_edges):
            if not nodes_i or not nodes_j:  # If we run out of available nodes, we stop
                break

            node_i = nodes_i.pop(0) + sum(community_sizes[:i])
            node_j = nodes_j.pop(0) + sum(community_sizes[:j])

            # Assign a random weight to the edge with a factor applied to inter-community edges
            weight = random.uniform(0.1, 10.0) * inter_community_weight_factor
            G.add_edge(node_i, node_j, weight=weight)
            connection_edges.append((node_i, node_j, weight))
            inter_community_edges.append((node_i, node_j, weight))  # Store inter-community edge
            used_nodes[i].add(node_i - sum(community_sizes[:i]))
            used_nodes[j].add(node_j - sum(community_sizes[:j]))

            # Track connector nodes for both communities
            connector_nodes_dict[i].append(node_i)
            connector_nodes_dict[j].append(node_j)

            # Add the same weighted edge to the master graph
            #master_graph.add_edge(node_i, node_j, weight=weight)
            master_graph.add_edge(i, j, weight=weight)
   
            
    return G, connection_edges, connector_nodes_dict, master_graph, community_sizes





def create_power_grid_line_initial_solution(community_num, nodes_per_comm, weight_min, weight_max):

    
    S = []
    path_num = []
    subgraph_nodes = [] # all non zero lists of subgraph nodes that we will use in reconstruction of the margins.
    subgraph_edges = [] # initial solution edges in each subproblem.
    subgraphs = [] # initial solution graphs
    reward_lists = [] # list holding the rewrds for each edge in the subproblem. We read it off to compute the travel cost.
    generated_margins = [] # all sub generated margins
    forbidden_connections_dict = {}
    starting_nodes_dict = {}
    ending_nodes_dict = {}
    initial_states = {} # dictionary holding the initial states. 
    available_actions_dict = {} # available actions for each community.
    
    
    G,connecting_edges,connector_nodes,path_graphs \
                                    = create_power_grid_graph_line(community_num, nodes_per_comm, weight_min, weight_max, True)
        
    nx.draw(G, node_color='black', edge_color='black', with_labels=False)
    plt.savefig("Figures/line_graph.png")
    plt.show()
        
    edge_weights = extract_weights_and_edges(G)
    objective_table = edge_weights
    
    
    # calculate community connection edges cost.
    community_connection_cost = 0
    for p in range(len(connecting_edges)):
        community_connection_cost += edge_weights[connecting_edges[p]]
    print("Community connection edges cost: ", community_connection_cost, connecting_edges)
    print("Connector node pairs: ", connector_nodes)
    community_connections_cost_list = [community_connection_cost]
    
    # compute the distance matrix for the integer program.
    distance_matrix = extract_distance_matrix(edge_weights)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j and distance_matrix[i, j] == 0:
                distance_matrix[i, j] = 10000
                
    # get each node's neighbours and store it in a dict.
    node_neighbours = {n: list(G.neighbors(n)) for n in list(G.nodes())}
    
    # diconnect the underlying graph.
    G.remove_edges_from(connecting_edges)
    
    S = [G.subgraph(c) for c in nx.connected_components(G)]

    for i in range(len(S)):
        forbidden_connections_dict[i] = []

    for i in range(len(S)-1):
        connections = connecting_edges[i*2:(i+1)*2]
        forbidden_connections_dict[i].append([x[0] for x in connections])
        forbidden_connections_dict[i+1].append([x[1] for x in connections])
    print("Forbidden node connections: \n", forbidden_connections_dict)
    
    
    for i in range(len(S)):
        print("############################################################")
        print("######################## Subgraph: " + str(i) + " #######################")

        start_nodes = []
        end_nodes = []
        edges_2_remove = [] # forbidden edges in each patch based on start and end nodes.
        pairs = len(connector_nodes[i])//2

        connector_nodes_sorted = sorted(connector_nodes[i])
        for pair in range(pairs):
            start_node = connector_nodes_sorted[2*pair]
            end_node = connector_nodes_sorted[2*pair+1]

            print(start_node, end_node)
            start_nodes.append(start_node)
            end_nodes.append(end_node)

        start_nodes = sorted(start_nodes)
        end_nodes = sorted(end_nodes)

        starting_nodes_dict[i] = start_nodes
        ending_nodes_dict[i] = end_nodes

        edge_weight_list = [(u, v, data['weight']) for u, v, data in S[i].edges(data=True)]
        path_nodes, path_edges = find_disjoint_hamiltonian_paths(len(start_nodes), nodes_per_comm, start_nodes, end_nodes, edge_weight_list, S[i])

        print("Path nodes: \n", path_nodes)
        print("Path edges: \n", path_edges)

        path_num.append(len(list(path_nodes.keys())))

            
        path_edges_combined = []
        for k in range(len(list(path_edges.keys()))):
            path_edges_combined.extend(path_edges[k])
        sub_g = nx.empty_graph()
        sub_g.add_edges_from(path_edges_combined)

        sort_nodes = sorted(list(sub_g.nodes()))
        subgraph_nodes.append(sort_nodes)
        print("sub nodes: \n", list(sub_g.nodes()))
        print("sub sorted nodes: \n", sort_nodes)
        print("sub edges: \n", list(sub_g.edges()))


        subgraphs.append(sub_g)

        adj = nx.adjacency_matrix(sub_g, nodelist=sort_nodes) # adjecency matrix
        init_sol, upper_diagonal = init_solution(adj)
        dm = create_design_mat(len(sort_nodes))
        initial_states[i] = init_sol
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
        generated_margins.append(margin)
        available_actions = extract_lattice_basis_sparse(dm) # get the lattice basis out of the design matrix.
        available_actions = convert_sym_to_np(available_actions) # convert to numpy.
        available_actions_dict[i] = available_actions
    return available_actions_dict, initial_states, distance_matrix, objective_table, reward_list, starting_nodes_dict, ending_nodes_dict, connecting_edges





def create_power_grid_graph_line(communities, nodes_per_comm, weight_min, weight_max, positive_w=True):
    
    G = nx.empty_graph()    
    patch_connections = {}
    connector_nodes_dict = {}
    path_graphs = {}
    for p in range(communities):
        p_g = create_fully_connected_graph(nodes_per_comm, positive_w, p, weight_min, weight_max)
        G = nx.disjoint_union(G, p_g)
        nodes = list(p_g.nodes)
        if p == 0 or p == communities-1:
            connector_nodes = random.sample(nodes, 2)
        else:
            connector_nodes = random.sample(nodes, 4)
        patch_connections[p] = [n+p*nodes_per_comm for n in connector_nodes]
        path_graphs[p] = p_g
        
    
    connecting_edges = []
    connector_nodes_dict = copy.deepcopy(patch_connections)
    for i in range(communities-1):
        
        curr_conn_1 = patch_connections[i][0]
        curr_conn_2 = patch_connections[i][1]
        
        next_conn_1 = patch_connections[i+1][0]
        next_conn_2 = patch_connections[i+1][1]
        e1 = (curr_conn_1, next_conn_1)
        e2 = (curr_conn_2, next_conn_2)
        
        patch_connections[i+1].remove(next_conn_1)
        patch_connections[i+1].remove(next_conn_2)
                
        connecting_edges.extend([e1,e2])
    
    G.add_edges_from(connecting_edges)
    
    
    if positive_w == True:
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.uniform(weight_min, weight_max)  # Random weight between 1 and 10
    else:
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = -1*random.uniform(weight_min, weight_max)  # Random weight between 1 and 10
    
    
    return G, connecting_edges, connector_nodes_dict, path_graphs



def create_exact_solution(distance_matrix, objective_table):
    tour_edges = TSP_integer_program(distance_matrix) # solve TSP-MZT integer program.
    print("Tour edges: \n", tour_edges)

    tour_edges = [(min(e),max(e)) for e in tour_edges]
    or_reward = 0
    for e in tour_edges:
        or_reward += objective_table[e]
    print("Reward: ", or_reward)
    
    return tour_edges



def find_disjoint_hamiltonian_paths(path_num, node_num, start_nodes, end_nodes, edge_weight_list, path_graph):
    
    if path_num != len(start_nodes) and path_num != len(end_nodes):
        print("Number of paths and number of connector nodes is not the same!")
        return None

    paths_nodes = {}
    paths_edges = {}
    average_path_length = (node_num-1)//path_num
    neighbours = {n: list(path_graph.neighbors(n)) for n in list(path_graph.nodes())}
    used_nodes = set()
    
    used_nodes.update(start_nodes)
    used_nodes.update(end_nodes)
    
    for p in range(path_num):
        
        path_step = 0
        start_node = start_nodes[p]
        end_node = end_nodes[p]
    
        path_nodes = [start_node]
        path_edges = []
        
        sorted_neighbours = sorted(neighbours[start_node])
        restricted_edges = [(start_node,n) for n in sorted_neighbours if n != end_node and n not in used_nodes]
        restricted_edges = [(min(tup), max(tup)) for tup in restricted_edges]
        indx = 0
        if len(restricted_edges) == 0:
            paths_nodes[p] = [start_node, end_node]
            paths_edges[p] = [(start_node, end_node)]
            return paths_nodes, paths_edges
        else:
            indx = np.random.randint(0, len(restricted_edges))
        edge = restricted_edges[indx]
        n1 = edge[0]
        n2 = edge[1]
        if n1 == start_node:
            current_node = n2
        elif n2 == start_node:
            current_node = n1
        prev_node = start_node
        
        path_nodes.append(current_node)
        path_edges.append((prev_node, current_node))
        
        while current_node != end_node:
    
            used_nodes.add(current_node)
                
            # Randomly pick next node in the cylce s.t it is not the previous node.
            current_neighbours = neighbours[current_node]
            restricted_neighbours = [n for n in current_neighbours if n not in path_nodes and n != end_node and n not in used_nodes]

            if path_step >= average_path_length or len(restricted_neighbours) == 0:
                path_nodes.append(end_node)
                path_edges.append((current_node, end_node))
                paths_nodes[p] = path_nodes
                paths_edges[p] = path_edges
                break
                #return path_nodes, path_edges
        
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
        
            path_step += 1
            path_nodes.append(current_node)
            path_edges.append((prev_node, current_node))

    return paths_nodes, paths_edges









