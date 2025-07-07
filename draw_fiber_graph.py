import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D 
from matplotlib.animation import PillowWriter
from IPython.display import HTML 

def draw_graph_animation(starting_states, agent_paths, ave_episode_reward_subset, path_length):
    


    fiber_graph = create_fiber_graph(starting_states)
    paths = list(agent_paths.values())[::100]
    ave_episode_reward_subset = ave_episode_reward[::100]

    pos = nx.spring_layout(fiber_graph)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    state = 0
    tick = -1
    edges = []
    valid_nodes = []

    def update(frame):
        global valid_nodes
        global edges
        global state
        global tick

        legend_elements = [
                        Line2D([0], [0], marker='x', color='black', label='Start Node',
                               markersize=10, linestyle='None', linewidth=2),
                        Line2D([0], [0], marker='*', color='black', label='End Node',
                               markersize=12, linestyle='None'),
                        Line2D([0], [0], marker='o', color='black', label='Current Node',
                               markerfacecolor='none', markersize=10, linestyle='None', linewidth=2),
                    ]

        if (frame)%path_length==0:
            tick += 1
            state = 0
            ax1.clear()
            ax2.clear()
            nx.draw_networkx_nodes(fiber_graph, pos, node_color='lightgray', node_size=100, ax=ax1)
            valid_nodes = paths[tick-1]
            edges = list(zip(valid_nodes[:-1], valid_nodes[1:]))

        nx.draw_networkx_edges(fiber_graph, pos, edgelist=edges[:state+1], edge_color='black', width=2, ax=ax1)
  
        # Current node (empty with black border)
        if state < len(valid_nodes):
            current_node = valid_nodes[state]
            x, y = pos[current_node]
            ax1.scatter(x, y, s=250, facecolors='none', edgecolors='black', linewidths=2, zorder=5)

        # Start node as cross (X)
        start_node = str(tuple(initial_states[0]))
        if start_node in pos:
            x, y = pos[start_node]
            ax1.scatter(x, y, s=300, c='black', marker='x', linewidths=2, zorder=6)

        # End node(s) as star (*)
        x, y = pos[opt_tour]
        ax1.scatter(x, y, s=300, c='black', marker='*', zorder=6)
        ax1.legend(handles=legend_elements, loc='upper right', frameon=False)
        ax1.set_title(f"Episode: {(tick-1)*100}")



        # Plot rewards up to current tick in ax2
        x_axis = [i*100 for i in range(len(ave_episode_reward_subset[:tick]))]
        ax2.plot(x_axis, ave_episode_reward_subset[:tick], color='green')
        ax2.set_title("Episode Rewards")
        ax2.set_xlabel("Sampled Episode")
        ax2.set_ylabel("Reward")
        ax2.grid(True)

        state += 1

    last_animation = animation.FuncAnimation(fig, update, frames=len(paths)*len(paths[0]), interval=300, blit=False)
    last_animation.save("tsp_learning.gif", writer=PillowWriter(fps=5), dpi=100)
    
    
def create_fiber_graph(visited_states):
    
    edges = []
    for s1 in visited_states:
        for s2 in visited_states:
            diff = np.subtract(s1,s2)
            if sum(np.abs(diff)) == 4:
                edges.append((str(s1), str(s2)))
    
    
    # Create an undirected graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    return G