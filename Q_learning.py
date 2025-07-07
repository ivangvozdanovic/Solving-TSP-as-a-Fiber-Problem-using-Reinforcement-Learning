import math as m
import matplotlib.pyplot as plt

from helper_functions import create_state_graph, share_element, create_state_edges
from policy_functions import time_independent_policy, filter_actions, check_feasability, edge_swap_policy, multi_edge_swap_policy
from Q_table_functions import getQvalue, getBestAction, updateQdict, updateQpath, checkAction, createNewAction
from EdgeSwap_Q_table_functions import getBestActionES, getQvalueES, updateQdictES
from Multi_EdgeSwap_Q_table_functions import getBestActionMultiES, getQvalueMultiES, updateQdictMultiES

def Q_learning(epsilon, 
               episode_num, 
               path_num, 
               available_actions, 
               table_size, 
               max_path_length, 
               discount_factor, 
               env, 
               lr, 
               save_plots, 
               lb, 
               ub):
    
    #Print some diagnostic.
    print("\n",
          "Episode Number: ", episode_num, "\n",
          "Path Number: ", path_num, "\n",
          "Learning Rates: ", lr, "\n",
          "Gamma: ",discount_factor, "\n",
          "Table length: ", table_size, "\n",
          "Maximum Path Length: ", max_path_length)

    Q = {}  # Initialize the Q dictionary.
   
    ave_episode_reward = [] # Collect average reward of each episode (averaged on (P) paths per episode).
    eps = [] # List for storing epsilons. Used for plotting the decay of exploration parameter.
    alpha = [] # List for storing learning parameters. Used for plotting the decay of TD learning parameter. 
    
    d = 1/episode_num # Decrement for the exploration parameter.
  
    for ep in range(episode_num):
        
        print("##############################  Episode: ", ep+1, "  ##############################")
        epsilon, epsilon_f = exploration_control(epsilon, ep, episode_num, d, 2) # Update epsilon for each episode.
        lr, lr_f = exploration_control(lr, ep, episode_num, d, 4) # Update the learning rate of the TD update. 
        eps.append(epsilon_f) # Store the epsilon for plotting.
        alpha.append(lr_f)
        episodic_reward = [] # Each element in the list is a reward of a single path in 1 episode.
        
        for p in range(path_num):
            
            # For each path p:
            state = env.reset() # Reset the environment.
            path_reward = 0 # Set path reward to 0.
          

            for t in range(max_path_length):

                #Pick an action from epsilon_greedy policy.
                action = time_independent_policy(Q, epsilon_f, state, available_actions, lb, ub)
                
                #Take a step in the environment.
                next_state, reward, done, found_solution, info = env.step(action)

#                 print("State: \n", state)
#                 print("Action: \n", action)
#                 print("Reward: \n", reward)
#                 print("Done: ", done)
#                 print("Next State: \n ", next_state)
#                 print("#################################")
                
                #Collect total discounted reward over 1 path.
                path_reward += (discount_factor**t)*reward 
                
                
                #Preform TD update:
                
                td_target = 0 # Temporal difference target.
                best_next_action = getBestAction(Q,next_state,available_actions,lb,ub) # Get the argmax action of Q.
                td_target = reward + discount_factor * getQvalue(Q,next_state,best_next_action) # Calc the Bellman target.
                td_delta = td_target - getQvalue(Q,state,action) # Compute the difference between current Q and Bellman target.
                updateQdict(Q,state,action, lr_f * td_delta) # Update the dictionary with the new value at (state,action) pos.

                #Move to the next state.
                state = next_state
                
            #print("Path reward:", path_reward)
            #Collect path reward after each path.
            episodic_reward.append(path_reward)

        #Take an average of total discounted rewards over all paths.
        ave_episode_reward.append(average(episodic_reward))


    print("The number of unique feasable solutions is : ", len(list(Q.keys())))
    
    x_axis = [i for i in range(episode_num)]
    plt.plot(x_axis, eps,color='red') 
    plt.plot(x_axis,alpha,color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Exploration Parameter/Learning Rate")
    if save_plots:
        plt.savefig("Figures/Q-Learning/exploration_explotation_graph_"+"_EP="+str(episode_num)
                    +"_P="+str(path_num)+"_PSI="+str(round(psi,2))+"_Beta="+str(discount_factor)+".eps",
                       format='eps',dpi=300,bbox_inches='tight')
    plt.show()
    plt.plot(ave_episode_reward,color='red')
    plt.xlabel("Episodes")
    plt.ylabel('Average Culmulative \n Episodic Reward')
    #plt.yscale('log')
    if save_plots:
        plt.savefig("Figures/Q-Learning/convergence"+"_EP="+str(episode_num)
                    +"_P="+str(path_num)+"_PSI="+str(round(psi,2))+"_Beta="+str(discount_factor)+".eps",
                       format='eps',dpi=300,bbox_inches='tight')
    plt.show()
    return Q




def EdgeSwap_Q_learning(epsilon, 
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
               n_step_lookup):
    
    #Print some diagnostic.
    print("\n",
          "Episode Number: ", episode_num, "\n",
          "Path Number: ", path_num, "\n",
          "Learning Rates: ", lr, "\n",
          "Gamma: ",discount_factor, "\n",
          "Table length: ", table_size, "\n",
          "Maximum Path Length: ", max_path_length)

    Q = {}  # Initialize the Q dictionary.
   
    ave_episode_reward = [] # Collect average reward of each episode (averaged on (P) paths per episode).
    eps = [] # List for storing epsilons. Used for plotting the decay of exploration parameter.
    alpha = [] # List for storing learning parameters. Used for plotting the decay of TD learning parameter.  
    cut_off = episode_num
    d = 1/(episode_num-episode_num//cut_off) # Decrement for the exploration parameter.
  
    n_states = []
    n_actions = []
    n_rewards = []
    
    agent_paths = {}

    for ep in range(episode_num):
        
        print("##############################  Episode: ", ep+1, "  ##############################")
        epsilon, epsilon_f = exploration_control(epsilon, cut_off, ep, episode_num, d, 2) # Update epsilon for each episode.
#         lr, lr_f = exploration_control(lr, episode_num, ep, episode_num, d, 2) # Update the learning rate of the TD update. 
        eps.append(epsilon_f) # Store the epsilon for plotting.
        alpha.append(lr)
        episodic_reward = [] # Each element in the list is a reward of a single path in 1 episode.
        
        state = env.reset() # Reset the environment.
        path_reward = 0 # Set path reward to 0.
        done = False
        step = 0
        
        
        path_trajectory = [str(tuple(state))]
        
       
    
        while not done:

            
            state_edges = create_state_edges(node_num, state)

            #Pick an action from epsilon_greedy policy.
#             action = edge_swap_policy(Q, state, state_edges, epsilon_f)
            action = edge_swap_policy(Q, state, state_edges, epsilon_f, node_num)

            #Take a step in the environment.
            next_state, reward, done, info = env.step(action)

            path_trajectory.append(str(tuple(next_state)))
            
            #Collect total discounted reward over 1 path.
            path_reward += (discount_factor**step)*reward 
            
            n_rewards.append(reward)
            n_states.append(state)
            n_actions.append(action)
           
            #Preform n-step TD update:
            if len(n_states) > n_step_lookup or done:
                G = sum([discount_factor ** i * n_rewards[i] for i in range(min(n_step_lookup, len(n_rewards)))])
                
                if not done and len(n_rewards) >= n_step_lookup:
                    next_state_edges = create_state_edges(node_num, next_state)
                    best_next_action = getBestActionES(Q, next_state, next_state_edges) # Get the argmax action of Q.
                    G += discount_factor ** n_step_lookup * getQvalueES(Q, next_state, best_next_action) # Calc the Bellman target.
                    
                # Perform Bellman update on the state at time t-N
                update_state = n_states.pop(0)
                update_action = n_actions.pop(0)
                td_delta = G - getQvalueES(Q, update_state, update_action) # Compute the difference between current Q and Bellman target.
                updateQdictES(Q, update_state, update_action, lr * td_delta) # Update the Q with the new value at (state,action) pos.
                n_rewards.pop(0)
                
                
            #Move to the next state.
            state = next_state
            step += 1
        
        agent_paths[ep] = path_trajectory
        
        # Update remaining states after episode ends
        while len(n_states) > 1:
            G = sum([discount_factor ** i * n_rewards[i] for i in range(len(n_rewards))])
            update_state = n_states.pop(0)
            update_action = n_actions.pop(0)
            updateQdictES(Q, update_state, update_action, lr * (G -  getQvalueES(Q, update_state, update_action)))
            n_rewards.pop(0) 
            
        ave_episode_reward.append(path_reward)

    print("The number of unique feasable solutions is : ", len(list(Q.keys())))
    
    x_axis = [i for i in range(episode_num)]
    plt.plot(x_axis, eps,color='red') 
    plt.plot(x_axis,alpha,color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Exploration Parameter/Learning Rate")
    if save_plots:
        plt.savefig("Figures/exploration_explotation_graph_"+"_EP="+str(episode_num)
                    +"_P="+str(path_num)+".png")
    plt.show()
    x_axis = [max_path_length*i for i in range(episode_num)]
    plt.plot(x_axis, ave_episode_reward,color='black')
    plt.xlabel("Steps")
    plt.ylabel('Cumulative Reward')
    #plt.yscale('log')
    if save_plots:
        plt.savefig("Figures/convergence"+"_EP="+str(episode_num)
                    +"_P="+str(path_num)+".png")
    plt.show()
    return Q, agent_paths, ave_episode_reward





def Multi_EdgeSwap_Q_learning(epsilon, 
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
                               n_step_lookup,
                               k_swaps):
    
    #Print some diagnostic.
    print("\n",
          "Episode Number: ", episode_num, "\n",
          "Path Number: ", path_num, "\n",
          "Learning Rates: ", lr, "\n",
          "Gamma: ",discount_factor, "\n",
          "Table length: ", table_size, "\n",
          "Maximum Path Length: ", max_path_length)

    Q = {}  # Initialize the Q dictionary.
   
    ave_episode_reward = [] # Collect average reward of each episode (averaged on (P) paths per episode).
    eps = [] # List for storing epsilons. Used for plotting the decay of exploration parameter.
    alpha = [] # List for storing learning parameters. Used for plotting the decay of TD learning parameter.  
    cut_off = episode_num
    d = 1/(episode_num-episode_num//cut_off) # Decrement for the exploration parameter.
  
    n_states = []
    n_actions = []
    n_rewards = []

    for ep in range(episode_num):
        
        print("##############################  Episode: ", ep+1, "  ##############################")
        epsilon, epsilon_f = exploration_control(epsilon, cut_off, ep, episode_num, d, 2) # Update epsilon for each episode.
#         lr, lr_f = exploration_control(lr, episode_num, ep, episode_num, d, 2) # Update the learning rate of the TD update. 
        eps.append(epsilon_f) # Store the epsilon for plotting.
        alpha.append(lr)
        episodic_reward = [] # Each element in the list is a reward of a single path in 1 episode.
        
        state = env.reset() # Reset the environment.
        path_reward = 0 # Set path reward to 0.
        done = False
        step = 0
        
       
    
        while not done:

            
            state_edges = create_state_edges(node_num, state)

            #Pick an action from epsilon_greedy policy.
            action = multi_edge_swap_policy(Q, state, state_edges, epsilon_f, k_swaps)

            #Take a step in the environment.
            next_state, reward, done, info = env.step(action)

            #Collect total discounted reward over 1 path.
            path_reward += (discount_factor**step)*reward 
            
            n_rewards.append(reward)
            n_states.append(state)
            n_actions.append(action)
           
            #Preform n-step TD update:
            if len(n_states) > n_step_lookup or done:
                G = sum([discount_factor ** i * n_rewards[i] for i in range(min(n_step_lookup, len(n_rewards)))])
                
                if not done and len(n_rewards) >= n_step_lookup:
                    next_state_edges = create_state_edges(node_num, next_state)
                    best_next_action = getBestActionMultiES(Q, next_state, next_state_edges, k_swaps) # Get the argmax action of Q.
                    G += discount_factor ** n_step_lookup * getQvalueMultiES(Q, next_state, best_next_action) # Calc the Bellman target.
                    
                # Perform Bellman update on the state at time t-N
                update_state = n_states.pop(0)
                update_action = n_actions.pop(0)
                td_delta = G - getQvalueMultiES(Q, update_state, update_action) # Compute the difference between current Q and Bellman target.
                updateQdictMultiES(Q, update_state, update_action, lr * td_delta) # Update the Q with the new value at (state,action) pos.
                n_rewards.pop(0)
                
                
            #Move to the next state.
            state = next_state
            step += 1
          
        # Update remaining states after episode ends
        while len(n_states) > 1:
            G = sum([discount_factor ** i * n_rewards[i] for i in range(len(n_rewards))])
            update_state = n_states.pop(0)
            update_action = n_actions.pop(0)
            updateMultiQdictES(Q, update_state, update_action, lr * (G -  getMultiQvalueES(Q, update_state, update_action)))
            n_rewards.pop(0) 
            
        ave_episode_reward.append(path_reward)

    print("The number of unique feasable solutions is : ", len(list(Q.keys())))
    
    x_axis = [i for i in range(episode_num)]
    plt.plot(x_axis, eps,color='red') 
    plt.plot(x_axis,alpha,color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Exploration Parameter/Learning Rate")
    if save_plots:
        plt.savefig("Figures/Q-Learning/exploration_explotation_graph_"+"_EP="+str(episode_num)
                    +"_P="+str(path_num)+"_PSI="+str(round(psi,2))+"_Beta="+str(discount_factor)+".eps",
                       format='eps',dpi=300,bbox_inches='tight')
    plt.show()
    plt.plot(ave_episode_reward,color='red')
    plt.xlabel("Episodes")
    plt.ylabel('Average Culmulative \n Episodic Reward')
    #plt.yscale('log')
    if save_plots:
        plt.savefig("Figures/Q-Learning/convergence"+"_EP="+str(episode_num)
                    +"_P="+str(path_num)+"_PSI="+str(round(psi,2))+"_Beta="+str(discount_factor)+".eps",
                       format='eps',dpi=300,bbox_inches='tight')
    plt.show()
    return Q






'''Exponential discounting of exploration parameter. This can be switched to be linear decay.'''
def exploration_control(eps,cut_off,current_ep,total_episodes,decrement,slope):
    if current_ep < total_episodes//cut_off:
        eps = 1
    else:
        eps -= decrement
    return eps, m.exp(-slope*(1-eps))*eps

'''Average the values in the given list l.'''
def average(l):
    total = len(l)
    sum = 0
    for i in range(total):
        sum+= l[i]
    return sum/total


'''Calculate the L2 error of the true solution and the approximation. Both X and X_ should be lists.'''
def calculate_norm(X,X_,l):
    if type(l) != int:
        raise Exception("Norm power has to be an integer.")
    if len(X) != len(X_):
        raise Exception("Samples in L-"+str(l)+
                        " norm function DO NOT have the same size. Size: ("+str(len(X))+","+str(len(X))+")")
    else:
        sum = 0
        for i in range(len(X)):
            sum += abs((X[i]-X_[i]))**l
        return sum**(1/l)


'''
Graph the error of between the approximate and exact solutions. 
You can specify for which time steps to plot the error (i.e. times = [1,2,5,6])
'''
def graph_error(times,samples_axis,error_axis,save_plots):
    
    leg = []
    for t in times:
        leg.append('T='+str(t))
        plt.plot(samples_axis,error_axis[t-1],'-o',label='T='+str(t)) 
    plt.xlabel("Episodes")
    plt.ylabel("L2 Norm Convergence")
    plt.legend(leg)
    plt.grid()
    if save_plots:
        plt.savefig('Figures/Q-Learning/error_analysis_RL_T='+str(episode_length)+'_PSI='+str(psi)
                                    +'_Beta='+str(discount_factor)+'.eps',format='eps', dpi=300,bbox_inches='tight')
    plt.show()
