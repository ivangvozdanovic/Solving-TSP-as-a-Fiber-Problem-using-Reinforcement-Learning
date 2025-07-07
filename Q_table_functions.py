import numpy as np

'''
Get a Q value given state and action. 
If the state has not been seen before, the Q value is set to zero.
If the action for the given state has not been used before, return Q value = 0.
'''
def getQvalue(Q,state,action):
    state_keys = list(Q.keys())
    state_str = state.tobytes()
    action_str = action.tobytes()
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
def getBestAction(Q,state,available_actions, lb, ub):
    state_keys = list(Q.keys())
    state_str = state.tobytes()
    if state_str not in state_keys:
        if len(available_actions) != 0:
            action_coeffs = np.random.randint(lb, ub, len(available_actions)) 
            return action_coeffs
    else:
        best_action_str = max(Q[state_str], key=lambda k: Q[state_str][k])
        best_action = np.frombuffer(best_action_str,dtype=int)
        return best_action

'''Update the Q table at the point given by (state,action) by value update_value.'''
def updateQdict(Q,state,action,update_value):
    state_keys = list(Q.keys())
    state_str = state.tobytes()
    action_str = action.tobytes()
    if state_str not in state_keys:
        Q[state_str] = {action_str : update_value}
    else:
        action_keys = list(Q[state_str].keys())
        if action_str not in action_keys:
            Q[state_str][action_str] = update_value
        else:
            Q[state_str][action_str] += update_value

'''Upadate the Q values along the path (state,action,state,action,....)'''
def updateQpath(Q,update_value,path,memory_length):
    print("Updating the Q path")
    for i in range(len(path)-memory_length, len(path)):
        state = path[i][0]
        action = path[i][1]
        updateQdict(Q,state,action,update_value)

'''Check whether the given action already exists in available_actions. '''     
# def checkAction(action,available_actions):
#     for a in available_actions:
#         diff = np.subtract(action,a)
#         if any(diff):
#             continue
#         else:
#             return False
#     return True
def checkAction(action,available_actions):
    if action in available_actions:
        return True
    else:
        return False

#Combine existing actions to create a new one and store it in available_actions.
def createNewAction(path,available_actions,table_size):
    new_action = [0 for i in range(table_size**2)]
    for i in range(len(path)):
        for j in range(table_size**2):
            new_action[j] += path[i][1][j]
    if checkAction(new_action,available_actions):
        print("New action: ",new_action)
        #print("New action neg: ", -1*np.array(new_action))
        available_actions.append(np.array(new_action))
        available_actions.append(-1*np.array(new_action))
    else:
        print("The action already exists!")


'''Convert the Q dicitonary with bytes into a dictionary with vectors'''
def Qbytes2vectors(Q):
    print("hello")