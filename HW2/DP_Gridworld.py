
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[66]:


class Agent:
    
    def __init__(self):
        self.gridsize = 4 # Grid size
        self.policy = np.ones((self.gridsize,self.gridsize,4))*0.25 # Policy
        self.discount = 1 # Discount factor for return calculation
        self.actions = [[0,1],[1,0],[0,-1],[-1,0]] # Right,Down,Left,Up
        self.actions_meanings = ["→","↓","←","↑"]
        self.v = np.zeros((self.gridsize,self.gridsize)) # estimated returns of each state (positions on the grid)
        self.actions_map = {}
        for i in range(len(self.actions)):
            self.actions_map[str(self.actions[i])] = self.actions_meanings[i]
            
        self.optimal_policy = [[ "↓ ↑ ← →" for i in range(self.gridsize)] for j in range(self.gridsize)]
        
    def display_policy(self):
        for i in range(self.gridsize):
            for j in range(self.gridsize): 
                action_seq = ""
                for action_ind in range(len(self.actions)): 
                    if self.policy[i,j,action_ind] > 0:
                        action_seq = action_seq+self.actions_map[str(self.actions[action_ind])]+" "
                        
                self.optimal_policy[i][j] = " "*int((8-len(action_seq))/2) + action_seq + " "*((8-len(action_seq) - int((8-len(action_seq))/2)))
        
        self.optimal_policy[0][0] = " "*7
        self.optimal_policy[self.gridsize-1][self.gridsize-1] = " "*7
        
        print('\n',np.array(self.optimal_policy))       
        
    def value_iteration(self,env):
            
        # Threshold to check for convergence
        threshold = 10**-4  
        # Iterating until convergence
        while True:  
            # Absolute difference between consecutive v(s) estimates
            cur_diff = 0
            for i in range(self.gridsize):
                for j in range(self.gridsize): 
                    # Next value for current state
                    next_v = -1000 
                    # Iterating over all actions in a state
                    for act_ind in range(len(self.actions)):
                        # performing given action and obtaining next state and reward
                        reward,next_x,next_y = env.perform_action([i,j],self.actions[act_ind])
                        # updating value function of given state using bellman's equation
                        next_v = max(next_v,reward+self.discount*self.v[next_x,next_y]) 
                    cur_diff = max(cur_diff,abs(next_v-self.v[i,j]))
                    self.v[i,j] = next_v
                    
            print("\n")
            print(np.round(self.v, 1))
                    
            # If difference sufficiently small --> convergence
            if cur_diff < threshold: 
                # rounding to 1 decimal place
                # self.v = np.round(self.v, 1) 
                break


        ###### Optimal Policy Selection ######
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                # Stores the maximum action value function of all actions in given state
                q = float('-inf')
                # Stores Optimal action(s) for given state
                a = []
                # Iterating over all actions in a state
                for act_ind in range(len(self.actions)):
                    # performing given action and obtaining next state and reward
                    reward,next_x,next_y = env.perform_action([i,j],self.actions[act_ind])

                    # checking whether action value of given state is greater/equal to current max
                    if q < reward+self.discount*self.v[next_x,next_y]:
                        q = reward+self.discount*self.v[next_x,next_y]
                        a = [act_ind]
                    elif q == reward+self.discount*self.v[next_x,next_y]:
                        a.append(act_ind)

                # Savin old state to check stability later
                old_state_policy = self.policy[i,j,:] > 0

                # Computing new policy (stocastic)
                for action_ind in range(4):
                    if action_ind not in a: 
                        self.policy[i,j,action_ind] = 0
                    else:
                        self.policy[i,j,action_ind] = 1./len(a)

        print("\n Optimal Policy and Value Function")
        print('\n',np.round(self.v, 1) )
        self.display_policy()
   
            

    def policy_iteration(self,env):
        
        # Run till convergence achieved
        while True:
            
            ###### Policy Evaluation ######
            # Threshold to check for convergence
            threshold = 10**-4  
            cur_diff = 0
            # Iterating until convergence
            while True:  
                # Absolute difference between consecutive v(s) estimates
                cur_diff = 0
                for i in range(self.gridsize):
                    for j in range(self.gridsize): 
                        # Next value for current state
                        next_v = 0
                        # Iterating over all actions in a state
                        for act_ind in range(len(self.actions)):
                            # performing given action and obtaining next state and reward
                            reward,next_x,next_y = env.perform_action([i,j],self.actions[act_ind])
                            # updating value function of given state using bellman's equation
                            next_v += self.policy[i,j,act_ind]*(reward+self.discount*self.v[next_x,next_y]) 
#                             if i==0 and j==1:
#                                 print("action = ",self.actions_map[str(self.actions[act_ind])],"  Next v = ",next_v,"  Reward = ",reward,"  Val func = ", self.v[next_x,next_y],next_x,next_y)
                        cur_diff = max(cur_diff,abs(next_v-self.v[i,j]))
                        self.v[i,j] = next_v
#                 print("\n")
                        
#                 print('\n',np.round(self.v, 1) )
                    
                # If difference sufficiently small --> convergence
                if cur_diff < threshold: 
                    # rounding to 1 decimal place
                    # self.v = np.round(self.v, 1) 
                    break

            ###### Policy Improvement ######
            policy_stable = True
            for i in range(self.gridsize):
                for j in range(self.gridsize):
                    # Stores the maximum action value function of all actions in given state
                    q = float('-inf')
                    # Stores Optimal action(s) for given state
                    a = []
                    # Iterating over all actions in a state
                    for act_ind in range(len(self.actions)):
                        # performing given action and obtaining next state and reward
                        reward,next_x,next_y = env.perform_action([i,j],self.actions[act_ind])

                        # checking whether action value of given state is greater/equal to current max
                        if q < reward+self.discount*self.v[next_x,next_y]:
                            q = reward+self.discount*self.v[next_x,next_y]
                            a = [act_ind]
                        elif q == reward+self.discount*self.v[next_x,next_y]:
                            a.append(act_ind)

                    # Savin old state to check stability later
                    old_state_policy = self.policy[i,j,:] > 0

                    # Computing new policy (stocastic)
                    for action_ind in range(4):
                        if action_ind not in a: 
                            self.policy[i,j,action_ind] = 0
                        else:
                            self.policy[i,j,action_ind] = 1./len(a)
                            
                    new_state_policy = self.policy[i,j,:] > 0
                    
                    # Checking if policy remained stable in given state
                    if np.any(np.logical_xor(old_state_policy,new_state_policy) == True) :
                        policy_stable = False
                        
            print('\n',np.round(self.v, 1) )
            self.display_policy()
                        
            
            # If policy stable ---> optimal policy achieved 
            if policy_stable: 
                break
            
        print("\n\n Dispalying Optimal Policy and Value Function")
        print('\n',np.round(self.v, 2) )
        self.display_policy()
        
    


# In[67]:


class Environment:
    
    def __init__(self):
        self.gridsize = 4 # Grid size
        self.terminal_states = [[0,0],[self.gridsize - 1,self.gridsize - 1]] # Terminal states
        self.terminal_rewards = 0
        self.actions = [[0,1],[1,0],[0,-1],[-1,0]]  # Right,Down,Left,Up
        self.non_term_reward = -1
        
    def perform_action(self, state, action):
        # Checking for special states
        if state in self.terminal_states: 
            return tuple([self.terminal_rewards]+state)
        
        # Calculating next state
        next_state = [state[0]+action[0],state[1]+action[1]]
        
        # Checking for out of grid state
        if next_state[0]<0 or next_state[1]<0 or next_state[0]>=self.gridsize or next_state[1]>=self.gridsize:
            return tuple([self.non_term_reward]+state)
        
        # Returning next state and reward
        return tuple([self.non_term_reward]+next_state)


# In[68]:


# Setup to generate fig 4.1 using policy iteration
agent = Agent()
env = Environment()
agent.policy_iteration(env)   


# In[69]:


# Setup to generate fig 4.1 using value iteration
agent = Agent()
env = Environment()
agent.value_iteration(env)   

