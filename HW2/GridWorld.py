
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[24]:


# Class that instantiates an object that creates and solves linear equation for the given gridworld problem
class LinearEqnSolver:
    
    def __init__(self):
        self.a = np.zeros((25,25),dtype="float32") # Stores coefficients of variables
        self.b = np.array([0.5,-10,0.25,-5,0.5,0.25,0,0,0,0.25,0.25,0,0,0,0.25,0.25,0,0,0,0.25,0.5,0.25,0.25,0.25,0.5]) # Stores constants
        self.populate_matrix() # Builds coefficient matrix

    # function updates values in coefficient matrix
    def add(self,current,states,vals):
        for i in range(len(states)):
            row = current - 1
            col = states[i]-1
            self.a[row,col] = vals[i]
        
    # Populates values of coefficient matrix from the linear equations
    def populate_matrix(self):
        self.add(1,[1,2,6],[-0.55,0.225,0.225])
        self.add(2,[2,22],[-1,0.9])
        self.add(3,[2,3,4,8],[0.225,-0.775,0.225,0.225])
        self.add(4,[4,14],[-1,0.9])
        self.add(5,[4,5,10],[0.225,-0.55,0.225])

        self.add(6,[1,6,7,11],[0.225,-0.775,0.225,0.225])
        self.add(7,[2,6,7,8,12],[0.225,0.225,-1,0.225,0.225])
        self.add(8,[3,7,8,9,13],[0.225,0.225,-1,0.225,0.225])
        self.add(9,[4,8,9,10,14],[0.225,0.225,-1,0.225,0.225])
        self.add(10,[5,10,9,15],[0.225,-0.775,0.225,0.225])

        self.add(11,[6,11,12,16],[0.225,-0.775,0.225,0.225])
        self.add(12,[7,11,12,13,17],[0.225,0.225,-1,0.225,0.225])
        self.add(13,[8,12,13,14,18],[0.225,0.225,-1,0.225,0.225])
        self.add(14,[9,13,14,15,19],[0.225,0.225,-1,0.225,0.225])
        self.add(15,[10,15,14,20],[0.225,-0.775,0.225,0.225])

        self.add(16,[11,16,17,21],[0.225,-0.775,0.225,0.225])
        self.add(17,[12,16,17,18,22],[0.225,0.225,-1,0.225,0.225])
        self.add(18,[13,17,18,19,23],[0.225,0.225,-1,0.225,0.225])
        self.add(19,[14,18,19,20,24],[0.225,0.225,-1,0.225,0.225])
        self.add(20,[15,20,19,25],[0.225,-0.775,0.225,0.225])

        self.add(21,[16,21,22],[0.225,-0.55,0.225])
        self.add(22,[17,22,21,23],[0.225,-0.775,0.225,0.225])
        self.add(23,[18,23,22,24],[0.225,-0.775,0.225,0.225])
        self.add(24,[19,24,23,25],[0.225,-0.775,0.225,0.225])
        self.add(25,[20,25,24],[0.225,-0.55,0.225])


    # Solves bellman equations for the gridworld by solving the linear equations
    def solve_bellman_eqns_fig3_2(self):
        x = np.linalg.solve(self.a, self.b)
        x = np.reshape(x,(5,5))
        print(np.round(x, 1) )
        


# In[4]:


class Agent:
    
    def __init__(self):
        self.gridsize = 5 # Grid size
        self.prob_actions = 0.25 # Probability of picking an action in a state
        self.discount = 0.9 # Discount factor for return calculation
        self.actions = [[0,1],[1,0],[0,-1],[-1,0]] # Right,Down,Left,Up
        self.actions_meanings = ["→","↓","←","↑"]
        self.v = np.zeros((self.gridsize,self.gridsize)) # estimated returns of each state (positions on the grid)
        self.actions_map = {}
        for i in range(len(self.actions)):
            self.actions_map[str(self.actions[i])] = self.actions_meanings[i]
            
        self.optimal_policy = [[ "↓ ↑ ← →" for i in range(self.gridsize)] for j in range(self.gridsize)]
            
            
        
    def create_optimal_policy(self,env):
        for i in range(self.gridsize):
            for j in range(self.gridsize): 
                # Stores the maximum action value function of all actions in given state
                q = float('-inf')
                # Stores Optimal action(s) for given state
                a = ""

                # Iterating over all actions in a state
                for action in self.actions:
                    # performing given action and obtaining next state and reward
                    reward,next_x,next_y = env.perform_action([i,j],action)

                    # checking whether action value of given state is greater/equal to current max
                    if q < reward+self.discount*self.v[next_x,next_y]:
                        q = reward+self.discount*self.v[next_x,next_y]
                        a = self.actions_map[str(action)]
                    elif q == reward+self.discount*self.v[next_x,next_y]:
                        a = a+" "+self.actions_map[str(action)]
                    
                # Updating optimal policy of given state
                self.optimal_policy[i][j] = " "*int((7-len(a))/2) + a + " "*((7-len(a) - int((7-len(a))/2)))
                
        print('\n',np.array(self.optimal_policy))

    def fig3_2(self,env):
        # Iterating until convergence
        while True: 
            # 2d array to store updated states
            next_v = np.zeros((self.gridsize,self.gridsize)) 
            
            for i in range(self.gridsize):
                for j in range(self.gridsize): 
                    # Iterating over all actions in a state
                    for action in self.actions:
                        # performing given action and obtaining next state and reward
                        reward,next_x,next_y = env.perform_action([i,j],action)
                        # updating value function of given state using bellman's equation
                        next_v[i,j] += self.prob_actions*(reward+self.discount*self.v[next_x,next_y]) 
            # Calculating absolute change over iteration
            total_change = float(np.sum(np.absolute(next_v-self.v))) 
            # if change sufficiently small as compared to value function --> convergence
            if total_change < float(np.sum(self.v))*(10**-6): 
                self.v = next_v
                # rounding to 1 decimal place
                self.v = np.round(self.v, 1) 
                print(self.v)
                break
            self.v = next_v
            
    def fig3_5(self,env):
        # Iterating until convergence
        while True: 
            # 2d array to store updated states
            next_v = np.zeros((self.gridsize,self.gridsize)) 
            for i in range(self.gridsize):
                for j in range(self.gridsize): 
                    # Stores the maximum action value function of all actions in given state
                    q = float('-inf')
                    
                    # Iterating over all actions in a state
                    for action in self.actions:
                        # performing given action and obtaining next state and reward
                        reward,next_x,next_y = env.perform_action([i,j],action)
                        # updating max action value in given state
                        q = max(q,reward+self.discount*self.v[next_x,next_y])
                    
                    # Assigning max action value for state as updated state value
                    next_v[i,j] = q
                         
            # Calculating absolute change over iteration
            total_change = float(np.sum(np.absolute(next_v-self.v))) 
            # if change sufficiently small as compared to value function --> convergence
            if total_change < float(np.sum(self.v))*(10**-6): 
                self.v = next_v
                # rounding to 1 decimal place
                self.v = np.round(self.v, 1) 
                print(self.v)
                break
            self.v = next_v
        self.create_optimal_policy(env)


# In[5]:


class Environment:
    
    def __init__(self):
        self.gridsize = 5 # Grid size
        self.A,self.B = [0,1],[0,3] # Special states
        self.A_dest,self.B_dest = [4,1],[2,3] # Destination from special states
        self.actions = [[0,1],[1,0],[0,-1],[-1,0]]  # Right,Down,Left,Up
        self.reward_A,self.reward_B = 10, 5 # Rewards from special states
        self.penalty_outside = -1 # Penalty of exiting grid
        self.reward_other_actions = 0 # Reward for any action onto non special, valid state
        
    def perform_action(self, state, action):
        # Checking for special states
        if state == self.A: 
            return tuple([self.reward_A]+self.A_dest)
        if state == self.B:
            return tuple([self.reward_B]+self.B_dest)
        
        # Calculating next state
        next_state = [state[0]+action[0],state[1]+action[1]]
        # Checking for out of grid state
        if next_state[0]<0 or next_state[1]<0 or next_state[0]>=self.gridsize or next_state[1]>=self.gridsize:
            return tuple([self.penalty_outside]+state)
        # Returning next state and reward
        return tuple([self.reward_other_actions]+next_state)


# In[25]:


# Generating Figure 3.2 by solving system of linear equations (bellman state value function equations of each state)
solver = LinearEqnSolver()
solver.solve_bellman_eqns_fig3_2()


# In[6]:


# Setup to generate fig 3.2 using dynamic programming and policy evaluation
agent = Agent()
env = Environment()
agent.fig3_2(env)   


# In[7]:


# Setup to generate fig 3.5 using dynamic programming
agent2 = Agent()
env2 = Environment()
agent2.fig3_5(env2)   


# In[18]:



    

