
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class CliffWalk:
    def __init__(self,epsilon,alpha):
        self.grid = (4,12) # Grid dimensions
        self.start_state =  (4-1,1-1) # Start state
        self.stop_state = (4-1,12-1) # Termianl State
        self.rewards = np.ones((4,12))*-1 # Rewards for arriving at a state
        # Describing cliff and its reward on grid
        self.cliff_state_cols = np.arange(10,dtype="uint8") + 1
        self.cliff_state_row = 4-1
        self.rewards[self.cliff_state_row,self.cliff_state_cols] = -100
        # Maping actions to directions 
        self.action_map = {0:(1,0),1:(-1,0),2:(0,1),3:(0,-1)} # South,North,Right,Left
        self.epsilon = epsilon
        self.alpha = alpha
        self.reset_q()
        
    def reset_q(self):
        self.q = np.random.rand(4,12,4)
        self.q[3,11,:] = 0. # Terminal State has 0 value
        
    # Picks greedy action from given state
    def greedy_action(self,state):
        x,y = state
        return np.argmax(self.q[x,y,:])
    
    # Picks epsilon greedy action from given state
    def epsilon_greedy_action(self,state):
        x,y = state
        greedy = np.random.uniform() > self.epsilon
        if greedy:
            return np.argmax(self.q[x,y,:])
        return np.random.randint(4)
    
    # Performs given action on current state
    # Accounts for falling off cliff and trying to go outside grid
    def perform_action(self,state,action):
        x,y = state   
        # Checking if action changes state leading to falling off the cliff 
        if x==self.cliff_state_row and (y in self.cliff_state_cols):
            return self.start_state, self.rewards[state]
        x_n,y_n = (state[0]+self.action_map[action][0],state[1]+self.action_map[action][1])
        # Check if action cause state to go beyond grid  
        if (x_n<0 or x_n>=self.grid[0]) or (y_n<0 or y_n>=self.grid[1]):
            return state,self.rewards[state]
        return (x_n,y_n), self.rewards[state]
    
    # Runs episode using the specified method
    def run_eps(self,episodes,method):
        
        # Tracks rewards received each episode
        reward_sum = np.zeros(episodes)
        
        for ep in range(episodes):
        
            # Begin at start_state each episode
            cur_state = self.start_state
            while cur_state != self.stop_state:
                
                # Generate episode using epsilon greedy behaviour policy
                cur_action = self.epsilon_greedy_action(cur_state)
                # Perform action based on action selected by behaviour policy
                next_state, reward = self.perform_action(cur_state,cur_action)
                
                # Save reward received
                reward_sum[ep]+=reward

                next_action = None
                # Select action based o target policy (Differs between methods)
                if method == "Q":
                    next_action = self.greedy_action(next_state)
                elif method == "SARSA":
                    next_action = self.epsilon_greedy_action(next_state)    

                # Computing index in q value matrix
                cur_ind = cur_state+(cur_action,)
                next_ind = next_state+(next_action,)

                # Updating q vallue for given state and action
                self.q[cur_ind] += self.alpha*(reward+self.q[next_ind]-self.q[cur_ind]) 

                # Updating current state to next state
                cur_state = next_state
                
        return reward_sum          
        


# In[4]:


cw = CliffWalk(epsilon = 0.1,alpha = 0.4)

fig = plt.figure()
ax = fig.add_subplot(111)

runs = 100
episodes = 500

q_rewards = np.zeros(episodes)
sarsa_rewards = np.zeros(episodes)

# iterating over runs
for r in tqdm(range(runs)):
    # Accumulating rewards for q learning
    q_rewards += cw.run_eps(episodes,"Q")
    # reseting q value for next run
    cw.reset_q()
    # Accumulating rewards for sarsa
    sarsa_rewards += cw.run_eps(episodes,"SARSA")
    # reseting q value for next run
    cw.reset_q()
    
# Smooth reward values to reduce noise in output
for i in range(2,episodes):
    q_rewards[i] = (q_rewards[i]+q_rewards[i-1]+q_rewards[i-2])/3.
    sarsa_rewards[i] = (sarsa_rewards[i]+sarsa_rewards[i-1]+sarsa_rewards[i-2])/3.

# Plotting code 
ax.plot(q_rewards/float(runs),label="Q Learning")
ax.plot(sarsa_rewards/float(runs),label="SARSA")

ax.legend(loc='lower right') 
ax.set_xlabel("Episodes")
ax.set_ylabel("Rewards")
ax.set_ylim(-100,0)
fig.savefig("Q_SARSA_Rewards_smoothed.png",dpi=1200)

