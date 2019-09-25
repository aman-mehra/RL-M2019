
# coding: utf-8

# In[31]:


import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


class MRP:
    
    def __init__(self,alpha):
        self.terminal_states = [0,6]
        self.valid_states = [1,2,3,4,5]
        
        # Setting rewards for all transitions
        self.rewards = np.zeros((7,7))
        self.rewards[5,6] = 1.
        
        self.true_v = [1./6,2./6,3./6,4./6,5./6]
        self.alpha = alpha
        self.reset_states()
        
    # Resets state value function at the start of a new run
    def reset_states(self):
        self.v = np.zeros(7)
        self.v[1:6] += 0.5
        
    def set_alpha(self,alpha):
        self.alpha = alpha
        
    # returns --> -1/1 === left/right direction
    def choose_direction(self): 
        return np.random.randint(2)*2 - 1
    
    # generates rms loss
    def get_rms_loss(self):     
        err = 0        
        for ind in range(1,6):
            err += (self.v[ind] - self.true_v[ind-1])**2    
        err /= 5.0
        return err
            
    # generates episode
    def generate_epsiode(self):
        prev_state = 3
        episode_history = []
        # Loop while not in terminal state
        while prev_state not in self.terminal_states:
            # get next state
            next_state = prev_state + self.choose_direction()
            # get reward for transition
            reward = self.rewards[prev_state,next_state]
            # add to episode
            episode_history.append((prev_state,reward))
            prev_state = next_state
        episode_history.append((prev_state,0))
        return episode_history
            
    # runs constant alpha monte carlo simulation
    def monte_carlo_const(self,episodes):
        
        # stores the rms error of each epsiode
        err_eps = np.zeros(episodes)
        
        for ep in range(episodes):
            
            err_eps[ep] = self.get_rms_loss()
            episode_history = self.generate_epsiode()
            # get return for epsiode
            ret = episode_history[-2][1] # stores reward for tranition to terminal state (only possible non 0 reward )
            
            for ind in range(len(episode_history)-1):
                state,_ = episode_history[ind]
                # update value function
                self.v[state] = (1-self.alpha)*self.v[state] + self.alpha*(ret)
                
        return err_eps
        
        
    # runs temporal difference (0) simulation
    def td_0(self,episodes,plot_at_eps = [],plot_fig = True):
        
        est_fig, est_ax = None, None 
        # Required only while comparing state values between episodes (first figure)
        if plot_fig:
            est_fig = plt.figure()
            est_ax = est_fig.add_subplot(111)     
        err_eps = np.zeros(episodes)
        
        for ep in range(episodes):
            
            err_eps[ep] = self.get_rms_loss()        
            if ep in plot_at_eps:
                est_ax.plot(self.v[1:6],label = "Ep="+str(ep))
            
            episode_history = self.generate_epsiode()
            # Stepping through episode
            for ind in range(len(episode_history)-1):
                # getting current state and reward for transition from it
                state,reward = episode_history[ind] 
                next_state,_ = episode_history[ind+1]
                # Calculating update of state value using td(0)
                self.v[state] = (1-self.alpha)*self.v[state] + self.alpha*(reward + self.v[next_state])
           
        # Required only while comparing state values between episodes (first figure)
        if plot_fig:
            est_ax.plot(self.true_v,label="Truth")
            est_ax.set_xticklabels([" ","A","B","C","D","E"])
            est_ax.set_xlim(-1,5)
            est_ax.set_xlabel("States")
            est_ax.set_ylabel("Estimates")
            est_ax.legend(loc='lower right')
            est_fig.savefig("TD_0_estimates.png",dpi=1200)
            
        return err_eps

            
                
        


# In[30]:


mrp = MRP(alpha = 0.1)
mrp.td_0(101,[0,1,10,100])

runs = 100 # Total runs of 100 episodes each
algos = ["td","mc"] # algorithms to test on 
alphas = {"td":[0.05,0.1,0.15],"mc":[0.01,0.02,0.03,0.04]} # alphas to test on for each algorithm

rms_fig = plt.figure()
rms_ax = rms_fig.add_subplot(111)
        
for algo in algos:
    for alpha in alphas[algo]:
        mrp.set_alpha(alpha)
        err = np.zeros(100)
        for r in tqdm(range(runs)):
            mrp.reset_states()
            if algo == "td":
                err += mrp.td_0(100,[],False) # accumulating error from td(0) method
            else:
                err += mrp.monte_carlo_const(100) # accumulating error from monte carlo method
        rms_ax.plot((err/float(runs))**0.5,label="α="+str(alpha)+"("+algo.upper()+")")
rms_ax.legend(loc='upper right') 
rms_ax.set_xlabel("Episodes")
rms_ax.set_ylabel("RMS Error")
rms_fig.savefig("TD_MC_RMS.png",dpi=1200)

