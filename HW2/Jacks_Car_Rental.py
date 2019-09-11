
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import time


# In[2]:


# Define costants

max_cars = 20 # maximum number of cars
max_car_move = 5 # maximum cars that can be moved

# Poisson distribution means for rentals and returns at both locations
loc_1_rent_mean = 3 
loc_2_rent_mean = 4
loc_1_return_mean = 3
loc_2_return_mean = 2

# Reward for each rental and transfer of car
rental_reward = 10
transfer_reward = -2

# Discount factor
discount = 0.9

# Arrays for storing - possible actions, state value function and policy
actions = np.arange(-max_car_move,max_car_move+1)
values = np.zeros((max_cars+1,max_cars+1),dtype='float32')
policy = np.zeros((max_cars+1,max_cars+1),dtype='float32')

# Buffers to store precomputed values from poisson distributions with different means
poisson_rent_1 = np.zeros((max_cars+1))
poisson_rent_2 = np.zeros((max_cars+1))
poisson_return_1 = np.zeros((max_cars+1))
poisson_return_2 = np.zeros((max_cars+1))


# In[3]:


# Function to precompute poisson pmf for different means
def init_probs():
    for i in range(max_cars+1):
        poisson_rent_1[i] = poisson.pmf(i, loc_1_rent_mean)
        poisson_rent_2[i] = poisson.pmf(i, loc_2_rent_mean)
        poisson_return_1[i] = poisson.pmf(i, loc_1_return_mean)
        poisson_return_2[i] = poisson.pmf(i, loc_2_return_mean)
    


# In[4]:


# Calling function to precompute poisson pmfs
init_probs()


# In[5]:


# Function to display current policy as a heatmap
def display_policy(iteration = -1):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 20) # Inverts y-axis coordinates by setting them in this specific order
    
    if iteration >= 0:
        fig.suptitle("Policy after iteration "+str(iteration))
    else:
        fig.suptitle("Optimal Policy")
    im = ax.imshow(policy, cmap='cool_r', interpolation='nearest') # Creating heatmp
    fig.colorbar(im, orientation='horizontal') 
    plt.show()
    if iteration >= 0:
        fig.savefig("policy-"+str(iteration)+".png", dpi=1200)
    else:
        fig.savefig("policy-opt.png", dpi=1200)
    


# In[6]:


# Function to calculate expected return of a given state action pair
def bellman_eqn_return(state,action):
    c1,c2 = state
    returns = 0.0
    
    # Computing cost of transfering vehicles. In the modified Jack's Car Rental problem, 
    # the cost of one car can be deducted if it is being moved from location 1 to 2
    returns = transfer_reward*abs(action*(action<=0) + (action-1)*(action>0)) 
    
    # Uncomment for original problem
    # returns = transfer_reward*abs(action)
    
    # Number of cars after transfer capped at 'max_car' vehicles at each location
    c1 = min(int(c1 - action),max_cars)
    c2 = min(int(c2 + action),max_cars)
    
    # Comment for original problem
    # Charging additional parking charge per location if number of vehicles at the location exceed 10
    if c1 > 10:
        returns -= 4
    if c2 > 10:
        returns -=4
        
    # Iterate over possible car rental numbers to calculate expected return from rentals
    # Iterating over possible rental and return numbers tht have a probability significant enough to effecct expected return]
    # All probabilities post P[n=12,lambda] are less than 10**-3, hence they are discarded to improve performance (reduce runtime)
    for r_1 in range(12): 
        for r_2 in range(12):
            
            # Calculating actual rental numbers as location cant rent out more cars than it has
            r1,r2 = min(c1,r_1),min(c2,r_2)
            
            # Probability of given rental sequence
            prob = poisson_rent_1[r_1]*poisson_rent_2[r_2]

            # Reward received for given number of rentals
            state_reward = (r1+r2)*rental_reward 
                        
            # Iterate over possible car return numbers
            for ret1 in range(12):
                for ret2 in range(12):
                    # Probability of given return sequence
                    prob_ret = poisson_return_1[ret1]*poisson_return_2[ret2]
                    
                    # updating expectation using bellman's equation
                    returns += prob*prob_ret*(state_reward + discount*values[min(c1-r1+ret1,max_cars),min(c2-r2+ret2,max_cars)])
                                                            
    return returns                      


# In[11]:


def policy_iteration():
    
    iteration = 0
    
    # Run till convergence achieved
    while True:
        
        iteration += 1
        
        ###### Policy Evaluation ######
        print("Evaluating Policy...")
        # Threshold to check for convergence
        threshold = 10**-3
        # Iterating until convergence
        while True:  
            # Absolute difference between consecutive v(s) estimates
            cur_diff = 0
            for i in range(max_cars+1):
                
                # Runtime progress display
                print(i,end=" ")
                
                for j in range(max_cars+1):                         
                    # Next value for current state
                    next_v = 0

                    # performing given action and obtaining expected return
                    next_v = bellman_eqn_return((i,j),policy[i,j])
                    
                    # Updating maximum difference
                    cur_diff = max(cur_diff,abs(next_v-values[i,j]))
                    
                    # Assigning new value to v(s)
                    values[i,j] = next_v
            
            # Displaying max diff of current iteration
            print(" | Max difference = ",cur_diff)

            # If difference sufficiently small --> convergence
            if cur_diff < threshold: 
                break


        ###### Policy Improvement ######
        print("Improving Policy...")
        policy_stable = True
        for i in range(max_cars+1):
            
            # Runtime progress display
            print(i,end=" ")
            
            for j in range(max_cars+1): 
                
                # Stores the maximum action value function of all actions in given state
                q = float('-inf')
                # Stores Optimal action for given state
                a = -1
                                
                # Iterating over all actions in a state
                for act_ind in range(actions.shape[0]):
                    
                    # Skipping actions which result in an invalid state
                    if i-actions[act_ind] < 0 or j+actions[act_ind] > max_cars:
                        continue
                    
                    # performing given action and obtaining expected reward
                    ret = bellman_eqn_return((i,j),actions[act_ind])

                    # updating maximum expected return
                    if q < ret:
                        q = ret
                        a = actions[act_ind]
                        
                # Savin old state to check stability later
                old_state_policy = policy[i,j] 

                # Computing new policy
                policy[i,j] = a
                new_state_policy = policy[i,j] 

                # Checking if policy remained stable in given state
                if old_state_policy != new_state_policy :
                        policy_stable = False


        # If policy stable ---> optimal policy achieved
        if policy_stable:
            break
        
        print('\n\n')
        
        # Displaying policy heatmap
        display_policy(iteration)

    print('\n',values ) 
    print('\n',policy )


# In[8]:


display_policy(0)
policy_iteration()


# In[9]:


display_policy() # Displaying optimal policy

