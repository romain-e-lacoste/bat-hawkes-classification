# -*- coding: utf-8 -*-
"""
 * @version 1.0
 * @author Romain E. LACOSTE 
 *
 * Classification of linear exponential Hawkes processes path
"""


#%% Import of the packages  
import torch
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(".."))

from functions.simulation import simul_hawkes_process

#%% Functions
'''Function for simulation of K-mixture of Hawkes process with exponential kernel
using cluster construction. 

   Input: - K : 'int', number of class of the mixture model
          - baseline : 'tensor', exogenous intensity of the process for each class
          - intensity : 'tensor', intensity rate of the exponential kernel for each class
          - decay : 'tensor', decay rate of the exponential kernel for each class
          - end_time : 'int', upper bound of the observation interval
          - n : 'int', number of observations 
            
   Output: - list_jump_times : 'list', list of jump times 
           - Y : 'list', labels associated with the observed path 
'''
def simul_hawkes_mixture(K, baseline, intensity, decay, end_time, n):
    
    # Creation of the label and feature list
    Y = torch.bernoulli(torch.ones(n)*0.5)
    list_jump_times = [None]*n
    
    for k in range(0,K):
        ind_k  = torch.where(Y == k)[0]
        for i in ind_k:
            list_jump_times[i] = simul_hawkes_process(end_time, baseline[k], intensity[k], decay[k])
            
    return(list_jump_times, Y)


''''Function for transforming the tensor of parameter
[alpha_1, ..., alpha_K, mu_1, ... mu_K, beta_1, ..., beta_K] into the tensor intensity alpha, 
into the tensor baseline mu and into the tensor decay beta. 

   Input: - param : 'tensor', given parameter
          - K : 'int', number of class of the mixture model
            
   Output: - alpha : 'tensor' intensity parameter
           - mu : 'tensor' baseline parameter
           - beta : 'tensor' decay parameter
'''
def reshape_param(param, K):
    
    alpha = param[0:K]
    mu = param[K:2*K]
    beta = param[2*K:3*K].reshape(2,1)
    
    return(alpha, mu, beta)

'''Function for computing pi_param, that is the regression function based on
the parameter theta given in input for each repetitions.

   Input : - list_jump_times : 'list', list of jump times
           - K : 'int', number of class of the mixture model
           - param : 'tensor', given parameter 
           - p : 'tensor', distribution of Y 
           - end_time : 'int', upper bound of the observation interval
             
   Output : - pi : 'tensor', computation of pi based on the parameter given in input
'''
def compute_pi(list_jump_times, K, param, p, end_time):
    
    # First get each parameter in the right shape
    alpha, mu, beta = reshape_param(param, K)
    n = len(list_jump_times)
    pi = torch.empty([n,K], dtype=torch.double)
    for rep in range(n):
        # If the process have jumped at least once 
        if (len(list_jump_times[rep])>0):
            time_max = list_jump_times[rep][-1]
            term_1 = (mu.mul(-end_time)).add(alpha.mul(torch.sum(torch.exp(beta.mul(-time_max.add(-list_jump_times[rep])))-1, dim=1)))
            jumps = list_jump_times[rep][list_jump_times[rep] < end_time]
            term_2 = sum([torch.log(mu.add((alpha.mul(beta.reshape((K)))).mul(torch.sum(torch.exp(-beta.mul(l.add(-list_jump_times[rep][list_jump_times[rep]<l]))), dim=1))) + 0.00001) for l in jumps])
            estim_F = term_1.add(term_2)
        
        # If the process have not jumped at least once 
        if (len(list_jump_times[rep])==0):
            estim_F = mu.mul(-end_time)
        
        # If the number precision limit is exceeded 
        if (estim_F[0] >= torch.tensor(709)):  
            estim_F[0] = torch.tensor(709, dtype=torch.double)
        if (estim_F[1] >= torch.tensor(709)):
            estim_F[1] = torch.tensor(709, dtype=torch.double)
            
        pi[rep, :] =  ((torch.exp(estim_F)).mul(p)).div(torch.sum((torch.exp(estim_F)).mul(p)))
        
    return(pi)

def estim_param_ERM(X_train, Y_train, p, end_time, K, init):
    Z  = torch.ones((Y_train.size(dim=0), K)).mul(-1)
    for k in range(Y_train.size(dim=0)):
        Z[k,int(Y_train[k])]=1

    def compute_phi_risk(theta):
        target_F = compute_pi(list_jump_times=X_train, K=K, param=torch.abs(theta), p=p, end_time=end_time).mul(2)-1
        return (torch.mean(torch.sum(torch.pow(Z-target_F, 2) , dim=1)))
    
    # Choose SGD as the optimizer and StepLR for learning rate decay
    optimizer = torch.optim.SGD([init], lr=0.008)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    for _ in tqdm(range(100), desc = 'Progress Bar'):
        optimizer.zero_grad(set_to_none=True)
        value = compute_phi_risk(init)
        value.backward()
        optimizer.step()
        scheduler.step()
        
    theta_hat = torch.abs(init)
    
    return(theta_hat)
