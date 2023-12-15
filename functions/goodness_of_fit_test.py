# -*- coding: utf-8 -*-
"""
 * @version 1.0
 * @author Romain E. LACOSTE 
 *
 * Goodnes-of-fit test for linear exponential Hawkes processes 
"""


#%% Import of the packages  
import torch
import numpy as np
from scipy import stats


#%% Functions 
'''Function for computing the compensator of Hawkes process with exponential kernel.

   Input: - t : 'float', upper bound of the integration interval
          - jump_times 'tensor', jump times of the Hawkes process
          - alpha : 'tensor', intensity rate of the expoential kernel
          - mu : 'tensor', exogenous intensity of the process
          - beta : 'tensor', decay rate of the exponential kernel
            
   Output: - Lambda(t) : 'tensor', compensator of the Hawkes process
'''
def compensator(t, jump_times, alpha, mu, beta):
   
    jump_times = jump_times[jump_times < t]
   
    return ((mu.mul(t)).add(-alpha.mul(torch.sum(torch.exp(beta.mul(-t.add(-jump_times)))-1))))

'''Function for performing the goodness-of-fit test. 

   Input: - jump_times 'tensor', jump times of the Hawkes process
          - alpha : 'tensor', intensity rate of the expoential kernel
          - mu : 'tensor', exogenous intensity of the process
          - beta : 'tensor', decay rate of the exponential kernel
            
   Output: - p-value : 'flaot', associated p-value given by the Kolmogorov-Smirnov test
'''
def goodness_of_fit_test(jump_times, alpha, mu, beta):
   
    unit_poisson_jumptimes = torch.empty(len(jump_times))
   
    for i in range (len(jump_times)):
        unit_poisson_jumptimes[i] = compensator(jump_times[i], jump_times, alpha, mu, beta)
       
    return (stats.kstest(np.array(torch.diff(unit_poisson_jumptimes)), stats.expon.cdf)[1])
