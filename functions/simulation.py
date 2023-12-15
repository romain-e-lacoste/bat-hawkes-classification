# -*- coding: utf-8 -*-
"""
 * @version 1.0
 * @author Romain E. LACOSTE 
 *
 * Simulation of Hawkes process path with exponential kernel
"""


#%% Import of the packages  
import numpy as np
import torch

#%% Functions 
'''Function for simulation of Hawkes process with exponential kernel
using cluster construction. 

   Input:  - end_time : 'float', upper bound of the observation interval
           - baseline : 'float', exogenous intensity of the process
           - intensity : 'float', intensity rate of the expoential kernel
           - decay : 'float', decay rate of the exponential kernel
            
   Output: - jump_times : 'tensor', jump times of the associated process
'''
def simul_hawkes_process(end_time, baseline, intensity, decay):
    
    # Creation of the jump times list
    jump_times = []
    
    # Simulation of the immigrants driven by the baseline intensity
    if (baseline < 0):
        raise ValueError("The baseline of the Hawkes process should be positive")
    elif (baseline == 0):
        return torch.tensor(jump_times)
    else:
        immigrants = [np.random.uniform(0, end_time) for _ in range(np.random.poisson(baseline*end_time))]
    jump_times.extend(immigrants)
    
    # Simulation of the offsprings 
    gen = 0
    ancestor = immigrants
    while (len(ancestor) > 0):
        offsprings = []
        offsprings_nb_list = [np.random.poisson(intensity) for _ in range(len(ancestor))]
        for i in range(len(offsprings_nb_list)):
            offsprings.extend([np.random.exponential(1/decay) + ancestor[i] for _  in range(offsprings_nb_list[i])])
        jump_times.extend(offsprings)
        ancestor = offsprings 
        gen += 1 
    
    # Remove elements above the upper bound 
    jump_times = [x for x in jump_times if x < end_time] 
    
    # Sort the list of jump times
    jump_times.sort()
    
    return(torch.tensor(jump_times))