# -*- coding: utf-8 -*-
"""
 * @version 1.0
 * @author Romain E. LACOSTE 
 *
 * Example of classification and test on synthetic data
"""

#%% Import of the packages  
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(".."))

from functions.classification import simul_hawkes_mixture, compute_pi, estim_param_ERM
from functions.goodness_of_fit_test import goodness_of_fit_test


#%% Script 
# Hyperparameter 
K = 2
p_star = torch.ones(K)/K 
end_time = 20
n_train = 300
n_test = 1000

# Parameter of the model
baseline = np.array([1.0, 1.0])
intensity = np.array([0.2, 0.7])
decay =  np.array([3.0, 1.5])

# Simulation of the training and test data set
list_jump_times_train, Y_train = simul_hawkes_mixture(K=K, baseline=baseline, intensity=intensity, decay=decay, end_time=end_time, n=n_train)
list_jump_times_test, Y_test = simul_hawkes_mixture(K=K, baseline=baseline, intensity=intensity, decay=decay, end_time=end_time, n=n_test)

# Estimation of p
p_hat = torch.zeros(K)
for k in range(K):
      p_hat[k]=torch.mean((Y_train==k).float())
      
# Bayes classifier 
theta_star = torch.tensor([intensity[0], intensity[1], baseline[0], baseline[1], decay[0], decay[1]], dtype=torch.double)
pi_star = compute_pi(list_jump_times=list_jump_times_test, K=K, param=theta_star, p=p_star, end_time=end_time)
pred_Bayes = torch.argmax(pi_star, dim=1)
error_Bayes = torch.mean(torch.ne(pred_Bayes, Y_test).float())
        

# ERM classifier
init = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.double, requires_grad=True)
theta_hat_ERM = estim_param_ERM(X_train=list_jump_times_train, Y_train=Y_train, p=p_hat, end_time=end_time, K=K, init=init)
pi_hat_ERM = compute_pi(list_jump_times=list_jump_times_test, K=K, param=theta_hat_ERM, p=p_hat, end_time=end_time).detach()
pred_ERM = torch.argmax(pi_hat_ERM, dim=1)
error_ERM = torch.mean(torch.ne(pred_ERM, Y_test).float())

print('Bayes classifier error:', error_Bayes)
print('ERM classifier error:', error_ERM)

cm_ERM = confusion_matrix(Y_test, pred_ERM, normalize='true')
fig, (ax1, axcb) = plt.subplots(1, 2, figsize= (7, 5), sharex=False, sharey=False, gridspec_kw={'width_ratios': [20, 1]})
sns.heatmap(cm_ERM, square=True, annot=True, cmap='Blues', ax=ax1, cbar_ax=axcb)
plt.yticks(rotation=0)
ax1.set_xlabel('Predicted label')
ax1.set_ylabel('True label')
ax1.set_ylim(len(cm_ERM), 0)
plt.tight_layout()


# Performing a goodness-of-fit test according to the predicted labels
p_value_0 = []
p_value_1 = []
rejet_0 = []
rejet_1 = []

for i in range(len(list_jump_times_test)):
        if (pred_ERM[i] == 0):
            p_value = goodness_of_fit_test(list_jump_times_test[i], theta_hat_ERM[0].detach(), theta_hat_ERM[2].detach(), theta_hat_ERM[4].detach())
            p_value_0.append(p_value)
            rejet_0.append((p_value<0.05)*1)
        else:
            p_value = goodness_of_fit_test(list_jump_times_test[i], theta_hat_ERM[1].detach(), theta_hat_ERM[3].detach(), theta_hat_ERM[5].detach())
            p_value_1.append(p_value)
            rejet_1.append((p_value<0.05)*1)
            
print('Mean p-value class 0:', np.mean(p_value_0))
print('Mean p-value class 0:', np.mean(p_value_1))
print('Mean Acceptance rate class 0:', np.mean(1-np.array(rejet_0)))
print('Mean Acceptance rate class 1:', np.mean(1-np.array(rejet_1)))