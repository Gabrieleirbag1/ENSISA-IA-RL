import numpy as np
from utils import policy_int_to_char, policy_one_step_look_ahead

def value_iteration(n, Gamma, threshold):
    v = np.zeros((n, n))
    pi = np.zeros((n, n), dtype=int)
    
    while True:
        delta = 0
        
        for i in range(n):
            for j in range(n):
                if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                    continue
                
                old_v = v[i, j]
                
                max_value = float('-inf')
                best_max_value = float('-inf')
                best_action = 0
                for action in range(4):
                    next_i = i + policy_one_step_look_ahead[action][0]
                    next_j = j + policy_one_step_look_ahead[action][1]
                    
                    if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
                        next_i, next_j = i, j
                    
                    value = -1 + Gamma * v[next_i, next_j] # Bellman immediate cost (-1 + future value actualized)
                    max_value = max(max_value, value)

                    if value > best_max_value:
                        best_max_value = value
                        best_action = action
                
                v[i, j] = max_value
                pi[i, j] = best_action
                delta = max(delta, abs(old_v - v[i, j]))
        
        if delta <= threshold:
            break
    
    return pi, v

n = 4

Gamma = [0.8,0.9,1]

threshhold = 1e-4

for _gamma in Gamma:

    pi , v = value_iteration(n=n,Gamma=_gamma,threshold=threshhold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print(f"Value Iteration Policy for Gamma={_gamma}:\n{pi_char}\n")
    print(f"Value Function for Gamma={_gamma}:\n{v}\n")