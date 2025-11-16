import numpy as np
from utils import policy_one_step_look_ahead, policy_int_to_char

def policy_evaluation(n, pi, v, Gamma, threshhold):
    while True:
        delta = 0
        
        for i in range(n):
            for j in range(n):
                if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                    continue
                
                old_v = v[i, j]
                
                action = pi[i, j]
                next_i = i + policy_one_step_look_ahead[action][0]
                next_j = j + policy_one_step_look_ahead[action][1]
                
                if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
                    next_i, next_j = i, j
                
                v[i, j] = -1 + Gamma * v[next_i, next_j]
                
                delta = max(delta, abs(old_v - v[i, j]))
        
        if delta <= threshhold:
            break
    
    return v

def policy_improvement(n, pi, v, Gamma):
    policy_stable = True
    
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                continue
            
            old_action = pi[i, j]
            
            max_value = float('-inf')
            best_action = 0
            
            for action in range(4):
                next_i = i + policy_one_step_look_ahead[action][0]
                next_j = j + policy_one_step_look_ahead[action][1]
                
                if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
                    next_i, next_j = i, j
                
                value = -1 + Gamma * v[next_i, next_j]
                
                if value > max_value:
                    max_value = value
                    best_action = action
            
            pi[i, j] = best_action
            
            if old_action != best_action:
                policy_stable = False
    
    return pi, policy_stable

def policy_initialization(n):
    return np.random.randint(0, 4, size=(n, n))

def policy_iteration(n, Gamma, threshhold):

    pi = policy_initialization(n=n)

    v = np.zeros(shape=(n,n))

    while True:

        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,Gamma=Gamma)

        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)

        if pi_stable:

            break

    return pi , v

n = 4

Gamma = [0.8,0.9,1]

threshhold = 1e-4

for _gamma in Gamma:

    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)

    print()

    print(pi_char)

    print()
    print()

    print(v)
