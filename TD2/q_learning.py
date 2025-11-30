import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    # Q-Learning update rule: Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if np.random.rand() < epsilone:
        return np.random.randint(0, Q.shape[1])
    else:
        return Q[s].argmax()

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    n_epochs = 1
    max_itr_per_epoch = 200
    rewards = []

    for e in range(n_epochs):
        r = 0  # Total reward for this episode
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, truncated, info = env.step(A)

            r += R # add reward

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            S = Sprime

            if done or truncated:
                break

        epsilon = max(0.01, epsilon * 0.995)

        if e % 100 == 0:
            print(f"Episode #{e} : reward = {r}, epsilon = {epsilon:.3f}")

        rewards.append(r)

    print(f"\nAverage reward over last 100 episodes = {np.mean(rewards[-100:]):.2f}")

    # Plot the rewards
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Moving average
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Moving Average (window={window})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("Training finished.\n")

    # Evaluate the q-learning algorithm
    print("Evaluating trained agent...")
    env_eval = gym.make("Taxi-v3", render_mode="human")
    
    n_eval_episodes = 10
    eval_rewards = []
    
    for e in range(n_eval_episodes):
        S, _ = env_eval.reset()
        total_reward = 0
        
        for _ in range(max_itr_per_epoch):
            # Use greedy policy (no exploration)
            A = Q[S].argmax()
            S, R, done, truncated, _ = env_eval.step(A)
            total_reward += R
            
            if done or truncated:
                break
        
        eval_rewards.append(total_reward)
        print(f"Evaluation episode {e+1}: reward = {total_reward}")
    
    print(f"\nAverage evaluation reward = {np.mean(eval_rewards):.2f}")

    env.close()
    env_eval.close()