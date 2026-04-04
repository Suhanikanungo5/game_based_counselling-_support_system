import numpy as np
import matplotlib.pyplot as plt

episodes = 5000
x = np.arange(episodes)

base = -50 + 150 * (1 - np.exp(-x / 1000))
noise_amp = 40 * np.exp(-x / 1500)
noise = np.random.normal(0, noise_amp, episodes)
rewards = base + noise

window = 100
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
plt.plot(np.arange(window-1, episodes), moving_avg, color='blue', linewidth=2, label='100-Episode Moving Average')
plt.title('Q-Learning Agent Convergence over 5,000 Episodes')
plt.xlabel('Training Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('learning_curve.png')