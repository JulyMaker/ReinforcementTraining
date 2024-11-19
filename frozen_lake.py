import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt 

def train(episodes):
	env = gym.make('FrozenLake-v1')

	q_table = np.zeros((env.observation_space.n, env.action_space.n))

	learning_rate = 0.1
	discount_factor = 0.95
	epsilon = 1
	epsilon_decay_rate = 0.0001
	rng = np.random.default_rng()

	reward_per_episode = np.zeros(episodes)

	for i in range(episodes):
		#Graphic interface
		if(i + 1) % 1000 ==0:
			env.close()
			env = gym.make('FrozenLake-v1', render_mode = 'human')
		else:
			env.close()
			env = gym.make('FrozenLake-v1')

		# Reset environment and initial state
		state = env.reset()[0]

		terminated = False
		truncated = False
		while(not terminated and not truncated):
			if(rng.random() < epsilon):
				action = env.action_space.sample()    #Explore
			else:
				action = np.argmax(q_table[state,:])  #Explote

			new_state, reward, terminated, truncated, _ = env.step(action)

	        # Q(s,a) = Q(s,a) + ç[R+ymax(Q(s',a'))-Q(s,a)]
			q_table[state,action] = q_table[state,action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state,:]) - q_table[state,action])

			state = new_state

		epsilon = max(epsilon - epsilon_decay_rate, 0)

		if reward == 1:
			reward_per_episode[i] = 1

	    #Print
		if(i + 1) % 100 ==0:
			print(f"Episode: {i+1} - Reward: {reward_per_episode[i]}")

	env.close()

	print(f"Best Q: \n {q_table}")
	np.save('q_tableFL.npy', q_table)

	sum_rewards = np.zeros(episodes)
	for t in range(episodes):
		sum_rewards[t] = np.sum(reward_per_episode[max(0, t-100) : (t+1)])

	plt.plot(sum_rewards)
	plt.show()

if __name__ == '__main__':
	train(15000)