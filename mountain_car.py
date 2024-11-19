import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt 

def discretizer(value, env):
	aux = ((value - env.observation_space.low)/(env.observation_space.high - env.observation_space.low))*20
	return tuple(aux.astype(np.int32))

def train(episodes):
	env = gym.make('MountainCar-v0')

	q_table = np.random.uniform(low=-1, high=1, size=[20,20,3])
	#q_table = np.zeros((env.observation_space.n, env.action_space.n))

	learning_rate = 0.1
	discount_factor = 0.95
	epsilon = 1
	epsilon_decay_rate = 0.0001
	reward_per_episode = np.zeros(episodes)

	for i in range(episodes):
		#Graphic interface
		if(i + 1) % 1000 ==0:
			env.close()
			env = gym.make('MountainCar-v0', render_mode = 'human')
		else:
			env.close()
			env = gym.make('MountainCar-v0')

		# Reset environment and initial state
		state = discretizer(env.reset()[0], env)

		done = False
		while not done:
			if(np.random.randint(0,10) > 2):
				action = np.argmax(q_table[state])
			else:
				action = np.random.randint(0,2)

			new_state, reward, done, info, _ = env.step(action)

			# Q(s,a) = Q(s,a) + ç[R+ymax(Q(s',a'))-Q(s,a)]
			q_table[state][action] = q_table[state][action] + learning_rate * (reward + discount_factor * np.max(q_table[discretizer(new_state, env)]) - q_table[state][action])
	
			state = discretizer(new_state, env)

			reward_per_episode[i] += reward

		#Print
		if(i + 1) % 100 ==0:
			print(f"Episode: {i+1} - Reward: {reward_per_episode[i]}")

	env.close()

	print(f"Best Q: \n {q_table}")
	np.save('q_tableMC.npy', q_table)

	sum_rewards = np.zeros(episodes)
	for t in range(episodes):
		sum_rewards[t] = np.sum(reward_per_episode[max(0, t-100) : (t+1)])

	plt.plot(sum_rewards)
	plt.show()

if __name__ == '__main__':
	train(4000)