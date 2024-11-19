import gymnasium as gym
import numpy as np

def discretizer(value, env):
    aux = ((value - env.observation_space.low) / 
           (env.observation_space.high - env.observation_space.low)) * 20
    return tuple(aux.astype(np.int32))

def test(episodes):
    q_table = np.load('q_tableMC.npy')

    env = gym.make('MountainCar-v0', render_mode='human')

    total_rewards = 0
    for episode in range(episodes):
        state = discretizer(env.reset()[0], env)
        done = False
        episode_reward = 0

        print(f"\n--- Episodio {episode + 1} ---")
        while not done:
            env.render()

            action = np.argmax(q_table[state])
            new_state, reward, done, _, _ = env.step(action)

            episode_reward += reward
            state = discretizer(new_state, env)

        total_rewards += episode_reward
        print(f"Recompensa en el episodio {episode + 1}: {episode_reward}")

    env.close()

    print(f"\nRecompensas totales después de {episodes} episodios: {total_rewards}")
    print(f"Recompensa promedio por episodio: {total_rewards / episodes:.2f}")

if __name__ == '__main__':
    test(5)
