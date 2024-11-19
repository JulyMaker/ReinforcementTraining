import numpy as np
import gymnasium as gym

def test(episodes):

    q_table = np.load('q_tableFL.npy')
    env = gym.make('FrozenLake-v1', render_mode='human')
    
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0

        print(f"\n--- Episodio {episode + 1} ---")
        while not terminated and not truncated:
            env.render()

            action = np.argmax(q_table[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            state = new_state

        total_rewards += episode_reward
        print(f"Recompensa en el episodio {episode + 1}: {episode_reward}")

    env.close()

    print(f"\nRecompensas totales después de {episodes} episodios: {total_rewards}")
    print(f"Recompensa promedio por episodio: {total_rewards / episodes:.2f}")

if __name__ == '__main__':
    test(10)