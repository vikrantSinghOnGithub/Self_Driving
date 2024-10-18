from agent import Agent
import numpy as np
import tensorflow as tf
from environment import CarEnv

def main():
    # Set the device to GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    else:
        print("No GPU found, using CPU.")

    # Check if TensorFlow is using GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow is using the following devices:")
    print(tf.config.list_physical_devices())

    env = CarEnv()
    agent = Agent()
    num_episodes = 10

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            env.render()
            if not env.paused:
                action = agent.act(state)  # Use the agent to decide the action
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done)  # Train the agent
                state = next_state
                total_reward += reward
                step_count += 1

                print(f"Episode: {episode + 1}, Steps: {step_count}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()