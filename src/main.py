import gym
from environment import CarEnv
from agent import Agent
import numpy as np

def main():
    env = CarEnv()
    agent = Agent()
    num_episodes = 10

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            action = agent.act(state)  # Use the agent to decide the action
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)  # Train the agent
            state = next_state
            total_reward += reward
            step_count += 1
            print(f"Episode: {episode + 1}, Steps: {step_count}, Total Reward: {total_reward}")
            env.render()        

    env.close()

if __name__ == "__main__":
    main()