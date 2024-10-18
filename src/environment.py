import os
import pygame
import numpy as np
import gym
from gym import spaces

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Example: 3 actions (left, right, forward)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        # Initialize pygame
        pygame.init()
        
        # Load images
        project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        car_path = os.path.join(project_folder, 'resources', 'car.png')
        trace_path = os.path.join(project_folder, 'resources', 'map.png')
        self.car_image = pygame.image.load(car_path)
        self.trace_image = pygame.image.load(trace_path)
        
        # Set screen size based on trace image dimensions
        self.trace_width, self.trace_height = self.trace_image.get_size()
        self.screen = pygame.display.set_mode((self.trace_width, self.trace_height))
        self.clock = pygame.time.Clock()
        
        # Scale car image to fit within the trace
        car_scale_factor = 0.1  # Adjust this factor as needed
        car_width = int(self.car_image.get_width() * car_scale_factor)
        car_height = int(self.car_image.get_height() * car_scale_factor)
        self.car_image = pygame.transform.scale(self.car_image, (car_width, car_height))
        
        # Set starting position for the car
        self.car_position = [700, 650]  # Example starting position, adjust as needed
        self.car_velocity = [0, 0]  # Initial velocity

        # Button properties
        self.button_color = (0, 255, 0)
        self.button_rect = pygame.Rect(10, 10, 100, 50)
        self.paused = False

    def step(self, action):
        # Implement the logic to take an action and return the next state, reward, done, and info
        if action == 0:  # Move left
            self.car_velocity[0] -= 1
        elif action == 1:  # Move right
            self.car_velocity[0] += 1
        elif action == 2:  # Move forward
            self.car_velocity[1] -= 1
        
        # Ensure the car only moves forward
        if self.car_velocity[1] > 0:
            self.car_velocity[1] = 0
        
        # Update car position based on velocity
        new_position = [self.car_position[0] + self.car_velocity[0], self.car_position[1] + self.car_velocity[1]]
        
        # Check if the new position is within bounds
        if 0 <= new_position[0] < self.trace_width and 0 <= new_position[1] < self.trace_height:
            # Check for collision with white color
            pixel_color = self.trace_image.get_at(new_position)
            if pixel_color == (255, 255, 255, 255):  # White color in RGBA
                done = True
                reward = -100  # Penalty for collision
            else:
                self.car_position = new_position
                done = False
                # Reward only for forward movement
                if self.car_velocity[1] < 0:
                    reward = 1  # Reward for moving forward
                else:
                    reward = -1  # Penalty for moving backward or staying still
        else:
            done = True
            reward = -100  # Penalty for going out of bounds
        
        # Placeholder for next state
        next_state = np.zeros((84, 84, 3))  # Placeholder
        info = {}
        
        return next_state, reward, done, info

    def reset(self):
        # Reset the environment to an initial state
        self.car_position = [700, 650]  # Reset to starting position
        self.car_velocity = [0, 0]  # Reset velocity
        initial_state = np.zeros((84, 84, 3))  # Placeholder
        return initial_state

    def render(self, mode='human'):
        if mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.button_rect.collidepoint(event.pos):
                        self.paused = not self.paused  # Toggle pause state

            self.screen.fill((0, 0, 0))  # Clear the screen with black
            self.screen.blit(self.trace_image, (0, 0))  # Draw the trace image
            self.screen.blit(self.car_image, self.car_position)  # Draw the car image

            # Draw the button
            pygame.draw.rect(self.screen, self.button_color, self.button_rect)
            font = pygame.font.Font(None, 36)
            text = font.render('Pause' if not self.paused else 'Resume', True, (0, 0, 0))
            self.screen.blit(text, (self.button_rect.x + 10, self.button_rect.y + 10))

            pygame.display.flip()
            self.clock.tick(30)  # Limit the frame rate to 30 FPS

    def close(self):
        pygame.quit()