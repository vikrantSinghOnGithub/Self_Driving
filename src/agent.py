import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class Agent:
    def __init__(self):
        self.model = self.build_model()
        self.memory = []

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(84, 84, 3)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(3, activation='linear'))  # Assuming 3 actions
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

        if len(self.memory) > 32:
            batch = np.random.choice(len(self.memory), 32)
            for i in batch:
                state, action, reward, next_state, done = self.memory[i]
                target = reward
                if not done:
                    next_state = np.expand_dims(next_state, axis=0)
                    target += 0.95 * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(np.expand_dims(state, axis=0))
                target_f[0][action] = target
                self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)