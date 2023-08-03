import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque


action_space = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0,   0), (0, 0, 0), (1, 0, 0)]


class DQL():
    def __init__(self, epsilon = 1.0):
        self.depth = 3
        self.action_space = action_space
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.main_model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        new_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu',
                                   input_shape=(96, 96, self.depth)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(4, 4), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(216, activation='relu'),
            tf.keras.layers.Dense(len(self.action_space), activation=None)
        ])

        new_model.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7), loss="mse")
        return new_model

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            action_value = self.main_model.predict(np.expand_dims(state, axis=0), verbose=0)
            action_index = np.argmax(action_value[0])
        else:
            action_index = random.randint(0, len(self.action_space)-1)
        return self.action_space[action_index]

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.main_model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            if done:
                target[action_index] = reward
            else:
                second_target = self.target_model.predict(np.expand_dims(next_state, axis=0),verbose=0)[0]
                target[action_index] = reward + self.gamma * np.amax(second_target)
            train_state.append(state)
            train_target.append(target)
        self.main_model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #self.memory = deque(maxlen=5000)

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def save_model(self, path):
        self.target_model.save_weights(path)

    def upload_model(self, path):
        self.main_model.load_weights(path)
        self.update_target_model()
