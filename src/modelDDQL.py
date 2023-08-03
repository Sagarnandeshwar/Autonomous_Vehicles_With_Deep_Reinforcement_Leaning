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

class DDQL():
    def __init__(self, epsilon = 1.0):
        self.depth = 3
        self.action_space = action_space
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.TAU = 0.01
        self.main_model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        new_model = tf.keras.models.Sequential([
            tf.keras.layers.Convolution2D(64, 8, 8, subsample=(4, 4), strides=3, activation='relu',
                                   input_shape=(96, 96, self.depth)),
            tf.keras.layers.Convolution2D(128, 4, 4, subsample=(2, 2), activation='relu'),
            tf.keras.layers.Convolution2D(128, 3, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(len(self.action_space), activation=None)
        ])
        new_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        return new_model

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            action_value = self.main_model.predict(state.reshape(1, 96, 96*3, self.depth), batch_size = 1, verbose=0)
            action_index = np.argmax(action_value)
        else:
            action_index = random.randint(0, len(self.action_space)-1)
        return self.action_space[action_index]

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.main_model.predict(state.reshape(1, 96, 96 * 3, 3), batch_size=1)
            if done:
                target[action_index] = reward
            else:
                second_target = self.target_model.predict(next_state.reshape(1, 96, 96 * 3, 3), batch_size=1)
                target[action_index] = reward + self.gamma * np.amax(second_target)
            train_state.append(state)
            train_target.append(target)
        self.main_model.fit(np.array(train_state), np.array(train_target))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        main_model_w = self.main_model.get_weights()
        target_model_w = self.target_model.get_weights()
        for i in range(len(main_model_w)):
            target_model_w[i] = self.TAU * main_model_w[i] + (1 - self.TAU) * target_model_w[i]
        self.target_model.set_weights(target_model_w)

    def save_model(self, path):
        self.target_model.save_weights(path)

    def upload_model(self, path):
        self.main_model.load_weights(path)
        self.update_target_model()
        