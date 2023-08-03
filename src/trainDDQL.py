import gym
import numpy as np
from modelDDQL import DDQL
from collections import deque

from util import get_state

frame_depth = 3
image_size = 96


action_space = [
    (-1, 1, 0.2), (0, 1, 0.2),  (1, 1, 0.2),
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0,   0), (0, 0, 0), (1, 0, 0)]


class trainDDQL():
    def __init__(self, file_name, load):
        self.fileName = file_name
        self.env = gym.make('CarRacing-v0')
        self.action_space = action_space
        self.agent = DDQL()
        if load:
            self.agent.upload_model(self.fileName)

    def upload_model(self):
        self.agent.upload_model(self.fileName)

    def train(self, total_episodes):
        for n_episode in range(1, total_episodes+1):
            state = self.env.reset()
            cur_state_buffer = []
            for _ in range(3):
                s = self.env.step(self.action_space[12])[0]
                cur_state_buffer.append(s)

            training_counter = 0
            n_iter = 0
            while True:
                cur_state = get_state(cur_state_buffer)
                cur_action = self.agent.choose_action(cur_state)

                next_state_buffer = []
                reward = 0
                done = False
                for i in range(3):
                    observations, r, d, info = self.env.step(cur_action)
                    reward += r
                    next_state_buffer.append(observations)
                    done = done | d

                if done:
                    break

                if cur_action[1] == 1 and cur_action[2] == 0:
                    reward *= 1.5

                next_state = get_state(next_state_buffer)

                self.agent.add_to_memory(cur_state, cur_action, reward, next_state, done)

                if training_counter > 64:
                    print("training model")
                    self.agent.train(50)
                    training_counter = 0

                cur_state_buffer = next_state_buffer
                n_iter += 1
                training_counter += 1

            if n_episode % 10 == 0:
                print("updating target model")
                self.agent.update_target_model()

            if n_episode % 25 == 0:
                print("saving model")
                self.agent.save_model(self.fileName)

        self.env.close()


if __name__ == "__main__":
    filename_dql = "./model/ddql.h5"
    train_m = trainDDQL(filename_dql, False)
    train_m.train(200)
