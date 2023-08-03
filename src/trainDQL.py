import gym
from modelDQL import DQL
from collections import deque

from util import image2grey
from util import get_frame_stack

frame_depth = 3
image_size = 96


action_space = [
    (-1, 1, 0.2), (0, 1, 0.2),  (1, 1, 0.2),
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0,   0), (0, 0, 0), (1, 0, 0)]


class trainDQL():
    def __init__(self, file_name, load):
        self.fileName = file_name
        self.env = gym.make('CarRacing-v0')
        self.action_space = action_space
        self.agent = DQL()
        if load:
            self.agent.upload_model(self.fileName)

    def upload_model(self):
        self.agent.upload_model(self.fileName)

    def train(self, total_episodes):
        for n_episode in range(1, total_episodes+1):
            start = self.env.reset()
            cur_state = image2grey(start)
            state_queue = deque([cur_state]*3, maxlen=3)

            total_reward = 0

            negative_reward = 0
            training_counter = 0
            n_iter = 0

            done = False
            while True:
                cur_state_frame_stack = get_frame_stack(state_queue)
                cur_action = self.agent.choose_action(cur_state_frame_stack)
                reward = 0
                for frames in range(2+1):
                    next_state, r, done, info = self.env.step(cur_action)
                    reward += r
                    if done:
                        break

                if reward < 0 and n_iter > 100:
                    negative_reward =+ 1

                if cur_action[1] == 1 and cur_action[2] == 0:
                    reward *= 1.5

                total_reward += reward

                next_state = image2grey(next_state)
                state_queue.append(next_state)
                next_state_frame_stack = get_frame_stack(state_queue)

                self.agent.add_to_memory(cur_state_frame_stack, cur_action, reward, next_state_frame_stack, done)

                if done or negative_reward >= 20 or total_reward < 0:
                    break

                if training_counter > 64:
                    print("training model")
                    self.agent.train(50)
                    training_counter = 0

                n_iter += 1
                training_counter += 1

            print("Run ended")

            if n_episode % 10 == 0:
                print("updating target model")
                self.agent.update_target_model()

            if n_episode % 25 == 0:
                print("saving model")
                self.agent.save_model(self.fileName)
        self.env.close()


if __name__ == "__main__":
    filename_dql = "./model/dql.h5"
    train_m = trainDQL(filename_dql, False)
    train_m.train(200)