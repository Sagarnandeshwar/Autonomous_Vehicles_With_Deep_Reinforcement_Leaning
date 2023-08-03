import gym
from modelDQL import DQL
from util import image2grey
from util import get_frame_stack
from collections import deque
import json


class runDQL():
    def __init__(self, model_loc, save_loc):
        self.model_loc = model_loc
        self.save_loc = save_loc
        self.env = gym.make('CarRacing-v0')
        self.agent = DQL(epsilon=0)
        self.agent.upload_model(self.model_loc)

    def run(self, total_episode):
        #data_dict = {}
        for n_episode in range(1, total_episode + 1):
            start = self.env.reset()
            cur_state = image2grey(start)
            state_queue = deque([cur_state] * 3, maxlen=3)

            total_reward = 0
            time_frame_counter = 1
            done = False

            while True:
                self.env.render()
                cur_state_frame_stack = get_frame_stack(state_queue)
                cur_action = self.agent.choose_action(cur_state_frame_stack)
                next_state, reward, done, info = self.env.step(cur_action)
                total_reward += reward

                next_state = image2grey(next_state)
                state_queue.append(next_state)

                if done:
                    break
                time_frame_counter += 1

            #data_dict[n_episode] = total_reward

        #with open("./performance/dql.json", "w") as fn:
        #    json.dump(data_dict, fn)


if __name__ == '__main__':
    run_model = runDQL("./model/dql.h5", "./performance/dql123.json")
    run_model.run(1)
