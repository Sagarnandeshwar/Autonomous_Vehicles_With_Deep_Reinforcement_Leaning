import gym
from modelDDQL import DDQL
import json
from util import get_state

action_space = [
    (-1, 0,   0), (0, 0, 0), (1, 0, 0),
    (-1, 1, 0.2), (0, 1, 0.2),  (1, 1, 0.2),
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2)]


class runDDQL():
    def __init__(self, model_loc, save_loc):
        self.model_loc = model_loc
        self.save_loc = save_loc
        self.action_space = action_space
        self.env = gym.make('CarRacing-v0')
        self.agent = DDQL(epsilon=0)
        self.agent.upload_model(self.model_loc)

    def run(self, total_episode):
        #data_dict = {}
        for n_episode in range(1, total_episode + 1):
            state = self.env.reset()
            cur_state_buffer = []
            for _ in range(3):
                s = self.env.step(self.action_space[12])[0]
                cur_state_buffer.append(s)

            n_iter = 0
            total_rewards = 0

            while True:
                cur_state = get_state(cur_state_buffer)
                cur_action = self.agent.choose_action(cur_state)

                cur_state_buffer = []

                reward = 0
                done = False
                for i in range(3):
                    obs, temp_reward, temp_done, _ = self.env.step(cur_action)
                    reward += temp_reward
                    cur_state_buffer.append(obs)
                    done = done | temp_done

                total_rewards += reward
                if done:
                    break
                n_iter += 1

            #data_dict[n_episode] = total_rewards

        #with open("./performance/ddql.json", "w") as fn:
        #    json.dump(data_dict, fn)


if __name__ == '__main__':
    run_model = runDDQL("./model/dql.h5", "./performance/dql.json")
    run_model.run(1)
