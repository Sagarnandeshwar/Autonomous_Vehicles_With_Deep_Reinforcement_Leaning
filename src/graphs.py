import json
import matplotlib.pyplot as plt
import numpy as np


def avg_reward_graph(filename):
    f = open(filename)
    data = json.load(f)
    x = []
    y = []
    for ind in range(1, 25+1):
        x.append(ind)
        y.append(data[str(ind)])

    y_mean = np.mean(y)

    plt.figure(dpi=140)
    plt.plot(x, y, color='b')
    plt.axhline(y_mean, color='#000000')
    plt.legend(["Returns", "Mean"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return vs Episode Plot for DDQL")
    #plt.show()
    plt.savefig("ddql.png")


if __name__ == "__main__":
    avg_reward_graph('./performance/ddql.json')
