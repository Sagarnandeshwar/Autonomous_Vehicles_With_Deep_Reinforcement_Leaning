import numpy as np
import cv2


def image2grey(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    # state /= 255.0
    return state


def get_frame_stack(queue):
    return np.transpose(np.array(queue), (1, 2, 0))


def get_state(state_buffer):
    state = np.array(state_buffer)
    state = np.reshape(state, (96, -1, 3))
    return state

