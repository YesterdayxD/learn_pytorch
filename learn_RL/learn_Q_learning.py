import random

import numpy as np

Q = np.zeros([5, 5])
reward = np.array([[-1, -1, -1, -1, 0, -1],
                   [-1, -1, -1, 0, -1, 100],
                   [-1, -1, -1, 0, -1, -1],
                   [-1, 0, 0, -1, 0, -1],
                   [0, -1, -1, 0, -1, 100],
                   [-1, 0, -1, -1, 0, 100]
                   ])

gamma = 0.7
choice = [0, 1, 2, 3, 4, 5]

for i in range(1):
    init_state = random.choice(choice)
    pos = np.where(reward[init_state] != -1)
    next = np.random.choice(pos[0], size=1).tolist()

    print(init_state)

# print(Q)
# print(reward)
