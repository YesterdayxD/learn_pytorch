import random

import numpy as np

Q = np.zeros([6, 6])
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
    # init_state = random.choice(choice)
    current_state = 1
    print('current_state: ', current_state)
    current_action_set = np.where(reward[current_state] != -1)
    print('current_action_set: ', current_action_set[0])
    current_action = random.choice(current_action_set[0].tolist())
    print('current_action: ', current_action)
    next_state = current_action
    print('next_state: ', next_state)

    next_action_set = np.where(reward[next_state] != -1)
    print('next_action_set', next_action_set[0])

    quit_condition = gamma * max([Q[next_state, next_action] for next_action in next_action_set[0]])

    Q[current_state, current_action] = reward[current_state, current_action] + quit_condition

    print([Q[next_state, next_action] for next_action in next_action_set[0]])
    print('Q', Q)

# print(Q)
# print(reward)
