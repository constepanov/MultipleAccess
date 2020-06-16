import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools
import pyDOE2 as pde
from scipy.special import comb

def get_user_buffer(user_count):
    user_buffer = {}
    for i in range(user_count):
        user_buffer[i] = -1
    return user_buffer


def simulate_adaptive_aloha(lmbda, message_count, user_count, prob_list_size):
    sent_messages = 0
    window_number = 0
    send_prob = 1
    user_buffer = get_user_buffer(user_count)
    delay = 0
    avgN = 0
    index_list = np.arange(prob_list_size)
    index = 0
    while sent_messages < message_count:
        time = random.expovariate(lmbda)
        while time < 1:
            user_number = np.random.randint(user_count)
            if user_buffer[user_number] == -1:
                user_buffer[user_number] = window_number
            time += random.expovariate(lmbda)
        
        for u in range(user_count):
            if user_buffer[u] != -1:
                avgN += 1
        ready_users = []
        
        for i in range(user_count):
            # Формируем массив абонентов, которые собираются передавать в этом окне
            if user_buffer[i] != -1 and user_buffer[i] != window_number:
                r = np.random.uniform()
                if (r < send_prob):
                    ready_users.append(i)
        if len(ready_users) == 1:
            # успех
            delay += window_number - user_buffer[ready_users[0]]
            sent_messages += 1
            user_buffer[ready_users[0]] = -1
        elif len(ready_users) > 1:
            # конфликт
            index = min(index + 1, index_list[-1])
        else:
            # пусто
            index = max(index - 1, index_list[0])
        send_prob = 1 / 2 ** index
        window_number += 1
    delay /= sent_messages
    avgN /= window_number
    return delay, avgN

def get_probability(current_state, new_state, prob_list, lambd, m, l):
    y = math.exp(-lambd / m) # вероятность того, что у пассивного абонента не появилось сообщений
    q = 1 - y # вероятность того, что пассивный абонент стал активным
    nt = current_state[0]
    st = current_state[1]
    nt_next = new_state[0]
    st_next = new_state[1]
    res = 0
    
    # Пусто
    if (st == st_next and st_next == 0 and nt == 0) or (st_next - st == -1 and nt_next >= nt):
        res = comb(m - nt, m - nt_next) * q ** (nt_next - nt) * y ** (m - nt_next) * (1 - prob_list[int(st)]) ** nt
    
    return res

def get_transition_matrix(lambd, m, l):
    matrix_size = (m + 1) * l
    matrix = np.zeros((matrix_size, matrix_size))
    pairs = pde.fullfact([M + 1, l])
    prob_list = []
    for i in range(l):
        prob_list.append(1 / 2 ** i)
    for i in range(matrix_size):
        for j in range(matrix_size):
            current_state = pairs[i]
            new_state = pairs[j]
            matrix[i][j] = get_probability(current_state, new_state, prob_list, lambd, m, l)
    return matrix

lmbda = 1
message_count = 10000
M = 1
l = 1
#matrix = get_transition_matrix(0.1, 4, 3)
#print(matrix)
#simulate_adaptive_aloha(lmbda, message_count, user_count, prob_list_size)

lambdas = np.arange(0.1, 1.1, 0.1)
delay_list = []
avgMsg_list = []
for lmbd in lambdas:
    print("λ = {}".format(lmbd))
    d, n = simulate_adaptive_aloha(lmbd, message_count, M, l)
    delay_list.append(d)
    avgMsg_list.append(n)
plt.plot(lambdas, delay_list, label='d')
plt.legend()
plt.show()
plt.close()

plt.plot(lambdas, avgMsg_list, label='N')
plt.legend()
plt.show()
plt.close()
