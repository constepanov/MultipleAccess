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
    requests = 0
    while sent_messages < message_count:
        time = random.expovariate(lmbda)
        requests += 1
        while time < 1:
            user_number = np.random.randint(user_count)
            if user_buffer[user_number] == -1:
                user_buffer[user_number] = window_number
            time += random.expovariate(lmbda)
        
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
        for u in range(user_count):
            if user_buffer[u] != -1:
                avgN += 1
        send_prob = 1 / 2 ** index
        window_number += 1
    delay /= sent_messages
    avgN /= window_number
    lambd_out = sent_messages / requests
    return delay, avgN, lambd_out

def get_probability(current_state, new_state, prob_list, lambd, m, l):
    y = math.exp(-lambd / m) # вероятность того, что у пассивного абонента не появилось сообщений
    q = 1 - y # вероятность того, что пассивный абонент стал активным
    nt = int(current_state[0]) # Число активных пользователей в текущем состоянии
    st = int(current_state[1]) # Индекс вероятности в текущем состоянии
    nt_next = int(new_state[0])
    st_next = int(new_state[1])
    res = 0

    # Пусто
    if (st == st_next and st_next == 0 and nt == 0) or (st_next - st == -1 and nt_next >= nt):
        #print('Empty:', current_state, new_state)
        res += (comb(m - nt, nt_next - nt) * 
                (q ** (nt_next - nt)) * 
                (y ** (m - nt_next)) *
                ((1 - prob_list[st]) ** nt))
    
    # Успех
    if ((st_next == st and nt > 0 and nt_next >= nt - 1 and st_next != 0) or (nt == 1 and st == 0 and st == st_next)) and nt_next < m:
        #print('Success:', current_state, new_state)
        res += (comb(m - nt, nt_next - nt + 1) * 
                (q ** (nt_next - nt + 1)) *
                (y ** (m - nt_next - 1)) *
                nt * prob_list[st] *
                ((1 - prob_list[st]) ** (nt - 1)))

    if ((st_next - st == 1) or (st_next == st and st_next == (l - 1))) and (nt_next >= nt) and (nt > 1):
        #print('Conflict:', current_state, new_state)
        res += (comb(m - nt, nt_next - nt) *
                (q ** (nt_next - nt)) *
                (y ** (m - nt_next)) *
                (1 - nt * prob_list[st] * ((1 - prob_list[st]) ** (nt - 1)) - (1 - prob_list[st]) ** nt))
        
    return res

def stationary_distribution(transition_matrix):
    vec = np.zeros(len(transition_matrix))
    vec[-1] = 1
    pt = transition_matrix.transpose()
    for i in range(len(pt)):
        pt[i][i] -= 1
        pt[-1][i] = 1
    #res = np.linalg.inv(pt).dot(vec)
    #res = vec.dot(np.linalg.inv(pt))
    res = np.linalg.solve(pt, vec)
    return res

def get_transition_matrix(lambd, m, l, pairs, prob_list):
    matrix_size = (m + 1) * l
    matrix = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            current_state = pairs[i]
            new_state = pairs[j]
            matrix[i][j] = get_probability(current_state, new_state, prob_list, lambd, m, l)
    return matrix

message_count = 40000
M = 15
l = 5
pairs = pde.fullfact([M + 1, l])
prob_list = []
for i in range(l):
    prob_list.append(1 / 2 ** i)
print(pairs)
lambdas = np.arange(0.1, 1, 0.1)
delay_list = []
avgMsg_list = []
avg_n_th_list = []
delay_theor_list = []
lambd_out_list = []
for lmbd in lambdas:
    print("λ = {}".format(lmbd))
    d, n, lambd_out = simulate_adaptive_aloha(lmbd, message_count, M, l)
    matrix = get_transition_matrix(lmbd, M, l, pairs, prob_list)
    dist = stationary_distribution(matrix)
    print(np.sum(dist))
    Pr = np.zeros(M)
    for i in range(M):
        for j in range((M + 1) * l):
            if pairs[j][0] == i:
                Pr[i] += dist[j]
    
    avg_n_th = 0
    for i in range(M):
        avg_n_th += i * Pr[i]
    lambd_out_theor = 0
    for j in range((M + 1) * l):
        u1 = int(pairs[j][0])
        u2 = int(pairs[j][1])
        if u1 != 0:
            lambd_out_theor += (u1 * prob_list[u2] * (1 - prob_list[u2]) ** (u1 - 1) * dist[j])
    
    avg_n_th_list.append(avg_n_th)
    delay_theor_list.append(avg_n_th / lambd_out_theor)
    delay_list.append(d)
    avgMsg_list.append(n)
    lambd_out_list.append(lambd_out)

plt.plot(lambdas, delay_list, label='d')
plt.plot(lambdas, delay_theor_list, label='d_theor')
plt.plot(lambdas, avgMsg_list, label='N')
plt.plot(lambdas, avg_n_th_list, label='N_theor')
plt.legend()
plt.show()
plt.close()

plt.plot(lambdas, lambd_out_list, label='lambda out')
plt.legend()
plt.show()
plt.close()