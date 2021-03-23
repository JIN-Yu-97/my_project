# -*- coding: utf-8 -*-
# author：Kyle time:16/3/2021
"""
SKUs:	A,B,C, 10 pieces for each, 30 in total, demand independent, same price
Goal:	max revenue in next 10 weeks, set weekly price, unsold with 0 value
Price set:	{999, 899, 799, 699, 599, 499, 399, 299, 199, 99}, adjusted every Mon
Customer arrival:	Week 1-8: N(1000,200)   Week 9-10: N(2000,400), integer，1/50 of the arrival interested in one of
ABC (equal Prob, 1/3)
Customer willingness-to-pay:	Week 1-6: U(0,1000)   Week 7-10: U(0,600)
"""
from MCTS import *


price = [999, 899, 799, 699, 599, 499, 399, 299, 199, 99]


def reward_func(week, start, end, idx):
    return price[idx]*(sum(start) - sum(end))


# generate a random customer dictionary, 1/50 interested, 1/3 in one of ABC
def Customer():
    customer = {}
    first8week = np.random.normal(1000, 200, 8).round()
    first8week = list(map(int, first8week))
    last2week = np.random.normal(2000, 400, 2).round()
    last2week = list(map(int, last2week))
    customer[0] = first8week + last2week
    for week in range(1, 7):
        # interested on ABC
        a = round(customer[0][week - 1] / 150)
        b = round(customer[0][week - 1] / 150)
        c = round(customer[0][week - 1] / 50) - a - b
        ran = np.random.randint(3)
        if ran == 0:
            a = round(customer[0][week - 1] / 150)
            c = round(customer[0][week - 1] / 150)
            b = round(customer[0][week - 1] / 50) - a - c
        if ran == 1:
            c = round(customer[0][week - 1] / 150)
            b = round(customer[0][week - 1] / 150)
            a = round(customer[0][week - 1] / 50) - c - b
        customer[week] = {}
        for n_a in range(0, a):
            W2P = np.random.uniform(0, 1000)
            customer[week][n_a] = ['A', int(W2P)]
        for n_b in range(a, a + b):
            W2P = np.random.uniform(0, 1000)
            customer[week][n_b] = ['B', int(W2P)]
        for n_c in range(a + b, a + b + c):
            W2P = np.random.uniform(0, 1000)
            customer[week][n_c] = ['C', int(W2P)]
    for week in range(7, 11):
        # interested on ABC
        a = round(customer[0][week - 1] / 150)
        b = round(customer[0][week - 1] / 150)
        c = round(customer[0][week - 1] / 50) - a - b
        ran = np.random.randint(3)
        if ran == 0:
            a = round(customer[0][week - 1] / 150)
            c = round(customer[0][week - 1] / 150)
            b = round(customer[0][week - 1] / 50) - a - c
        if ran == 1:
            c = round(customer[0][week - 1] / 150)
            b = round(customer[0][week - 1] / 150)
            a = round(customer[0][week - 1] / 50) - c - b
        customer[week] = {}
        for n_a in range(0, a):
            W2P = np.random.uniform(0, 600)
            customer[week][n_a] = ['A', int(W2P)]
        for n_b in range(a, a + b):
            W2P = np.random.uniform(0, 600)
            customer[week][n_b] = ['B', int(W2P)]
        for n_c in range(a + b, a + b + c):
            W2P = np.random.uniform(0, 600)
            customer[week][n_c] = ['C', int(W2P)]
    return customer


cust = Customer()


def react_func(week, start, idx):
    week += 1
    end = list(start)
    for i in cust[week].keys():
        if cust[week][i][0] == 'A' and end[0] >= 1 and cust[week][i][1] >= price[idx]:
            end[0] -= 1
        if cust[week][i][0] == 'B' and end[1] >= 1 and cust[week][i][1] >= price[idx]:
            end[1] -= 1
        if cust[week][i][0] == 'C' and end[2] >= 1 and cust[week][i][1] >= price[idx]:
            end[2] -= 1
    end.sort()
    return tuple(end)


# mct = MCtreeSearch(decision_set=price, depth=10, initial_depth=0, initial_state=(10, 10, 10), final_state=(0, 0, 0),
#                   ucb_param=1000, react_func=react_func, reward_func=reward_func)

mct = load_tree('msba7003')
mct.set_react_func(react_func)
mct.set_reward_func(reward_func)
mct.set_ucb_param('unstable')
mct.simulate(100000)
mct.save_tree('msba7003')

mct.find_solution(10000)
mct.print_solution()
