# -*- coding: utf-8 -*-
# author：Kyle time:16/3/2021
"""
With this package, you can easily
1. do simulation and build a Monte Carlo Tree -> define an MCtreeSearch object, and call simulate()
2. save the tree -> call save_tree()
3. load the pre-built tree -> call load_tree()
4. continue to do simulation and develop the previous tree -> call simulation()
5. give solutions based on the simulation -> call find_solution() and print_solution()

What you need to do:
1. define a react function
2. define a reward function
"""

import json
import math
import numpy as np
import time
import ast


def load_tree(filename, react_func=None, reward_func=None):
    with open('%s.json' % filename, 'r') as f:
        tree = f.read()
        tree = json.loads(tree)
    for key in list(tree.keys()):
        tuple_key = tuple(ast.literal_eval(key))
        tree[tuple_key] = tree.pop(key)
        if tuple_key[1] == 's':
            for key2 in list(tree[tuple_key].keys()):
                tree[tuple_key][tuple(ast.literal_eval(key2))] = tree[tuple_key].pop(key2)
        if tuple_key[1] == 'd':
            for key2 in list(tree[tuple_key].keys()):
                tree[tuple_key][int(ast.literal_eval(key2))] = tree[tuple_key].pop(key2)
    decision_set, depth, initial_depth, initial_state, final_state, C = tree[0, 0]
    mcts_ = MCtreeSearch(decision_set, depth, initial_depth, tuple(initial_state), tuple(final_state), C, react_func,
                         reward_func, tree)
    print('Successfully load the tree!')
    return mcts_


class MCtreeSearch:
    def __init__(self, decision_set=None, depth=None, initial_depth=0, initial_state=None, final_state=None,
                 ucb_param=1, react_func=None, reward_func=None, tree=None):
        self.decision_set = decision_set  # list
        self.n = len(self.decision_set)
        self.depth = depth  # int, how many decisions to make
        self.initial_depth = initial_depth  # start simulation from the ith decision
        self.initial_state = initial_state  # tuple
        self.final_state = final_state  # tuple
        self.C = ucb_param  # hyperparameter for UCB
        self.react_func = react_func
        # (d, start_state, decision_id_in_set), decision to state, return end_state, 0<=d<=depth-1
        self.reward_func = reward_func
        # (d, start_state, end_state, decision_id_in_set), calculate reward for each decision, return reward
        if tree is None:
            self.tree = dict()
        else:
            self.tree = tree
        self.solution = dict()

    def set_decision_set(self, value):
        self.decision_set = value

    def set_depth(self, value):
        self.depth = value

    def set_initial_depth(self, value):
        self.initial_depth = value

    def set_initial_state(self, value):
        self.initial_state = value

    def set_final_state(self, value):
        self.final_state = value

    def set_ucb_param(self, value):
        self.C = value

    def set_react_func(self, value):
        self.react_func = value

    def set_reward_func(self, value):
        self.reward_func = value

    def save_tree(self, filename):
        tree = self.tree.copy()
        tree[0, 0] = [self.decision_set, self.depth, self.initial_depth, self.initial_state,
                      self.final_state, self.C]
        for key in list(tree.keys()):
            if key[1] == 's':
                for key2 in list(tree[key].keys()):
                    tree[key][str(key2)] = tree[key].pop(key2)
            tree[str(key)] = tree.pop(key)
        Obj = json.dumps(tree)
        with open('%s.json' % filename, 'w') as f:
            f.write(Obj)
        print('Successfully save the tree!')

    def expand(self, depth, state):
        idx = len(self.tree[depth, 's'])
        self.tree[depth, 's'][state] = [idx, 0]  # [id, visit_times]
        if (depth + 1, 'd') not in self.tree.keys():
            self.tree[depth + 1, 'd'] = dict()
        for i in range(idx * self.n, idx * self.n + self.n):
            self.tree[depth + 1, 'd'][i] = {'visit_times': 0, 'reward': 0}

    def select(self, depth, state):  # for simulation
        if (depth, 's') not in self.tree.keys():
            self.tree[depth, 's'] = dict()
        if state not in self.tree[depth, 's'].keys():
            self.expand(depth, state)
        state_id, _ = self.tree[depth, 's'].get(state)
        N = self.tree[depth, 's'][state][1]
        decision_id = np.random.randint(0, self.n - 1) + self.n * state_id
        for i in range(self.n * state_id, self.n * state_id + self.n):
            if self.ucb(depth + 1, i, N) > self.ucb(depth + 1, decision_id, N):
                decision_id = i
        return decision_id

    def select2(self, depth, state):  # for giving solutions
        state_id, _ = self.tree[depth, 's'].get(state)
        reward = 0
        decision_id = 0
        for i in range(self.n * state_id, self.n * state_id + self.n):
            if self.tree[depth + 1, 'd'][i]['reward'] > reward:
                reward = self.tree[depth + 1, 'd'][i]['reward']
                decision_id = i
        return decision_id

    def ucb(self, depth, idx, N):  # N is visit_times of the parent node
        if self.tree[depth, 'd'][idx]['visit_times'] == 0:
            return np.inf
        else:
            n = self.tree[depth, 'd'][idx]['visit_times']
            if self.C == 'unstable':
                C = self.tree[depth, 'd'][idx]['reward']
            else:
                C = self.C
            value = self.tree[depth, 'd'][idx]['reward'] + 2 * C * (math.log(N) / n) ** 0.5
            return value

    def backpropagation(self, initial_depth, final_depth, reward_list, decision_id_list, state_list):
        num = final_depth - initial_depth
        for d in range(num):
            self.tree[d + initial_depth, 's'][state_list[d]][1] += 1
            reward = sum(reward_list[d:])
            n = self.tree[d + initial_depth + 1, 'd'][decision_id_list[d]]['visit_times']
            self.tree[d + initial_depth + 1, 'd'][decision_id_list[d]]['reward'] = \
                (n * self.tree[d + initial_depth + 1, 'd'][decision_id_list[d]]['reward'] + reward) / (n + 1)
            self.tree[d + initial_depth + 1, 'd'][decision_id_list[d]]['visit_times'] += 1
        if state_list[-1] not in self.tree[final_depth, 's'].keys():
            self.tree[final_depth, 's'][state_list[-1]] = [len(self.tree[final_depth, 's']), 0]
        self.tree[final_depth, 's'][state_list[-1]][1] += 1

    def simulate(self, K, initial_depth=None, initial_state=None, mute=0):  # simulate K times from start_state
        if initial_state is None:
            initial_state = self.initial_state
        if initial_depth is None:
            initial_depth = self.initial_depth
        if initial_state == self.final_state:
            if mute == 0:
                print('Initial state = final state! Done.')
            return True
        if mute == 0:
            t = ' '.join(time.asctime(time.localtime(time.time())).split()[1: -1])
            print('Now time: %s, from depth=%s state=%s simulate %s times.' % (t, initial_depth, initial_state, K))
        for k in range(K):
            state = initial_state
            reward_list = list()
            decision_id_list = list()
            state_list = list()
            final_depth = self.depth
            for d in range(initial_depth, self.depth):
                start_state = state
                state_list.append(start_state)
                decision_id = self.select(d, start_state)
                decision_id_list.append(decision_id)
                state = self.react_func(d, start_state, decision_id % self.n)
                reward = self.reward_func(d, start_state, state, decision_id % self.n)
                reward_list.append(reward)
                if state == self.final_state:
                    final_depth = d + 1
                    break
            state_list.append(state)
            if (final_depth, 's') not in self.tree.keys():
                self.tree[final_depth, 's'] = {state: [0, 0]}

            self.backpropagation(initial_depth, final_depth, reward_list, decision_id_list, state_list)

            if mute == 0:
                monitor = int(K / 10)
                if ((k % monitor) == 0) and (k > 0):
                    t = ' '.join(time.asctime(time.localtime(time.time())).split()[1: -1])
                    print('Now time: %s, %s percent done.' % (t, 10 * int(k / monitor)))
        if mute == 0:
            t = ' '.join(time.asctime(time.localtime(time.time())).split()[1: -1])
            print('Now time: %s, all done!' % t)

    def find_solution(self, K, initial_depth=0, initial_state=None):
        if initial_state is None:
            state = self.initial_state
        else:
            state = initial_state
        solution = dict()
        for d in range(initial_depth, self.depth):
            start_state = state
            if start_state == self.final_state:
                break
            if start_state not in self.tree[d, 's'].keys():
                self.simulate(K, d, start_state, mute=1)
            decision_id = self.select2(d, start_state)
            state = self.react_func(d, start_state, decision_id % self.n)
            reward = self.reward_func(d, start_state, state, decision_id % self.n)
            solution[d, start_state] = [decision_id % self.n, state, reward]
        self.solution = solution

    def print_solution(self):
        reward = list()
        for (depth, state) in self.solution.keys():
            a, b, c = self.solution[depth, state]
            reward.append(c)
            print('Depth=%s, state=%s: decision=%s, end_state=%s, reward=%s' % (depth, state, a, b, c))
        print('Total reward:', sum(reward))