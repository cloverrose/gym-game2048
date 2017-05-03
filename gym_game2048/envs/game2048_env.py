# -*- coding:utf-8 -*-
import random

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    size = 4  # width == heigh
    start_tiles = 2  # how many tiles at initial grid

    dtype = np.uint8

    # 0 means 0
    # n > 0 means 2 ** n, e.g. 1 means 2, 11 means 2048
    space_to_real = [0] + [2 ** i for i in range(1, 11 + 1)]
    real_to_space = {real: space for space, real in enumerate(space_to_real)}

    def _setup(self):
        zero_grid = np.zeros((Game2048Env.size, Game2048Env.size), dtype=Game2048Env.dtype)
        self.grid = Game2048Env.add_new_tile(zero_grid, Game2048Env.start_tiles)
        self.score = 0

    def __init__(self):
        self.action_space = spaces.Discrete(4)  # left, right, up, down
        self.observation_space = spaces.MultiDiscrete(
            [[Game2048Env.real_to_space[0], Game2048Env.real_to_space[2048]] for _ in range(Game2048Env.size * Game2048Env.size)])
        self._setup()
        self.illegal_move_mode = 'lose'

    def _step(self, action):
        state, reward, done, info = self.mystep(action)
        return np.reshape(state, Game2048Env.size ** 2), reward, done, info

    def mystep(self, action):
        space_to_func = [Game2048Env.left, Game2048Env.right, Game2048Env.up, Game2048Env.down]
        equal_flag, next_grid, acquire_score = space_to_func[action](self.grid)
        if not equal_flag:
            self.grid = next_grid
            self.score += acquire_score
            if Game2048Env._is_complete(self.grid):
                return self.grid, acquire_score, True, {}
            elif Game2048Env._is_done(self.grid):
                return self.grid, acquire_score, True, {}
            else:
                self.grid = Game2048Env.add_new_tile(self.grid)
                if Game2048Env._is_done(self.grid):
                    return self.grid, acquire_score, True, {}
                else:
                    return self.grid, acquire_score, False, {}
        else:
            if self.illegal_move_mode == 'lose':
                return self.grid, -1.0, False, {}
            elif self.illegal_move_mode == 'continue':
                raise Exception("invalid action")

    def _reset(self):
        self._setup()
        return np.reshape(self.grid, Game2048Env.size ** 2)

    def _render(self, mode='human', close=False):
        print '-' * 80
        print 'score = {0}'.format(self.score)
        for y in range(Game2048Env.size):
            row = [Game2048Env.space_to_real[v] for v in self.grid[y,]]
            print ' '.join(map(str, row))

    @staticmethod
    def _is_complete(grid):
        return np.any(grid == Game2048Env.real_to_space[2048])

    @staticmethod
    def _is_done(grid):
        if np.any(grid == Game2048Env.real_to_space[0]):
            return False
        l_flg, l_grid, l_score = Game2048Env.left(grid)
        if not l_flg:
            return False
        u_flg, u_grid, u_score = Game2048Env.up(grid)
        return u_flg

    @staticmethod
    def add_new_tile(grid, num=1):
        new_tile_values = [Game2048Env.real_to_space[2] if random.random() < 0.9 else Game2048Env.real_to_space[4]
                           for _ in range(num)]
        available_y, available_x = np.where(grid == Game2048Env.real_to_space[0])
        if len(available_y) == 0:
            return grid
        idxs = random.sample(range(len(available_y)), num)
        tile = np.zeros((Game2048Env.size, Game2048Env.size), dtype=Game2048Env.dtype)
        for idx, value in zip(idxs, new_tile_values):
            tile[available_y[idx], available_x[idx]] = value
        next_grid = grid + tile
        return next_grid

    @staticmethod
    def move_left(grid):
        rows = []
        score = 0
        for ri in range(Game2048Env.size):
            row = grid[ri,]
            buff = []
            i = 0
            while i < Game2048Env.size:
                a = row[i]
                if a == Game2048Env.real_to_space[0]:
                    i += 1
                    continue
                b = None
                j = i + 1
                while j < Game2048Env.size:
                    b = row[j]
                    if b != Game2048Env.real_to_space[0]:
                        break
                    j += 1
                if b == a:
                    buff.append(a + 1)
                    score += Game2048Env.space_to_real[a + 1]
                    i = j + 1
                else:
                    buff.append(a)
                    i = j
            rows.append(buff + [0] * (Game2048Env.size - len(buff)))
        next_grid = np.array(rows)
        return next_grid, score

    @staticmethod
    def left(grid):
        next_grid, score = Game2048Env.move_left(grid)
        return np.array_equal(next_grid, grid), next_grid, score

    @staticmethod
    def right(grid):
        _next_grid, score = Game2048Env.move_left(np.fliplr(grid))
        next_grid = np.fliplr(_next_grid)
        return np.array_equal(next_grid, grid), next_grid, score

    @staticmethod
    def up(grid):
        _next_grid, score = Game2048Env.move_left(np.rot90(grid))
        next_grid = np.rot90(_next_grid, -1)
        return np.array_equal(next_grid, grid), next_grid, score

    @staticmethod
    def down(grid):
        _next_grid, score = Game2048Env.move_left(np.rot90(grid, -1))
        next_grid = np.rot90(_next_grid)
        return np.array_equal(next_grid, grid), next_grid, score


def test():
    high_score = 0
    env = Game2048Env()
    for step_i in range(100):
        print "step {0}".format(step_i)
        done = False
        observation = env.reset()
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
        high_score = max(env.score, high_score)
    print "high_score {0}".format(high_score)


if __name__ == '__main__':
    test()
