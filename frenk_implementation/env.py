import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from a2c_ppo_acktr.distributions import FixedNormal
import random
import matplotlib.pyplot as plt
import math

VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 5.79, 5.79
HEIGHT_ALL, WIDTH_ALL = 8.0, 8.0
FRAME_RATE = 60
PI = np.pi
DELTA_X = 1.105
DELTA_Y = 1.105
"""
implementation of the rdw env
"""

def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


def outbound(x, y):
    if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT:
        return True
    else:
        return False



def min_length_direction(x, y, a, b, cos):  # cause the heading has the direction
    p1 = torch.Tensor([0, b])
    # p2 = np.array([1,a+b])
    p2 = torch.Tensor([1, a + b])
    # p3 = np.array([-b/a,0])
    p3 = torch.Tensor([-b / a, 0])
    # p4 =np.array([(1-b)/a,1])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    # p = np.concatenate((p1,p2,p3,p4),axis=0)
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    if cos > 0:
        c, d = p[2]
    else:
        c, d = p[1]
    len = distance(x, y, c, d)
    # len = min(distance(x, y, c, d), distance(x, y, e, f))
    return len


def min_length(x, y, a, b):  # min length of the line y = ax+b with intersection with the bounding box of [0,1]
    p1 = torch.Tensor([0, b])
    # p2 = np.array([1,a+b])
    p2 = torch.Tensor([1, a + b])
    # p3 = np.array([-b/a,0])
    p3 = torch.Tensor([-b / a, 0])
    # p4 =np.array([(1-b)/a,1])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    # p = np.concatenate((p1,p2,p3,p4),axis=0)
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    c, d = p[1]
    e, f = p[2]
    return min(distance(x, y, c, d), distance(x, y, e, f))


def distance(x, y, a, b):
    # return np.sqrt(np.square(x-a)+np.square(y-b))
    # return torch.sqrt((x - a).pow(2) + (y - b).pow(2))
    return math.sqrt((x - a) * (x - a) + (y - b) * (y - b))


def toTensor(x):
    return torch.Tensor(x)



class PassiveHapticsEnv(object, radius = 0.5, random=False):
    def __init__(self):
        self.v_direction = 0
        self.p_direction = 0
        self.obj_x = 4.0
        self.obj_y = 4.0
        self.obj_d = PI / 4.0
        self.x_physical = 0
        self.y_physical = 0
        self.x_virtual = 0
        self.y_virtual = 0
        self.time_step = 0
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0

        self.path_cnt = 0
        #print(random)
        if not random:
            self.pas_path_file = np.load('./Dataset/new_passive_haptics_path.npy', allow_pickle=True)
            #print("Loading the /Dataset/passive_haptics_path.npy---------")
        else:
            self.pas_path_file = np.load('./Dataset/random_path.npy', allow_pickle=True)
            #print("Loading the /Dataset/random_path.npy---------")
        self.v_path = self.pas_path_file[self.path_cnt]
        self.v_step_pointer = 0  # the v_path counter

    def reset(self):
        self.v_path = self.pas_path_file[self.path_cnt]
        self.path_cnt += 1  # next v_path
        self.path_cnt = self.path_cnt % len(self.pas_path_file)
        self.v_direction = 0
        self.p_direction = 0
        self.v_step_pointer = 0

        self.time_step = 0
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0

        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter \
            = self.v_path[self.v_step_pointer]
        x, y, d = initialize(seed)
        self.x_physical = x
        self.y_physical = y
        self.p_direction = d

        return self.obs

    def step(self, action):
        gt, gr, gc = split(action)
        # print(gt.item(), gr.item(), gc.item())
        k = random.randint(5, 10)  # action repetition
        reward = torch.Tensor([0.0])
        for ep in range(k):  # for every iter, get the virtual path info, and steering
            self.vPathUpdate()
            signal = self.physical_step(gt, gr, gc)  # steering the physical env using the actions
            self.time_step += 1
            if (not signal) or (self.v_step_pointer == len(self.v_path) - 1):  # collision with the wall
                # reward += self.get_reward()
                if not signal:  # collide with the wall
                    reward -= PENALTY
                break
            elif ep == 0:
                tmp_r = self.get_reward()
            #     tmp_ = 0
            # else:
            #     reward += self.get_reward()
            reward += tmp_r
            # print(tmp_r)
        obs = self.current_obs[9:]
        obs.extend([(self.x_physical + DELTA_X) / WIDTH_ALL, (self.y_physical + DELTA_Y) / WIDTH_ALL,
                    (self.p_direction + PI) / (2 * PI),
                    self.x_virtual / WIDTH_ALL, self.y_virtual / HEIGHT_ALL, (self.v_direction + PI) / (2 * PI),
                    self.obj_x / WIDTH_ALL, self.obj_y / HEIGHT_ALL, (self.obj_d + PI) / (2 * PI)])
        self.current_obs = obs
        obs = toTensor(obs)
        ret_reward = reward
        self.reward += reward
        if not signal:  # reset the env
            bad_mask = 1
            r_reward = self.reward
            self.reset()
            return obs, ret_reward, [1], [bad_mask], r_reward
        elif signal and self.v_step_pointer >= len(self.v_path) - 1:
            r_reward = self.reward
            self.reset()
            return obs, ret_reward, [1], [0], r_reward
        else:
            return obs, ret_reward, [0], [0], ret_reward

    def vPathUpdate(self):
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = \
            self.v_path[self.v_step_pointer]  # unpack the next timestep virtual value
        self.v_step_pointer += 1

    def physical_step(self, gt, gr, gc):
        delta_curvature = gc * VELOCITY
        delta_rotation = self.delta_direction_per_iter / gr
        self.p_direction = norm(self.p_direction + delta_curvature + delta_rotation)
        delta_dis = VELOCITY / gt
        self.x_physical = self.x_physical + torch.cos(self.p_direction) * delta_dis
        self.y_physical = self.y_physical + torch.sin(self.p_direction) * delta_dis
        if outbound(self.x_physical, self.y_physical):
            return False
        else:
            return True
    """
    when in eval mode, initialize the user's postion
    """
    def init_eval_state(self, ind):
        m = ind % 4
        n = ind % 4 + 1
        ratio = WIDTH / WIDTH_ALL
        if abs(self.x_virtual) < 0.1:
            self.x_physical = 0
            self.y_physical = self.y_virtual * ratio
            self.p_direction = 0
        elif abs(self.y_virtual - HEIGHT_ALL) < 0.1:
            self.x_physical = self.x_virtual * ratio
            self.y_physical = HEIGHT
            self.p_direction = -PI/2
        elif abs(self.x_virtual - WIDTH_ALL) < 0.1:
            self.x_physical = WIDTH
            self.y_physical = self.y_virtual * ratio
            self.p_direction = -PI
        elif abs(self.y_virtual) < 0.1:
            self.x_physical = self.x_virtual * ratio
            self.y_physical = 0
            self.p_direction = PI / 2
        # self.p_direction = self.v_direction

    def step_specific_path(self, actor_critic, ind, ep=None):
        x_l = []
        y_l = []
        gt_l = []
        gr_l = []
        gc_l = []
        self.v_path = self.pas_path_file[ind]
        self.v_step_pointer = 0
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = self.v_path[
            self.v_step_pointer]
        self.init_eval_state(ind)
        # self.x_physical = 3.0
        # self.y_physical = 0.0
        # self.p_direction = PI / 2
        init_obs = []
        init_obs.extend(10 * [(self.x_physical + DELTA_X) / WIDTH_ALL, (self.y_physical + DELTA_Y) / WIDTH_ALL,
                              (self.p_direction + PI) / (2 * PI),
                              self.x_virtual / WIDTH_ALL, self.y_virtual / HEIGHT_ALL,
                              (self.v_direction + PI) / (2 * PI),
                              self.obj_x / WIDTH_ALL, self.obj_y / HEIGHT_ALL, (self.obj_d + PI) / (2 * PI)])
        self.current_obs = init_obs
        i = 0
        tmp_r = 0
        while i < len(self.v_path):
            # print(i)
            with torch.no_grad():
                value, action_mean, action_log_std = actor_critic.act(
                    torch.Tensor(self.current_obs).unsqueeze(0))
                dist = FixedNormal(action_mean, action_log_std)
                # action = action_mean
                action = dist.sample()
                action = action.clamp(-1.0, 1.0)
                # print("mean:", action_mean, '\tstd:', action_log_std, "\taction:", action)
            gt, gr, gc = split(action)
            gt_l.append(gt.item())
            gr_l.append(gr.item())
            gc_l.append(gc.item())
            for m in range(10):
                if i > len(self.v_path) - 1:
                    signal = False
                    # self.reward += self.get_reward()
                    break
                signal = self.physical_step(gt, gr, gc)
                self.vPathUpdate()
                x_l.append(self.x_physical)
                y_l.append(self.y_physical)
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    # print("srl_id:",ind,"penalty:",PENALTY)
                    # self.reward += self.get_reward()
                    break
                elif m == 0:
                    tmp_r = self.get_reward()
                self.reward += tmp_r
                # self.reward += tmp_r
            if not signal:
                break
            obs = self.current_obs[9:]
            obs.extend([(self.x_physical + DELTA_X) / WIDTH_ALL, (self.y_physical + DELTA_Y) / WIDTH_ALL,
                        (self.p_direction + PI) / (2 * PI),
                        self.x_virtual / WIDTH_ALL, self.y_virtual / HEIGHT_ALL, (self.v_direction + PI) / (2 * PI),
                        self.obj_x / WIDTH_ALL, self.obj_y / HEIGHT_ALL, (self.obj_d + PI) / (2 * PI)])
            self.current_obs = obs
        vx_l = self.v_path[:, 0]
        vy_l = self.v_path[:, 1]
        # self.reward = self.get_reward()
        final_dis = math.sqrt((self.x_physical + DELTA_X - self.obj_x) * (self.x_physical + DELTA_X - self.obj_x) +
                              (self.y_physical + DELTA_Y - self.obj_y) * (self.y_physical + DELTA_Y - self.obj_y))
        # print(self.reward)
        return self.reward, final_dis, self.err_angle(), gt_l, gr_l, gc_l, x_l, y_l, vx_l, vy_l

    def err_angle(self):
        return abs(delta_angle_norm(self.p_direction - self.v_direction))

       
    '''
    whether has a intersection with the obj
    '''
    def isIntersect():
        a = np.tan(self.v_direction)
        b = self.y_virtual - a * self.x_virtual
        # y = ax + b ----> ax - y + b = 0
        distance = abs(a * self.x_virtual - self.y_virtual + b)/ np.sqrt(a * a + 1) # distance from the centor to the stright line
        vec_x, vec_y = self.obj_x - self.x_virtual, self.obj_y - self.y_virtual # vec points the obj
        vec_x_, vec_y_ = np.cos(self.v_direction), np.sin(self.v_direction)   # vec of the v_direction
        result = vec_x * vec_x_ + vec_y * vec_y_
        if( result >= 0 && distance <= self.radius):
            return True
        else:
            return False
        


    '''
    scale the angle into 0-pi
    '''
def delta_angle_norm(x):
    if x >= PI:
        x = 2 * PI - x
    elif x <= -PI:
        x = x + 2 * PI
    return x
