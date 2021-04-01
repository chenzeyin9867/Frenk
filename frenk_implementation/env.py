import os
import numpy as np
import torch
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


class PassiveHapticsEnv(object):
    '''
    Frenk part params
    '''
    objRadius = 0.1
    action_repeat = 10
    target_x = 0.0
    target_y = 0.0
    target_dir = -3 * PI/4
    pathType = 'i'  # initial state
    currentPos = np.array([0.0, 0.0])
    targetPos = np.array([0.0, 0.0])
    currentDir = np.array([0.0, 0.0])
    targetDir = np.array([0.0, 0.0])
    radius = 0.0
    tangentPos = np.array([0.0, 0.0])
    passedTangent = False
    gt = 1.0
    gc = 0.0
    gc_part1 = 0.0  # in this paper, we only take gt and gc into consideration
    gc_part2 = 0.0
    firstPartCnt = 0

    def __init__(self, path, radius=0.5, random=False):
        self.v_direction = 0
        self.p_direction = 0
        self.obj_x_v = 4.0
        self.obj_y_v = 4.0
        self.obj_x_p = WIDTH / 2
        self.obj_y_p = WIDTH / 2
        self.obj_d = PI / 4.0
        self.x_physical = 0
        self.y_physical = 0
        self.x_virtual = 0
        self.y_virtual = 0
        self.time_step = 0
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0

        self.path_cnt = 0
        # print(random)
        self.v_path = path
        self.v_step_pointer = 0  # the v_path counter

        # target_x = 0.0
        # target_y = 0.0

    '''
    func to eval a path
    '''
    def eval(self):
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = self.v_path[
            self.v_step_pointer]
        self.init_eval_state()
        x_l = []
        y_l = []
        cnt = 0
        signal = True
        while cnt < len(self.v_path):
            if not self.isIntersect():
                self.gc = 0.0
                self.gt = 1.0
                if cnt == len(self.v_path) - 1:
                    signal = False
                    break
                signal = self.step()
                x_l.append(self.x_physical)
                y_l.append(self.y_physical)
                cnt = cnt + 1
                if not signal:
                    break
            else:
                print("intersect")
                self.calculatePath()
                self.ApplyRedirection()
                self.gc = self.gc1
                print(self.gt, self.gc)
                print(self.pathType)
                # print(cnt, len(self.v_path))
                while self.isIntersect():
                    if cnt == len(self.v_path) - 1:
                        signal = False
                        break
                    signal = self.step()
                    self.firstPartCnt = self.firstPartCnt - 1
                    if self.firstPartCnt <= 0:
                        print("*******here*******")
                        self.gc = self.gc2
                    x_l.append(self.x_physical)
                    y_l.append(self.y_physical)
                    cnt = cnt + 1
                    if not signal:
                        break
            if not signal:
                break
        return x_l, y_l

    def init_eval_state(self):
        # ratio = WIDTH / WIDTH_ALL
        # if abs(self.x_virtual) < 0.1:
        #     self.x_physical = 0
        #     self.y_physical = self.y_virtual * ratio
        #     self.p_direction = 0
        # elif abs(self.y_virtual - HEIGHT_ALL) < 0.1:
        #     self.x_physical = self.x_virtual * ratio
        #     self.y_physical = HEIGHT
        #     self.p_direction = -PI / 2
        # elif abs(self.x_virtual - WIDTH_ALL) < 0.1:
        #     self.x_physical = WIDTH
        #     self.y_physical = self.y_virtual * ratio
        #     self.p_direction = -PI
        # elif abs(self.y_virtual) < 0.1:
        #     self.x_physical = self.x_virtual * ratio
        #     self.y_physical = 0
        #     self.p_direction = PI / 2
        # self.p_direction = self.v_direction
        m = np.random.randint(0, 4)
        n = np.random.random()
        if m == 0:
            self.x_physical = 0
            self.y_physical = HEIGHT * n
            self.p_direction = 0
        elif m == 1:
            self.x_physical = WIDTH * n
            self.y_physical = HEIGHT
            self.p_direction = -PI / 2
        elif m == 2:
            self.x_physical = WIDTH
            self.y_physical = HEIGHT * n
            self.p_direction = -PI
        elif m == 3:
            self.x_physical = WIDTH * n
            self.y_physical = 0
            self.p_direction = PI / 2

    def vPathUpdate(self):
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = \
            self.v_path[self.v_step_pointer]  # unpack the next timestep virtual value
        self.v_step_pointer += 1

    def physical_step(self):
        delta_curvature = self.gc * VELOCITY
        delta_rotation = self.delta_direction_per_iter
        # print(delta_rotation)
        self.p_direction = norm(self.p_direction + delta_curvature + delta_rotation)
        delta_dis = VELOCITY / self.gt
        self.x_physical = self.x_physical + np.cos(self.p_direction) * delta_dis
        self.y_physical = self.y_physical + np.sin(self.p_direction) * delta_dis
        if outbound(self.x_physical, self.y_physical):
            return False
        else:
            return True

    def step(self):
        self.vPathUpdate()                                                               
        signal = self.physical_step()  # steering the physical env using the actions
        return signal



    def err_angle(self):
        return abs(delta_angle_norm(self.p_direction - self.v_direction))

    '''
    whether has a intersection with the obj
    '''

    def isIntersect(self):
        a = np.tan(self.v_direction)
        b = self.y_virtual - a * self.x_virtual
        # y = ax + b ----> ax - y + b = 0
        distance = abs(a * self.obj_x_v - self.obj_y_v + b) / np.sqrt(
            a * a + 1)  # distance from the centor to the stright line
        print(distance)
        vec_x, vec_y = self.obj_x_v - self.x_virtual, self.obj_y_v - self.y_virtual  # vec points the obj
        vec_x_, vec_y_ = np.cos(self.v_direction), np.sin(self.v_direction)  # vec of the v_direction
        result = vec_x * vec_x_ + vec_y * vec_y_
        if result >= 0 and distance <= self.objRadius:
            return True
        else:
            return False

    """
    compute the length of current physical path in order to compute the in-time translation gain
    """                                                

    def getLengthOfPhysicalPath(self):
        if self.pathType == 'a':
            deltaAngle = abs(delta_angle_norm(self.target_dir - self.p_direction))
            circleLength = self.radius * deltaAngle
            self.firstPartCnt = circleLength / VELOCITY #the cnt of the circle
            lineSegmentLength = distance(self.tangentPos[0], self.tangentPos[1], self.targetPos[0], self.targetPos[1])
            return lineSegmentLength + circleLength
        elif self.pathType == 'b' or self.pathType == 'd':
            deltaAngle = abs(delta_angle_norm(self.target_dir - self.p_direction))
            circleLength = self.radius * deltaAngle
            lineSegmentLength = distance(self.currentPos[0], self.currentPos[1], self.tangentPos[0], self.tangentPos[1])
            self.firstPartCnt = lineSegmentLength / VELOCITY
            return lineSegmentLength + circleLength
        elif self.pathType == 'c':  # 90 degree circle and second part circle
            # part2Angle = abs(np.arctan(self.targetDir[1] / self.targetDir[0]))
            part2Angle = PI/2 - abs(delta_angle_norm(self.p_direction - self.target_dir))
            # print("angle", part2Angle * 180 /PI)
            self.firstPartCnt = (PI/2) * self.radius / VELOCITY
            return (PI / 2 + part2Angle) * self.radius
        elif self.pathType == 'f':
            part2Angle = PI / 2 + abs(delta_angle_norm(self.p_direction - self.target_dir))
            # print("angle", part2Angle * 180 /PI)
            self.firstPartCnt = (PI / 2) * self.radius / VELOCITY
            return (PI / 2 + part2Angle) * self.radius

    def getLengthOfVirtualPath(self):
        return distance(self.x_virtual, self.y_virtual, self.obj_x_v, self.obj_y_v)

    def calculatePath(self):
        self.currentPos = np.array([self.x_physical, self.y_physical])
        self.currentDir = np.array([np.cos(self.p_direction), np.sin(self.p_direction)])  # physical direction
        self.targetPos = np.array([self.obj_x_p, self.obj_y_p])
        self.targetDir = np.array([np.cos(self.target_dir), np.sin(self.target_dir)])
        x1 = self.x_physical
        y1 = self.y_physical
        x2 = self.targetPos[0]
        y2 = self.targetPos[1]
        s1 = np.cos(self.p_direction)
        t1 = np.sin(self.p_direction)
        s2 = np.cos(self.target_dir)
        t2 = np.sin(self.target_dir)

        d = s1 * (-t2) + s2 * t1  # cramer rules, compute m, n
        d1 = (x2 - x1) * (-t2) + s2 * (y2 - y1)
        d2 = s1 * (y2 - y1) - t1 * (x2 - x1)
        m = d1 / d
        n = d2 / d

        if s1 * s2 + t1 * t2 > 0 and m > 0 and n < 0:  # situation a or b
            currentOrthoDir = np.array([-np.sin(self.p_direction), np.cos(self.p_direction)])
            if np.dot(self.targetDir, currentOrthoDir) < 0:
                currentOrthoDir = - currentOrthoDir
            targetOrthoDir = np.array([-np.sin(self.target_dir), np.cos(self.target_dir)])
            if np.dot(targetOrthoDir, self.currentDir) < 0:
                targetOrthoDir = - targetOrthoDir
            u1 = currentOrthoDir[0]
            v1 = currentOrthoDir[1]  # (u1,v1)为(s1,t1)法向量
            u2 = targetOrthoDir[0]
            v2 = targetOrthoDir[1]  # (u2,v2)为(s2,t2)法向量

            # r = abs(((x2 - x1) * u2 + (y2 - y1) * v2) / (1 + u1 * u2 + v1 * v2)) # radius of the tangent circle
            self.radius = ((x2 - x1) * u2 + (y2 - y1) * v2) / (1 + u1 * u2 + v1 * v2)  # 求出半径
            tangentPos = self.currentPos + (currentOrthoDir * self.radius + self.radius * targetOrthoDir)  # 为a中大圆弧切点的位置
            # if  (targetPos - tangentPos).x / targetDir.x > 0:      # a情况
            if (x2 - tangentPos[0]) / self.targetDir[0] > 0:
                self.pathType = 'a'
                self.tangentPos = tangentPos
            else:  # b: (x1, y1) + p(s1, t1) + r(u1, v1) + r(u2, v2) = (x2, y2)
                d = s1 * (v1 + v2) - t1 * (u1 + u2)  # 利用cramer法则，算行列式
                d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2)
                d2 = s1 * (y2 - y1) - t1 * (x2 - x1)
                p = d1 / d
                self.radius = d2 / d
                self.tangentPos = self.currentPos + p * self.currentDir
                self.pathType = 'b'

        elif m < 0 and n < 0:  # situation c  假设两个圆弧半径相等
            self.pathType = 'c'
            targetOrthoDir = np.array([-np.sin(self.target_dir), np.cos(self.target_dir)])
            if np.dot(targetOrthoDir, self.currentDir) < 0:
                targetOrthoDir = - targetOrthoDir
            currentOrthoDir = np.array([-np.sin(self.p_direction), np.cos(self.p_direction)])
            if np.dot(self.targetDir, currentOrthoDir) < 0:
                currentOrthoDir = - currentOrthoDir
            u1 = currentOrthoDir[0]
            v1 = currentOrthoDir[1]
            u2 = targetOrthoDir[0]
            v2 = targetOrthoDir[1]

            # (x2 + ru2 - x1 - ru1, y2 + rv2 - y1 - rv1) = 2r
            a = np.power(u2 - u1, 2) + np.power(v2 - v1, 2) - 4
            b = 2 * (x2 - x1) * (u2 - u1) + 2 * (v2 - v1) * (y2 - y1)
            c = np.power(x2 - x1, 2) + np.power(y2 - y1, 2)
            r = solveEquation(a, b, c)
            self.radius = r
            self.tangentPos = (self.currentPos + r * currentOrthoDir + self.targetPos + targetOrthoDir * r) / 2
        elif m > 0 and n > 0: # not mention on the paper, but similary with c
            self.pathType = 'f'
            targetOrthoDir = np.array([-np.sin(self.target_dir), np.cos(self.target_dir)])
            if np.dot(targetOrthoDir, self.currentDir) > 0:
                targetOrthoDir = - targetOrthoDir
            currentOrthoDir = np.array([-np.sin(self.p_direction), np.cos(self.p_direction)])
            if np.dot(self.targetDir, currentOrthoDir) > 0:
                currentOrthoDir = - currentOrthoDir
            u1 = currentOrthoDir[0]
            v1 = currentOrthoDir[1]
            u2 = targetOrthoDir[0]
            v2 = targetOrthoDir[1]

            # (x2 + ru2 - x1 - ru1, y2 + rv2 - y1 - rv1) = 2r
            a = np.power(u2 - u1, 2) + np.power(v2 - v1, 2) - 4
            b = 2 * (x2 - x1) * (u2 - u1) + 2 * (v2 - v1) * (y2 - y1)
            c = np.power(x2 - x1, 2) + np.power(y2 - y1, 2)
            r = solveEquation(a, b, c)
            self.radius = r
            self.tangentPos = (self.currentPos + r * currentOrthoDir + self.targetPos + targetOrthoDir * r) / 2

            # print("Special situation!****************" * 10)
        else:  # d情况
            # print(m , n)
            self.pathType = 'd'
            targetOrthoDir = np.array([-np.sin(self.target_dir), np.cos(self.target_dir)])
            if np.dot(targetOrthoDir, self.currentDir) > 0:
                targetOrthoDir = - targetOrthoDir
            currentOrthoDir = np.array([-np.sin(self.p_direction), np.cos(self.p_direction)])
            if np.dot(self.targetDir, currentOrthoDir) > 0:
                currentOrthoDir = - currentOrthoDir

            u1 = currentOrthoDir[0]
            v1 = currentOrthoDir[1]
            u2 = targetOrthoDir[0]
            v2 = targetOrthoDir[1]

            # (x1, y1) + p(s1, t1) + r(u1, v1) + r(u2, v2) = (x2, y2)
            d = s1 * (v1 + v2) - t1 * (u1 + u2)  # 利用cramer法则，算行列式
            d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2)
            d2 = s1 * (y2 - y1) - t1 * (x2 - x1)
            p = d1 / d
            self.radius = d2 / d
            self.tangentPos = self.currentPos + p * self.currentDir
        # 不考虑e情况，因为可以把e情况当a情况考虑，虽然可能会有圆弧的曲率过大，但是这篇论文算法本身就不能保证所有路径曲率在max范围内

    '''
    compute the gt and gc based on different situation
    '''

    def ApplyRedirection(self):
        virtualPathLength = self.getLengthOfVirtualPath()
        physicalPathLength = self.getLengthOfPhysicalPath()
        gt = virtualPathLength / physicalPathLength
        # gt = 1.0
        gc1 = 0.0
        gc2 = 0.0
        if self.pathType == 'a':  # situation a
            gc1 = 1.0 / self.radius
            if np.cross(self.targetDir, self.currentDir) > 0:
                gc1 = -gc1
        elif self.pathType == 'b':  # situation b
            gc1 = 0.0
            gc2 = 1.0 / self.radius
            if np.cross(self.targetDir, self.currentDir) > 0:
                gc2 = -gc2
        elif self.pathType == 'c':  # situation c, gc minus when passed the tangent point
            gc = 1.0 / self.radius
            if np.cross(self.currentDir, self.targetDir) < 0:
                gc1 = -gc
                gc2 = gc
            elif np.cross(self.currentDir, self.targetDir) > 0:
                gc1 = gc
                gc2 = -gc
        elif self.pathType == 'f':
            gc = 1.0 / self.radius
            if np.cross(self.currentDir, self.targetDir) > 0:
                gc1 = -gc
                gc2 = gc
            elif np.cross(self.currentDir, self.targetDir) < 0:
                gc1 = gc
                gc2 = -gc
        else:  # d situation
            gc1 = 0.0
            gc2 = 1.0 / self.radius
            if np.cross(self.currentDir, self.targetDir) > 0:
                gc2 = - gc2

        # gc1 = np.clip(gc1, -0.13, 0.13)
        # gc2 = np.clip(gc2, -0.13, 0.13)
        # gt = np.clip(gt, 0.8, 1.26)
        self.gt = gt
        self.gc1 = gc1
        self.gc2 = gc2

    def print(self):
        print("virtual:", self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter)
        print("physical:", self.x_physical, self.y_physical, self.p_direction)
'''
scale the angle into 0-pi
'''


def delta_angle_norm(x):
    if x >= PI:
        x = 2 * PI - x
    elif x <= -PI:
        x = x + 2 * PI
    return x


def solveEquation(a, b, c):
    delta = b * b - 4 * a * c
    if ((-b + np.sqrt(delta)) / (2 * a)) > 0:
        return (-b + np.sqrt(delta)) / (2 * a)
    return (-b - np.sqrt(delta)) / (2 * a)
    # return (-b + np.sqrt(delta)) / (2 * a) > 0
