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


class PassiveHapticsEnv(object, radius=0.5, random=False):
    '''
    Frenk part params
    '''
    target_x = 0.0
    target_y = 0.0
    target_dir = 0.0

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
        # print(random)
        if not random:
            self.pas_path_file = np.load('./Dataset/new_passive_haptics_path.npy', allow_pickle=True)
            # print("Loading the /Dataset/passive_haptics_path.npy---------")
        else:
            self.pas_path_file = np.load('./Dataset/random_path.npy', allow_pickle=True)
            # print("Loading the /Dataset/random_path.npy---------")
        self.v_path = self.pas_path_file[self.path_cnt]
        self.v_step_pointer = 0  # the v_path counter

        # target_x = 0.0
        # target_y = 0.0

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
            self.p_direction = -PI / 2
        elif abs(self.x_virtual - WIDTH_ALL) < 0.1:
            self.x_physical = WIDTH
            self.y_physical = self.y_virtual * ratio
            self.p_direction = -PI
        elif abs(self.y_virtual) < 0.1:
            self.x_physical = self.x_virtual * ratio
            self.y_physical = 0
            self.p_direction = PI / 2
        # self.p_direction = self.v_direction

    def err_angle(self):
        return abs(delta_angle_norm(self.p_direction - self.v_direction))

    '''
    whether has a intersection with the obj
    '''

    def isIntersect(self):
        a = np.tan(self.v_direction)
        b = self.y_virtual - a * self.x_virtual
        # y = ax + b ----> ax - y + b = 0
        distance = abs(a * self.x_virtual - self.y_virtual + b) / np.sqrt(
            a * a + 1)  # distance from the centor to the stright line
        vec_x, vec_y = self.obj_x - self.x_virtual, self.obj_y - self.y_virtual  # vec points the obj
        vec_x_, vec_y_ = np.cos(self.v_direction), np.sin(self.v_direction)  # vec of the v_direction
        result = vec_x * vec_x_ + vec_y * vec_y_
        if result >= 0 and distance <= self.radius:
            return True
        else:
            return False

    def calculatePath(self):
        currentDir = np.array([np.cos(self.p_direction), np.sin(self.p_direction)])  # physical direction
        targetDir = np.array([np.cos(self.target_dir), np.sin(self.target_dir)])
        currentPos = np.array([self.x_physical, self.y_physical])
        x1 = self.x_physical
        y1 = self.y_physical
        x2 = self.target_x
        y2 = self.target_y
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
            currentOrthoDir = np.array([-np.sin(self.p_direction, np.cos(self.p_direction))])
            if np.dot(targetDir, currentOrthoDir) < 0:
                currentOrthoDir = - currentOrthoDir
            targetOrthoDir  = np.array([-np.sin(self.target_dir), np.cos(self.target_dir)])
            if np.dot(targetOrthoDir, currentDir) < 0:
                targetOrthoDir = - targetOrthoDir
            u1 = currentOrthoDir[0]
            v1 = currentOrthoDir[1] # (u1,v1)为(s1,t1)法向量
            u2 = targetOrthoDir[0]
            v2 = targetOrthoDir[1] # (u2,v2)为(s2,t2)法向量
            
            # r = abs(((x2 - x1) * u2 + (y2 - y1) * v2) / (1 + u1 * u2 + v1 * v2)) # radius of the tangent circle
            r = ((x2 - x1) * u2 + (y2 - y1) * v2) / (1 + u1 * u2 + v1 * v2) # 求出半径
            tangentPos = currentPos + (currentOrthoDir * r + r * targetOrthoDir)  # 为a中大圆弧切点的位置
            # if  (targetPos - tangentPos).x / targetDir.x > 0:      # a情况
            if (targetPos - tangentPos)[0] / targetDir[0] > 0:
                middlePos = tangentPos 
            else: # b: (x1, y1) + p(s1, t1) + r(u1, v1) + r(u2, v2) = (x2, y2)
                d = s1 * (v1 + v2) - t1 * (u1 + u2) #  利用cramer法则，算行列式
                d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2)
                d2 = s1 * (y2 - y1) - t1 * (x2 - x1)
                p = d1 / d
                r = d2 / d
                middlePos = currentPos + p * currentDir
            
        elif m < 0 and  n < 0: # situation c
                
                // 要解非线性方程组才能算出r1, r2，这也太难了！！
                // 两条圆弧的两个半径非常难求解，我想不出办法了
                
        else: # d情况
            # pathType = PathType.d
            targetOrthoDir = np.array([-np.sin(self.target_dir), np.cos(self.target_dir)])
            if np.dot(targetOrthoDir, currentDir) > 0:
                targetOrthoDir = - targetOrthoDir
            currentOrthoDir = np.array([-np.sin(self.p_direction, np.cos(self.p_direction))])
            if np.dot(targetDir, currentOrthoDir) > 0:
                currentOrthoDir = - currentOrthoDir

            u1 = currentOrthoDir[0]
            v1 = currentOrthoDir[1]
            u2 = targetOrthoDir[0]
            v2 = targetOrthoDir[1]

            # (x1, y1) + p(s1, t1) + r(u1, v1) + r(u2, v2) = (x2, y2)
            可解出p, r
            float d = s1 * (v1 + v2) - t1 * (u1 + u2); // 利用cramer法则，算行列式
            float d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2);
            float d2 = s1 * (y2 - y1) - t1 * (x2 - x1);
            float
            p = d1 / d;
            r = d2 / d;
            middlePos = currentPos + p * currentDir;
                
                # 不考虑e情况，因为可以把e情况当a情况考虑，虽然可能会有圆弧的曲率过大，但是这篇论文算法本身就不能保证所有路径曲率在max范围内

    }
    public
    override
    void
    ApplyRedirection()
    {
    if (isRedirecting == false | | Utilities.Approximately(redirectionManager.currPosReal, targetPos))
    findTarget();
    if (isRedirecting == true)
    {
    if (Utilities.Approximately(Utilities.FlattenedPos2D(redirectionManager.currPosReal), middlePos))


{
    passMiddlePoint = true;
}
// Get
Required
Data
Vector3
deltaPos = redirectionManager.deltaPos;
float
deltaDir = redirectionManager.deltaDir;
switch(pathType) // 判断路径类型
{
    case
PathType.a:
{
if (!passMiddlePoint)
{
if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) // User is moving
{
// 走过的角度
rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
}
}
break;
}
case
PathType.b:
{
if (passMiddlePoint)
    {
    if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) // User is moving
    {
    // 走过的角度
    rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
    }
    }
    break;
}
case
PathType.c:
{
// // / c情况如上所说，两条圆弧半径非常难算，待解决此问题
// /
break;

}
case
PathType.d:
{
if (passMiddlePoint)
    {
    if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) // User is moving
    {
    // 走过的角度
    rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
    }
    }
    break;
    }
    }
    }
    InjectCurvature(rotationFromCurvatureGain);
    }
    }


    '''
    scale the angle into 0-pi
    '''


    def delta_angle_norm(x):
        if x >= PI:
            x = 2 * PI - x
        elif x <= -PI:
            x = x + 2 * PI
        return x


    class Steinickef_Redirector(object):
        # User Experience Improvement Parameters
        MOVEMENT_THRESHOLD = 0.2
        f  # meters per second
        ROTATION_THRESHOLD = 1.5
        f  # degrees per second

        rotationFromCurvatureGain = 0.0  # Proposed curvature gain based on user speed
        rotationFromRotationGain = 0.0  # Proposed rotation gain based on head's yaw
        middlePos = 0.0  # 表示目前确定了目标点的位置和方向以后，路径的分割点（分割线，圆弧的一些特殊点）
        targetPos = 0.0  # 表示目前确定的目标点的物理位置
        targetDir  # 表示目标点的物理方向
        isRedirecting = false  # 正在重定向过程中
        private
        float
        r, r1, r2; // 圆弧的半径参数，其中r针对路径中只有一段圆弧情况，r1, r2针对路径中有两段圆弧情况
        # private enum PathType  { a,b,c,d,e};
        # private PathType pathType;
        passMiddlePoint = false  # whether passed the mod point

        public
        void
        findTarget()
        {
            isRedirecting = false;
        // .............这里待解决
           // 根据论文，如果当前的方向能够与虚拟物品重叠，则找到了target就是这个虚拟物品
                                    // 论文里说是从虚拟物品和物理物品的注册关系得到物理的targetPos跟targetDir这两个属性
                                    // 所以这里还需要做的就是确定targetPos跟targetDir这两个属性
                                    // 暂时还没考虑路径中存在障碍物的情况，也就是论文中的4
        .3
        内容
        // ..............
        isRedirecting = true;
        calculatePath();
        }

        def calculatePath():
            curDirection = np.array()
            Vector2
            currentPos = Utilities.FlattenedPos2D(redirectionManager.currPosReal);
            float
            x1 = currentPos.x, y1 = currentPos.y; // (x1, y1) + m * (s1, t1) = (x2, y2) + n * (s2, t2)
            float
            x2 = targetPos.x, y2 = targetPos.y;
            float
            s1 = currentDir.x, t1 = currentDir.y; // (s1, t1)
            是当前方向单位向量
            float
            s2 = targetDir.x, t2 = targetDir.y;
            float
            D = s1 * (-t2) + s2 * t1; // 用cramer法则，用行列式算m, n
            float
            D1 = (x2 - x1) * (-t2) + s2 * (y2 - y1);
            float
            D2 = s1 * (y2 - y1) - t1 * (x2 - x1);
            float
            m = D1 / D;
            float
            n = D2 / D;

            if (Vector2.Dot(currentDir, targetDir) > 0 & & m > 0 & & n < 0) // a, b情况
            {
                Vector2
            targetOrthoDir = new
            Vector2(-targetDir.y, targetDir.x).normalized; // 目标点的法向量，方向可能有两个
            targetOrthoDir = Mathf.Sign(Vector2.Dot(targetOrthoDir, currentDir)) * targetOrthoDir; // 确定目标点的法向量方向
            Vector2
            currentOrthoDir = new
            Vector2(-currentDir.y, currentDir.x).normalized; // 当前方向的法向量
            currentOrthoDir = Mathf.Sign(Vector2.Dot(currentOrthoDir, targetDir)) * currentOrthoDir; // 法向量有两个，选对的那个
            float
            u1 = currentOrthoDir.x, v1 = currentOrthoDir.y; // (u1, v1)
            为(s1, t1)
            法向量
            float
            u2 = targetOrthoDir.x, v2 = targetOrthoDir.y; // (u2, v2)
            为(s2, t2)
            法向量
            // (x2 - x1 - u1 * r, y2 - y1 - v1 * r)·(u2, v2) = r
            r = ((x2 - x1) * u2 + (y2 - y1) * v2) / (1 + u1 * u2 + v1 * v2); // 求出半径
            Vector2
            tangentPos = currentPos + (currentOrthoDir * (float)r) + r * targetOrthoDir; // 为a中大圆弧切点的位置
            if ((targetPos - tangentPos).x / targetDir.x > 0) // a情况
            {
                pathType = PathType.a;
            middlePos = tangentPos;
            }
            else // b情况
            {
                pathType = PathType.b;
            // (x1, y1) + p(s1, t1) + r(u1, v1) + r(u2, v2) = (x2, y2)
            可解出p, r
            float
            d = s1 * (v1 + v2) - t1 * (u1 + u2); // 利用cramer法则，算行列式
            float
            d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2);
            float
            d2 = s1 * (y2 - y1) - t1 * (x2 - x1);
            float
            p = d1 / d;
            r = d2 / d;
            middlePos = currentPos + p * currentDir;
            }
            } else if (m < 0 & & n < 0) // c情况
                {
                // 要解非线性方程组才能算出r1, r2，这也太难了！！
                // 两条圆弧的两个半径非常难求解，我想不出办法了
                }
                else // d情况
                {
                    pathType = PathType.d;
                Vector2
                targetOrthoDir = new
                Vector2(-targetDir.y, targetDir.x).normalized; // 目标点的法单位向量，方向可能有两个
                targetOrthoDir = -Mathf.Sign(Vector2.Dot(targetOrthoDir, currentDir)) * targetOrthoDir; // 确定目标点的法向量方向
                Vector2
                currentOrthoDir = new
                Vector2(-currentDir.y, currentDir.x).normalized; // 当前方向的法向量
                currentOrthoDir = -Mathf.Sign(
                    Vector2.Dot(currentOrthoDir, targetDir)) * currentOrthoDir; // 法向量有两个，选对的那个
                float
                u1 = currentOrthoDir.x, v1 = currentOrthoDir.y; // (u1, v1)
                为(s1, t1)
                法向量
                float
                u2 = targetOrthoDir.x, v2 = targetOrthoDir.y; // (u2, v2)
                为(s2, t2)
                法向量
                // (x1, y1) + p(s1, t1) + r(u1, v1) + r(u2, v2) = (x2, y2)
                可解出p, r
                float
                d = s1 * (v1 + v2) - t1 * (u1 + u2); // 利用cramer法则，算行列式
                float
                d1 = (x2 - x1) * (v1 + v2) - (y2 - y1) * (u1 + u2);
                float
                d2 = s1 * (y2 - y1) - t1 * (x2 - x1);
                float
                p = d1 / d;
                r = d2 / d;
                middlePos = currentPos + p * currentDir;
                }
                // 不考虑e情况，因为可以把e情况当a情况考虑，虽然可能会有圆弧的曲率过大，但是这篇论文算法本身就不能保证所有路径曲率在max范围内

            }
            public
            override
            void
            ApplyRedirection()
            {
            if (isRedirecting == false | | Utilities.Approximately(redirectionManager.currPosReal, targetPos))
                findTarget();
            if (isRedirecting == true)
                {
                if (Utilities.Approximately(Utilities.FlattenedPos2D(redirectionManager.currPosReal), middlePos))
                {
                    passMiddlePoint = true;
                }
                // Get
                Required
                Data
                Vector3
                deltaPos = redirectionManager.deltaPos;
                float
                deltaDir = redirectionManager.deltaDir;
                switch(pathType) // 判断路径类型
                {
                    case
                PathType.a:
                {
                if (!passMiddlePoint)
                {
                if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) // User is moving
                {
                // 走过的角度
                rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
                }
                }
                break;
                }
                case
                PathType.b:
                {
                if (passMiddlePoint)
                    {
                    if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) // User is moving
                    {
                    // 走过的角度
                    rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
                    }
                    }
                    break;
                }
                case
                PathType.c:
                {
                // // / c情况如上所说，两条圆弧半径非常难算，待解决此问题
                // /
                break;

            }
            case
            PathType.d:
            {
            if (passMiddlePoint)
                {
                if (deltaPos.magnitude / redirectionManager.GetDeltaTime() > MOVEMENT_THRESHOLD) // User is moving
                {
                // 走过的角度
                rotationFromCurvatureGain = Mathf.Rad2Deg * (deltaPos.magnitude / r);
                }
                }
                break;
                }
                }
                }
                InjectCurvature(rotationFromCurvatureGain);
                }
                }
