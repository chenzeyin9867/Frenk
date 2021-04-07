"""
This file is used to visualize the data generalization
"""
import numpy as np
import random
import matplotlib.pyplot as plt

VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 8.0, 8.0
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(1.5 / VELOCITY)

random.seed()


def outbound(x, y):
    if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT:
        return True
    else:
        return False


"""
four seeds represent 4 initial location
"""


def initialize(seed):
    xt = WIDTH / 2.0
    yt = HEIGHT / 2.0
    dt = PI / 4.0
    return xt, yt, dt


"""
normalize the theta into [-PI,PI]
"""


def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


if __name__ == '__main__':
    result = []
    for Epoch in range(3000):
        if Epoch % 10 == 0:
            print("Epoch:", Epoch)
        seed = random.randint(1, 4)
        x, y, d = initialize(seed)
        Xt = []
        Yt = []
        Dt = []
        Dchange = []
        Xt.append(x)
        Yt.append(y)
        Dt.append(norm(d+PI))
        Dchange.append(0)
        iter = 0
        delta_direction_per_iter = 0
        num_change_direction = 0
        turn_flag = 0
        for t in range(3600):
            if turn_flag == 0:
                turn_flag = np.random.randint(STEP_LOW, STEP_HIGH)
                delta_direction = random.normalvariate(0, 45)
                delta_direction = delta_direction * PI / 180.
                random_radius = random.random() + 1.0
                num_change_direction = abs(delta_direction * random_radius / VELOCITY)
                # print(delta_direction * 180 / PI, num_change_direction)
                delta_direction_per_iter = delta_direction / num_change_direction
                # print("delta_per:", delta_direction_per_iter)

            if num_change_direction > 0:
                d = norm(d + delta_direction_per_iter)

                num_change_direction = num_change_direction - 1
            else:
                turn_flag = turn_flag - 1
                delta_direction_per_iter = 0
            x = x + VELOCITY * np.cos(d)
            y = y + VELOCITY * np.sin(d)
            if outbound(x, y):
                break
            Xt.append(x)
            Yt.append(y)
            Dt.append(norm(d+PI))
            Dchange.append(-delta_direction_per_iter)
            # print(-delta_direction_per_iter)
        Xt = Xt[::-1]
        Yt = Yt[::-1]
        Dt = Dt[::-1]
        Dchange = Dchange[::-1]
            # print(D_t * 180 / np.pi)
            # plt.axis((0, 0, 20, 20))
        #     plt.scatter(X_t,Y_t,c = 'r',s = 0.1)
        #     plt.pause(0.1)
        # plt.show()
        Xt_np = np.array(Xt)
        Yt_np = np.array(Yt)
        Dt_np = np.array(Dt)
        Dchange_np = np.array(Dchange)
        stack_data = np.stack((Xt_np, Yt_np, Dt, Dchange_np), axis=-1)
        result.append(stack_data)

        plt.axis([0.0, WIDTH, 0.0, HEIGHT])
        dst = str(Epoch) + '.png'
        plt.scatter(Xt, Yt, c='r', s=1)
        plt.savefig("../Dataset/pas_haptics_path/" + dst)
        plt.clf()
        # plt.savefig("../Dataset/virtual_path_red/" + dst)

    save_np = np.array(result)
    np.save('../Dataset/new_passive_haptics_path.npy', save_np)
