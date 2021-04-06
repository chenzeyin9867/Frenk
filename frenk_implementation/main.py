import numpy as np
from env import *

def frenk_algorithm(file):
    testData = np.load(file, allow_pickle=True)
    distance = []
    delta_angle = []
    distance_none = []
    delta_angle_none = []
    for _ in range(100):
        # if _ % 10 == 0:
        #     print(_)
        print(10 * "*")
        passive_haptics_env = PassiveHapticsEnv(testData[_], _)
        x_l, y_l, x_v_flag, y_v_flag, x_p_flag, y_p_flag, tangent_x, tangent_y, dis, ang = \
            passive_haptics_env.eval()
        passive_haptics_env_none = PassiveHapticsEnv(testData[_], _)
        x_l_none, y_l_none, dis_none, ang_none = \
            passive_haptics_env_none.eval_none()
        distance.append(dis)
        delta_angle.append(ang)
        distance_none.append(dis_none)
        delta_angle_none.append(ang_none)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.axis([0.0, 8.0, 0.0, 8.0])
        plt.scatter(x_v_flag, y_v_flag, s=30, c='y')
        plt.scatter(testData[_][:, 0], testData[_][:, 1], s=1)
        plt.scatter(4,4,s=10,c='r')
        plt.subplot(1, 3, 2)
        plt.axis([0.0, 5.79, 0.0, 5.79])
        plt.scatter(x_l, y_l, s=1)
        plt.scatter(x_p_flag, y_p_flag, s=30, c='y')
        plt.scatter(WIDTH/2, WIDTH/2, s=10, c='r')
        plt.scatter(tangent_x, tangent_y, s=20, c='g')

        plt.subplot(1,3,3)
        plt.axis([0.0, WIDTH, 0.0, HEIGHT])
        plt.scatter(x_l_none, y_l_none, s=1)
        plt.scatter(WIDTH/2, WIDTH/2, s=20, c='r')

        plt.savefig('../result/result_' + str(_) + ".png")
        plt.cla()
        plt.close()
        del passive_haptics_env
        del passive_haptics_env_none
    mid_index = int(len(distance)/2)
    distance = sorted(distance)
    delta_angle = sorted(delta_angle)
    distance_none = sorted(distance_none)
    delta_angle_none = sorted(delta_angle_none)
    print("mean Distance:{:.2f}\t{:.2f}|{:.2f}|{:.2f}\tmean Angle:{:.2f}\t{:.2f}|{:.2f}|{:.2f}\t"
          .format(sum(distance)/(len(distance)), distance[0], distance[mid_index], distance[-1],
                  sum(delta_angle)/(len(delta_angle)), delta_angle[0], delta_angle[mid_index], delta_angle[-1]))
    print("mean Distance None:{:.2f}\t{:.2f}|{:.2f}|{:.2f}\tmean Angle:{:.2f}\t{:.2f}|{:.2f}|{:.2f}\t"
          .format(sum(distance_none)/(len(distance_none)), distance_none[0], distance_none[mid_index], distance_none[-1],
                  sum(delta_angle_none)/(len(delta_angle_none)), delta_angle_none[0], delta_angle_none[mid_index], delta_angle_none[-1]))

if __name__ == "__main__":
    pathFile = '../dataset/new_passive_haptics_path.npy'
    frenk_algorithm(pathFile)