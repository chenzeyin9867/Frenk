import numpy as np
from env import *

def frenk_algorithm(file):
    testData = np.load(file, allow_pickle=True)
    for _ in range(100):
        # if _ % 10 == 0:
        #     print(_)
        passive_haptics_env = PassiveHapticsEnv(testData[_], _)
        x_l, y_l, x_v_flag, y_v_flag, x_p_flag, y_p_flag, tangent_x, tangent_y = passive_haptics_env.eval()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.axis([0.0, 8.0, 0.0, 8.0])
        plt.scatter(x_v_flag, y_v_flag, s=30, c='y')
        plt.scatter(testData[_][:, 0], testData[_][:, 1], s=1)
        plt.scatter(4,4,s=10,c='r')
        plt.subplot(1, 2, 2)
        plt.axis([0.0, 5.79, 0.0, 5.79])
        plt.scatter(x_l, y_l, s=1)
        plt.scatter(x_p_flag, y_p_flag, s=30, c='y')
        plt.scatter(WIDTH/2, WIDTH/2, s=10, c='r')
        plt.scatter(tangent_x, tangent_y, s=20, c='g')
        plt.savefig('../result/result_' + str(_) + ".png")
        plt.cla()
        plt.close()
        del passive_haptics_env

if __name__ == "__main__":
    pathFile = '../dataset/new_passive_haptics_path.npy'
    frenk_algorithm(pathFile)