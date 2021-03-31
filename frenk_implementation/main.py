import numpy as np
from env import *

def frenk_algorithm(file):
    testData = np.load(file, allow_pickle=True)
    for _ in range(100):
        if _ % 10 == 0:
            print(_)
        passive_haptics_env = PassiveHapticsEnv(testData[_])
        x_l, y_l = passive_haptics_env.eval()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.axis([0.0, 8.0, 0.0, 8.0])
        plt.scatter(testData[_][:, 0], testData[_][:, 1], s=1)
        plt.subplot(1, 2, 2)
        plt.axis([0.0, 5.79, 0.0, 5.79])
        plt.scatter(x_l, y_l, s=1)
        plt.savefig('../result/result_'+ str(_) + ".png")
        del passive_haptics_env

if __name__ == "__main__":
    pathFile = '/home/czy/MyWorkSpace/Frenk/dataset/new_passive_haptics_path.npy'
    frenk_algorithm(pathFile)