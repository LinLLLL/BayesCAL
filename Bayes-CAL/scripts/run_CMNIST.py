import os
import numpy as np
import random


CSC = 'False' 

for seed in [3, 2, 1]:
    random.seed(seed)
    for hp_state in range(20):
        alpha1 = str(10 ** random.uniform(-3,0))[:6]
        alpha2 = str(10 ** random.uniform(-3,1))[:6]
        alpha3 = str(10 ** random.uniform(-3,0))[:6]
        CTP = 'end' # random.choice(['end', 'middle'])
        CSC = random.choice(['True', 'False'])
        os.system('CUDA_VISIBLE_DEVICES=0 bash main_cmnist.sh ColoredMNIST rn50_ep30 ' + ' ' + CTP + ' ' + CSC + ' ' + alpha1 + ' ' + alpha2 + ' ' + alpha3 + ' ' + str(seed))
