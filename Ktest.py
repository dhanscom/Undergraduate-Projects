import numpy as np
import scipy.stats as sp
import csv
import os.path
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn import preprocessing

def makeMatrix(path, folder="."):
    with open(folder + "/" + path, 'r') as L:
        reader = csv.reader(L, delimiter=',')
        matrix = np.array(list(reader)).astype(float)
    return matrix

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

input_folder = "input_data_Stationary"
input_files = os.listdir(input_folder)

results_folder = "Results"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
else:
    for file in os.listdir(results_folder):
        os.remove(os.path.join(results_folder, file))

file = open("Results/Ktest", "a")
file.write("Event - Station - Band - statistic - pvalue - Null Hypothesis" + "\n")
file.close()

x = chi2.rvs(size=7201, df=2)
for input_file in input_files:
    created_matrix = makeMatrix(input_file, input_folder)
    file = open("Results/Ktest", "a")
    for count, row_index in enumerate(created_matrix):
        statistic, p = sp.kstest(row_index, chi2.cdf(x,2))
        YN = 'reject'
        if p > statistic:
            YN = 'accept'
        file.write(
            input_file.split()[1] + "        " + input_file.split()[2].split(".")[0] +
            "       " + str(count + 1) + "    " + str(statistic) + " " + str(p) + " " + YN + "\n")
file.close()
