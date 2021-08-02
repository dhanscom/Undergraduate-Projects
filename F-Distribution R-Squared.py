import os, os.path
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm
import scipy as sci
import random as rand
import numpy as np
import math as math
from scipy.stats import f, norm

numBins = 500


def DistroFit(dfn, dfd, distroType, intervalStart, intervalEnd, n, bins, numBins):
    x = np.linspace(intervalStart, intervalEnd, numBins)
    try:
        coefs, coVar = sci.optimize.curve_fit(lambda t, a, b: a * f.pdf(b * t, dfn, dfd), x, n, maxfev=1000)

        fitPoints = coefs[0] * f.pdf(coefs[1] * x, dfn, dfd)
        ssr = np.sum((n - fitPoints) ** 2)
        sst = np.sum((n - np.average(fitPoints)) ** 2)

        R2 = 1 - (ssr / sst)

        if R2 < 0:
            R2 = -1
    except:
        R2 = -1

    return R2







def makeMatrix(path, folder="."):
    with open(folder + "/" + path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        matrix = np.array(list(reader)).astype(float)
    return matrix


input_folder = "Paul_Data"
input_files = os.listdir(input_folder)

results_folder = "Results"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
else:
    for file in os.listdir(results_folder):
        os.remove(os.path.join(results_folder, file))



file = open("F-FitTestOutput", "a")
file.write("Event - Station - Band - R2 | Number of Bins: " + str(numBins) + " 90%  95%  99%" + "\n")
file.close()

countTotal = 0
countPass90 = 0
countPass95 = 0
countPass99 = 0
yn90 = ''
yn95 = ''
yn99 = ''

freqCount = np.zeros(55)

for input_file in input_files:
    created_matrix = makeMatrix(input_file, input_folder)

    #name_of_file = "Results_of_" + input_file.split("_")[1].split()[0] + "_" + input_file.split()[1] + "_" + input_file.split()[2].split(".")[0]
    #name_of_entry = input_file.split("_")[1].split()[0] + "_" + input_file.split()[1] + "_" + input_file.split()[2].split(".")[0]
    file = open("F-FitTestOutput", "a")
    for count, row_index in enumerate(created_matrix):
        countTotal += 1
        n, bins, _ = plt.hist(row_index, bins=numBins, histtype='bar', log=False, color='b')

        R2 = DistroFit(1, 1, "f", bins[0], bins[numBins], n, bins, numBins)
        yn90 = 'Fail'
        yn95 = 'Fail'
        yn99 = 'Fail'
        if R2 >= .90:
            countPass90 += 1
            yn90 = 'Pass'
        if R2 >= .95:
            countPass95 += 1
            yn95 = 'Pass'
            freqCount[count] += 1
        if R2 >= .99:
            countPass99 += 1
            yn99 = 'Pass'

        file.write(input_file.split()[1] + " " + input_file.split()[2].split(".")[0] + " " + str(count + 1) + " R2: " + str(R2) + " " + yn90 + " " + yn95 + " " + yn99 + " " + "\n")
        plt.clf()
        print(count)
    file.close()
file = open("F-FitTestOutput", "a")
file.write("Percent Pass: " + str(countPass90 / countTotal) + " " + str(countPass95 / countTotal) + " " + str(countPass99 / countTotal) + "\n")
file.write("Counts by Frequency Band 95% Threshold: \n")
for count in range(0, 55):
    file.write(str(count + 1) + " " + str(freqCount[count]) + "\n")
file.close()
print('here')