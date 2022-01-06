import yfinance as yf
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import math
import os.path
cash = 100000

stocks = ['BRK-A', 'AAPL', 'MSFT', 'JPM', 'GOOG', 'BAC', 'INTC', 'WFC',
          'C', 'VZ', 'FB', 'PFE', 'JNJ', 'WMT', 'XOM',
          'FNMA', 'T', 'UNH', 'CMCSA', 'V', "NVDA", 'TSLA', 'GOOGL', 'HD', 'LOW', 'PYPL']

data = yf.download(stocks, period='2y')

closes = np.transpose(np.array(data.Close))  # matrix of daily closing prices
abs_diff = np.diff(closes)                   # change in closing price each day
rel_diff = np.divide(abs_diff, closes[:,:-1]) # relative change in daily closing price
delta = np.mean(rel_diff, axis=1)            # mean price change
sigma = np.cov(rel_diff)                     # covariance (standard deviations)
std = np.std(rel_diff, axis=1)               # standard deviation
size = len(data.Close)
CP = np.zeros(len(stocks))
count = 0
for i in stocks:
    CP[count] = data.Close[i][size-1]
    count = count + 1

# Create an empty model
m = gp.Model('portfolio')

# Add matrix variable for the stocks
x = m.addMVar(len(stocks))

# Objective is to minimize risk (squared).  This is modeled using the
# covariance matrix, which measures the historical correlation between stocks
portfolio_risk = x @ sigma @ x
m.setObjective(portfolio_risk, GRB.MINIMIZE)

# Fix budget with a constraint
m.addConstr(x.sum() == 1, 'budget')

# Verify model formulation
m.write('portfolio_selection_optimization.lp')

# Optimize model to find the minimum risk portfolio
m.optimize()

results_folder = "Results"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
else:
    for file in os.listdir(results_folder):
        os.remove(os.path.join(results_folder, file))


port_proportions = cash*np.array(m.x)
amount = np.zeros(len(stocks))
total = 0
file = open("Results/Portfolio", "a")
file.write("Total Cash: " + str(cash) + "\n" + "Minimum Risk Portfolio" + "\n" + "Stock" + " - " + "Portfolio Percentages" + " - "
           + "Amount of Dollars for Stock" + " - " + "Price of Stock"
           + " - " + "Amount of Stocks" + "\n")
file.close()
total = 0
for i in range(len(stocks)):
    if (CP[i] < port_proportions[i]):
        amount[i] = math.floor(port_proportions[i]/CP[i])
        total = total + CP[i]*amount[i]
        file = open("Results/Portfolio", "a")
        file.write(stocks[i].ljust(7) + " " + str(m.x[i]).ljust(23) + " " + str(cash*np.array(m.x[i])).ljust(29) + " "
                   + str(round(CP[i], 2)).ljust(16) + " " + str(amount[i]) + "\n")
        file.close()

file = open("Results/Portfolio", "a")
temp = np.sort(m.x)
max_ind= np.where(m.x == temp[len(temp)-1])[0][0]
second_max = np.where(m.x == temp[len(temp)-2])[0][0]
file.write("Left Over Cash: " + str(cash - total) + "\n" + "Advised to put left over cash into "
           + stocks[max_ind] + " And " + stocks[second_max] + "\n" + "\n")
file.close()

minrisk_volatility = sqrt(m.ObjVal)
minrisk_return = delta @ x.X
pd.DataFrame(data=np.append(x.X, [minrisk_volatility, minrisk_return]),
             index=stocks + ['Volatility', 'Expected Return'],
             columns=['Minimum Risk Portfolio'])


# Create an expression representing the expected return for the portfolio
portfolio_return = delta @ x
target = m.addConstr(portfolio_return == minrisk_return, 'target')

# Solve for efficient frontier by varying target return
frontier = np.empty((2,0))
for r in np.linspace(delta.min(), delta.max(), 25):
    # Added code
    file = open("Results/Portfolio", "a")
    target[0].rhs = r
    m.optimize()
    file.write("Expected Returns: " + str(r) + "\n")
    file.close()
    total = 0
    for i in range(len(stocks)):
        port_proportions = cash * np.array(m.x)
        amount = np.zeros(len(stocks))
        if (CP[i] < port_proportions[i]):
            amount[i] = math.floor(port_proportions[i] / CP[i])
            total = total + CP[i] * amount[i]
            file = open("Results/Portfolio", "a")
            file.write(
                stocks[i].ljust(7) + " " + str(m.x[i]).ljust(23) + " " + str(cash * np.array(m.x[i])).ljust(29) + " "
                + str(round(CP[i], 2)).ljust(16) + " " + str(amount[i]) + "\n")
            file.close()

    file = open("Results/Portfolio", "a")
    temp = np.sort(m.x)
    max_ind = np.where(m.x == temp[len(temp) - 1])[0][0]
    second_max = np.where(m.x == temp[len(temp) - 2])[0][0]
    file.write("Left Over Cash: " + str(cash - total) + "\n" + "Advised to put left over cash into "
               + stocks[max_ind] + " And " + stocks[second_max] + "\n" + "\n")
    file.close()

    frontier = np.append(frontier, [[sqrt(m.ObjVal)],[r]], axis=1)


fig, ax = plt.subplots(figsize=(10, 8))

# Plot volatility versus expected return for individual stocks
ax.scatter(x=std, y=delta,
           color='Blue', label='Individual Stocks')
for i, stock in enumerate(stocks):
    ax.annotate(stock, (std[i], delta[i]))

# Plot volatility versus expected return for minimum risk portfolio
ax.scatter(x=minrisk_volatility, y=minrisk_return, color='DarkGreen')
ax.annotate('Minimum\nRisk\nPortfolio', (minrisk_volatility, minrisk_return),
            horizontalalignment='right')

# Plot efficient frontier
ax.plot(frontier[0], frontier[1], label='Efficient Frontier', color='DarkGreen')

# Format and display the final plot
ax.axis([frontier[0].min()*0.7, frontier[0].max()*1.3, delta.min()*1.2, delta.max()*1.2])
ax.set_xlabel('Volatility (standard deviation)')
ax.set_ylabel('Expected Return')
ax.legend()
ax.grid()
plt.show()
file.close()
