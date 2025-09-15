# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 23:09:24 2025

@author: Fatima
"""

"""

"""

import pandas
import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

### Data Processing:
    
rawData = pandas.read_csv('Prices.csv')
#print(rawData.shape) #(3848, 733)

#store the SNP Index data time series seperately
SNPIndex = pandas.read_csv('Prices.csv', usecols=['Date', 'ES Index-A1-UnAdj on CME_USD (USD)'])

#store only the SNP stocks and their time series
StockValues = rawData.drop(columns = ['ES Index-A1-UnAdj on CME_USD (USD)'])

#fill in any initial NaNs in the Stock time series columns:
for col in StockValues.select_dtypes(include='number').columns: #go through each column
    #remove any Nan's from the column and record the first value
    first_val = StockValues[col].dropna().iloc[0]
    #replace the Nan's in the original column with that first value
    StockValues[col] = StockValues[col].fillna(first_val)
    
#Finding the daily returns: 
StockPR = StockValues.drop(columns = ['Date']) #remove date col
StockPR = StockPR.pct_change() #calculate the day-to-day percentage change
StockPR = StockPR.iloc[1:] #drop the first row (only Nas)

#Get the SNP index as a daily returns
SNP_PR = SNPIndex.drop(columns = ['Date']) #remove date col
SNP_PR = SNP_PR.pct_change() #calculate the day-to-day percentage change
SNP_PR = SNP_PR.dropna() #drop the first row (only Nas)

Dates = pandas.read_csv('Prices.csv', usecols=['Date']).to_numpy().flatten()
Dates = Dates[1:]
Dates = pandas.to_datetime(Dates, dayfirst=True)


del col, first_val, rawData, SNPIndex, StockValues

#Our data matrices
R = StockPR.cumsum()
r = SNP_PR.cumsum()

R = R.to_numpy()
r = r.to_numpy().flatten()


##### Define function to return sparse tracking portfolio weightings:

def cardinality_tracking(P, q, A, l, u, max_cardinality, counter = True):
    n = q.shape[0]  # number of stocks
    Trading_Stocks = n
    w_mask = np.ones(n)
    weightings = []

    while Trading_Stocks > max_cardinality:
        # Update upper bounds based on the current mask
        u_iter = np.hstack([1., np.ones(n) * w_mask])
        Trading_Stocks = int(sum(w_mask))
        
        # Setup and solve OSQP
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u_iter, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
        res = prob.solve()
        w_opt_i = res.x
        weightings.append(w_opt_i)
        
        # Update mask: remove smallest half (below median) of non-zero weights
        median_w = np.median(w_opt_i[abs(w_opt_i) > 1e-5])
        w_mask = np.ones(n) * (w_opt_i > median_w)
        
        if counter:
            print(f"Trading stocks remaining: {Trading_Stocks}")
    
    # Return the final weighting
    w_opt_final = weightings[-1]
    return w_opt_final, weightings, Trading_Stocks  # return all iterations optionally


#### Making Windows

# Make windows of two-year intervals, for which the tracking portfolio is constructed

#store the years of the dates time series only
years = Dates.year
#find the index of the first entry of a new (unique) year
unique_years, first_pos = np.unique(years, return_index=True) 

window_weightings = []
window_portfolio_size = []

for i in range(2, len(unique_years) ): 
    start_idx = first_pos[i-2]
    end_idx = first_pos[i]

    # Slice the returns matrix R and benchmark r
    R_window = R[start_idx:end_idx, :]
    r_window = r[start_idx:end_idx]

    # Define optimization parameters for this window
    P_window = 2 * (R_window.T @ R_window)
    q_window = -2 * (R_window.T @ r_window)

    P_sparse_window = sparse.csc_matrix(P_window)
    n_window = R_window.shape[1]

    # Constraints
    A_window = sparse.bmat([[np.ones((1, n_window)), None],
                            [sparse.eye(n_window), None]], format='csc')
    l_window = np.hstack([1., np.zeros(n_window)])
    u_window = np.hstack([1., np.ones(n_window)])

    # Set the maximum cardinality (number of stocks allowed in the portfolio)
    max_card = 50

    # Run the cardinality tracking function
    w_opt_final, all_iterations, portfolio_size = cardinality_tracking(P_sparse_window, q_window, 
                                                       A_window, l_window, u_window, 
                                                       max_card, counter=False)
    window_weightings.append(w_opt_final)
    window_portfolio_size.append(portfolio_size)
    
    # Plot cumulative return of the portfolio vs benchmark
    # plt.figure(figsize=(12,5))
    # plt.plot(Dates[start_idx:end_idx], r_window, label="Benchmark Index")
    # plt.plot(Dates[start_idx:end_idx], R_window @ w_opt_final, label="Tracker Portfolio")
    # plt.xlabel("Date")
    # plt.ylabel("Cumulative Returns")
    # plt.title(f"Tracker Portfolio vs Benchmark (years {i-2} - {i})")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    

### Tracking Portfolio Projection

# Now we have the optimal weightings for tracking within each window, we will
# project solutions forwards in time by one year to assess their performance

projected_portfoilo = []

for i in range(2, len(unique_years)-1 ): 
    start_idx = first_pos[i]
    end_idx = first_pos[i+1]

    # Slice the returns matrix R and benchmark r
    R_projected = R[start_idx:end_idx, :]
    projected_portfoilo.append(R_projected @ window_weightings[i-2])



start_idx = first_pos[2]
end_idx = first_pos[-1]


projected_portfoilo = np.concatenate(projected_portfoilo)
   
# Plot backtested portfolio against actual index returns
plt.figure(figsize=(12,5))
plt.plot(Dates[start_idx:end_idx], r[start_idx:end_idx], label="Benchmark Index")
plt.plot(Dates[start_idx:end_idx], projected_portfoilo, label="Tracker Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title(f"Out of Sample Tracker Portfolio vs SNP Benchmark (years {2012} - {2025})")
plt.legend()
plt.grid(True)
plt.show()


# Can also  project the optimal weightings for the first window forwards
# to compare the strategies performance with and without rebalancing


projected_portfoilo_win1 = R[start_idx:end_idx, :] @ window_weightings[0]

projected_portfoilo_win1 = np.concatenate(projected_portfoilo_win1)
   
# Plot backtested portfolio against actual index returns
plt.figure(figsize=(12,5))
plt.plot(Dates[start_idx:end_idx], r[start_idx:end_idx], label="Benchmark Index")
plt.plot(Dates[start_idx:end_idx], projected_portfoilo_win1, label="Tracker Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title(f"Out of Sample Tracker Portfolio vs SNP Benchmark (years {2012} - {2025})")
plt.legend()
plt.grid(True)
plt.show()


# we can find the MSE of both strategies: 
print("The annual rebalancing strategy has a MSE of: ",
      np.mean((projected_portfoilo - r[start_idx:end_idx])**2),
      "in the out of sample period of 2012-2025") 
print("Fixed weighting strategy has a MSE of: ",
      np.mean((projected_portfoilo_win1 - r[start_idx:end_idx])**2),
      "in the out of sample period of 2012-2025") 



fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# First plot
axes[0].plot(Dates[start_idx:end_idx], projected_portfoilo-r[start_idx:end_idx]  , color='blue')
axes[0].set_title("Tracking Error: Annually Rebalanced Protfolio")

# Second plot
axes[1].plot(Dates[start_idx:end_idx], projected_portfoilo_win1-r[start_idx:end_idx] , color='red')
axes[1].set_title("Tracking Error: Fixed Weighting Portfolio")

# Layout adjustment
plt.tight_layout()
plt.show()



# Clear trade-off in tracking performance. 
# Benefit to no rebalancing is we don't have cover any trading costs 
# Benefit to rebalancing is the portfolio more closely tracks the index performance

abs_weight_changes = window_weightings[:-1] - window_weightings[1:] 

# Convert list of arrays to 2D NumPy array
weights_array = np.array(window_weightings)

# Subtract consecutive rows
abs_weight_changes = np.abs(weights_array[1:] - weights_array[:-1])
yearly_trading_costs = np.sum(abs_weight_changes, axis = 1)*(4)
print(f"we lose {abs_weight_changes} BP on average each year by rebalancing")



# Visualising the cardinality of each optimal weighting solution 
# produced by the window of the preceeding two year period. 
plt.figure(figsize=(12,5))
plt.step(Dates[first_pos[2:]], window_portfolio_size, where='post', label="Portfolio Size")
plt.xlabel("Date")
plt.ylabel("Portfolio Cardinality")
plt.title(f"Out of Sample Tracker Portfolio vs SNP Benchmark (years {2012} - {2025})")
plt.legend()
plt.grid(True)
plt.show()









