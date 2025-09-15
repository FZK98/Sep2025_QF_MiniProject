# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:46:27 2025

@author: Fatima
"""
### Import Packages
import pandas
import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


### Data Processing 

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
    
#Finding the daily percentage returns: 
StockPR = StockValues.drop(columns = ['Date']) #remove date col
StockPR = StockPR.pct_change() #calculate the day-to-day percentage change
StockPR = StockPR.iloc[1:] #drop the first row (only Nas)

#Get the SNP index as a daily percentage change
SNP_PR = SNPIndex.drop(columns = ['Date']) #remove date col
SNP_PR = SNP_PR.pct_change() #calculate the day-to-day percentage change
SNP_PR = SNP_PR.dropna() #drop the first row (only Nas)

Dates = pandas.read_csv('Prices.csv', usecols=['Date']).to_numpy().flatten()
Dates = Dates[1:]
Dates = pandas.to_datetime(Dates, dayfirst=True)


del col, first_val, rawData, SNPIndex, StockValues

### Construct the Quadratic Programming Problem

# Create the data matrix and vectors. 
R = StockPR.cumsum()
r = SNP_PR.cumsum()
R = R.to_numpy()
r = r.to_numpy().flatten()

# OSQP solves the Quadratic equation: (1/2)w^tPw + q^tw .
# we define our P and q as required:

# Construct P and q
P = 2 * (R.T @ R)   # shape (n, n)
q = -2 * (R.T @ r)  # shape (n,)

# OSQP needs sparse matrices
P_sparse = sparse.csc_matrix(P)

# Constraints: None here, so set params as empty
A = sparse.csc_matrix((0, R.shape[1]))
l = np.array([])
u = np.array([])

# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P_sparse, q, A, l, u, verbose=True)

# Solve
res = prob.solve()

# Solution vector w (minimizer)
w_opt = res.x


# Portfolio Tracking 
plt.figure(figsize=(12, 5))
plt.plot(Dates, r, label="Benchmark Index", linewidth=1)
plt.plot(Dates, R @ w_opt, label="Tracker Portfolio", linewidth=1)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Tracker Portfolio vs S&P 500 Benchmark ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Tracking Error 
plt.figure(figsize=(12, 5))
plt.plot(Dates, (R @ w_opt) - r, label="Tracking Error", color='darkorange')
plt.xlabel("Date")
plt.ylabel("Difference (Tracker - Benchmark)")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Tracking Error Over Time")
plt.legend()
plt.show()


#Portfolio Weightings:
plt.figure(figsize=(12, 5))
plt.bar(np.arange(len(w_opt)), w_opt, label= "fit") 
plt.xlabel("Individual Stock")
plt.ylabel("Portfolio Allocation")
plt.title("Tracker Portfolio Weight Distribution Across SNP Stocks")
plt.ylim(min(w_opt)*1.1, max(w_opt)*1.1)  # add a bit of padding
plt.show()





############# Optimising Weights with Added Constraints
# Same problem with some constraints on the weightings
# Sum of weights = 1
# lowe bound of weights = 0 


n = R.shape[1] #the number of stocks to balance

#Optimisation Parameters 
A = sparse.bmat([[np.ones((1, n)),  None],
                 [sparse.eye(n),    None]], format='csc')
l = np.hstack([1., np.zeros(n)])
u = np.hstack([1., np.ones(n)])


# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P_sparse, q, A, l, u, verbose=True)
res = prob.solve()
w_opt = res.x

#print("Optimal w:", w_opt)

# Portfolio Tracking
plt.figure(figsize=(12, 5))
plt.plot(Dates, r, label="Benchmark Index", linewidth=1)
plt.plot(Dates, R @ w_opt, label="Tracker Portfolio", linewidth=1)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Tracker Portfolio vs S&P 500 Benchmark ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracking Error 
plt.figure(figsize=(12, 5))
plt.plot(Dates, (R @ w_opt) - r, label="Tracking Error", color='darkorange')
plt.xlabel("Date")
plt.ylabel("Difference (Tracker - Benchmark)")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Tracking Error Over Time")
plt.legend()
plt.show()


#Portfolio Weightings:
plt.figure(figsize=(12, 5))
plt.bar(np.arange(len(w_opt)), w_opt, label= "fit") 
plt.xlabel("Individual Stock")
plt.ylabel("Portfolio Allocation")
plt.title("Tracker Portfolio Weight Distribution Across SNP Stocks")
plt.ylim(min(w_opt)*1.1, max(w_opt)*1.1)  
plt.show()

#Default Tolerance of OSQP is 1e-3 
nonzero_count = sum(w_opt > 1e-3) #360
nonzero_count

# 360 individual stocks have non-zero weightings in the tracker solution
# the tolerance is too high, large numerical imprecisions produce 
# substantial weightings.

########### Constrained Problem with Lower Tolerance 

# Setup OSQP problem (constraints unchanged)
prob = osqp.OSQP()
# enforcing stricter tolerance on our solution. 
# 0<w_i<0 constraints enforces -1e-5<w_i<1e-5 in practise
prob.setup(P_sparse, q, A, l, u, verbose=True, eps_abs =1e-5, eps_rel=1e-5)
res = prob.solve()
w_opt = res.x

#print("Optimal w:", w_opt)

# Portfolio Tracking:
plt.figure(figsize=(12, 5))
plt.plot(Dates, r, label="Benchmark Index", linewidth=1)
plt.plot(Dates, R @ w_opt, label="Tracker Portfolio", linewidth=1)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Tracker Portfolio vs S&P 500 Benchmark (295 Stocks)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracking Error: 
plt.figure(figsize=(12, 5))
plt.plot(Dates, (R @ w_opt) - r, label="Tracking Error", color='darkorange')
plt.xlabel("Date")
plt.ylabel("Difference (Tracker - Benchmark)")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Tracking Error Over Time")
plt.legend()
plt.show()


# Portfolio Weightings:
plt.figure(figsize=(12, 5))
plt.bar(np.arange(len(w_opt)), w_opt, label= "fit") 
plt.xlabel("Individual Stock")
plt.ylabel("Portfolio Allocation")
plt.title("Tracker Portfolio Weight Distribution Across SNP Stocks")
plt.ylim(min(w_opt)*1.1, max(w_opt)*1.1)  
plt.show()

sum(u-1) #731 - how many weights we can allocate
nonzero_count = sum(w_opt > 1e-3) #295 - number of significant allocations
nonzero_count

#We actually have 295 stocks with at least 10BP 



######### Sparse Tracking Solution

# Ideally we would use a cardinality constraint, but OSQP does not support 
# mixed integer optimisation. Instead we will implement a heuristic which 
# pushes the smallest 50% of (non-zero) weightings to zero

#Find the weightings that have less than the median weight value 
#threshold value: find the median of the weights which are non-zero
median_nonzero = np.median(w_opt[abs(w_opt)>1e-5])

#Any weights smaller than the following value will be shrunk to zero
np.median(w_opt[abs(w_opt)>1e-5]) #0.0019376211235125184

#number of stocks we eliminate from the portfolio
sum(w_opt < median_nonzero) #533
# out of total number of stocks
len(w_opt) #731

# Masking vector, index=1 to keep the stock, index=0 to remove the stock
w_opt_dash = w_opt*(w_opt > median_nonzero)
sum(w_opt_dash) #how much original weighting we we keep (about 80%)

# re-normalise weightings:
w_opt_dash = w_opt_dash/sum(w_opt_dash)

#see the difference in the weighting vectors
fig, axes = plt.subplots(1, 2, figsize=(12,5))
# UnPruned
axes[0].bar(np.arange(len(w_opt)), w_opt) 
axes[0].set_title("Weightings from OSQP")
# Pruned
axes[1].bar(np.arange(len(w_opt_dash)), w_opt_dash) 
axes[1].set_title("Pruned Weightings from OSQP")
plt.show()

# Tracking performance of the pruned portfolio.
plt.figure(figsize=(12, 5))
plt.plot(Dates, r, label="Benchmark Index", linewidth=1)
plt.plot(Dates, R @ w_opt_dash, label="Reduced Tracker Portfolio", linewidth=1)

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))

plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Reduced Tracker Portfolio vs S&P 500 Benchmark (198 Stocks)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# We see an increased tracker error - we need to re-optimise weights for closer tracking

### Skip to line 440 to see the iterative solution fully implemented. 


######### Sparse Tracker Second Iteration 1 (proof of concept)

# re-optimising with the reduced number of stocks. 
# not deleting the stocks from the original data set 
# just changing the upper bounds to 0 for the stocks we don't want 


n = R.shape[1]
#
A = sparse.bmat([[np.ones((1, n)),  None],
                 [sparse.eye(n),    None]], format='csc')
l = np.hstack([1., np.zeros(n)])
u = np.hstack([1., np.ones(n)*(w_opt > median_nonzero)]) 

sum(u-1) #198 - we have this many remaining stocks to allocate weights to (731-533)

# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P_sparse, q, A, l, u, verbose=True, eps_abs = 1e-5, eps_rel = 1e-5)
res = prob.solve()
w_opt2 = res.x


# Tracking Portfolio:
plt.figure(figsize=(12, 5))
plt.plot(Dates, r, label="Benchmark Index", linewidth=1)
plt.plot(Dates, R @ w_opt2, label="Tracker Portfolio", linewidth=1)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Optimised Tracker Portfolio vs S&P 500 Benchmark (198 Stocks)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracking Error: 
plt.figure(figsize=(12, 5))
plt.plot(Dates, (R @ w_opt2) - r, label="Tracking Error", color='darkorange')
plt.xlabel("Date")
plt.ylabel("Difference (Tracker - Benchmark)")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Tracking Error Over Time")
plt.legend()
plt.show()


#Portfolio Weightings:
plt.figure(figsize=(12, 5))
plt.bar(np.arange(len(w_opt2)), w_opt2, label= "fit") 
plt.xlabel("Individual Stock")
plt.ylabel("Portfolio Allocation")
plt.title("Tracker Portfolio Weight Distribution Across SNP Stocks")
plt.ylim(min(w_opt2)*1.1, max(w_opt2)*1.1)  
plt.show()



nonzero_count = sum(w_opt2 > 1e-3) #131 stocks with at least 10BP

#New weighting cut-off threshold. ALl weights smalles than this will be force to zero
median_nonzero2 = np.median(w_opt2[abs(w_opt2)>1e-5])

np.median(w_opt2[abs(w_opt2)>1e-5]) #0.006086128314099168

#This leaves us with 
sum(w_opt2>median_nonzero2) #71
#stocks to construct our portfolio



######### Sparse Tracking Iteraction 2 (proof of concept)

#re-optimising with the reduced number of stocks. 
#not deleting the stocks from the original data set 
#just changing the upper bounds to 0 for the stocks we don't want 


n = R.shape[1]
#
A = sparse.bmat([[np.ones((1, n)),  None],
                 [sparse.eye(n),    None]], format='csc')
l = np.hstack([1., np.zeros(n)])
u = np.hstack([1., np.ones(n)*(w_opt2 > median_nonzero2)])

sum(u)-1 #71 stocks 

# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P_sparse, q, A, l, u, verbose=True, eps_abs = 1e-5, eps_rel = 1e-5)

# Solve
res = prob.solve()

# Solution vector w (minimizer)
w_opt3 = res.x

#print("Optimal w:", w_opt2)



# Tracking Portfolio: 
plt.figure(figsize=(12, 5))
plt.plot(Dates, r, label="Benchmark Index", linewidth=1)
plt.plot(Dates, R @ w_opt3, label="Tracker Portfolio", linewidth=1)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Optimised Tracker Portfolio vs S&P 500 Benchmark (71 Stocks)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracking Error
plt.figure(figsize=(12, 5))
plt.plot(Dates, (R @ w_opt3) - r, label="Tracking Error", color='darkorange')
plt.xlabel("Date")
plt.ylabel("Difference (Tracker - Benchmark)")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Tracking Error Over Time")
plt.legend()
plt.show()


# Portfolio Weightings:
plt.figure(figsize=(12, 5))
plt.bar(np.arange(len(w_opt3)), w_opt3, label= "fit") 
plt.xlabel("Individual Stock")
plt.ylabel("Portfolio Allocation")
plt.title("Tracker Portfolio Weight Distribution Across SNP Stocks")
plt.ylim(min(w_opt3)*1.1, max(w_opt3)*1.1)  
plt.show()


nonzero_count = sum(w_opt3 > 1e-3) #69 stocks with at least 10 BP
nonzero_count


############ Complete Iterative Heuristic Solution #############

# Instead of manually iterating through smaller portfolios, we will iterate,
# until the number of non-zero weightings falls below some cardinality threshold.


#Implement the optimisation in a loop. each iteration produces optimal weightings

# Reinitialise OSQP optimisation problem - only need to run this block to find
# the solution: 
    
#Optimization Parameters:
P = 2 * (R.T @ R)   
q = -2 * (R.T @ r)  
P_sparse = sparse.csc_matrix(P)
n = R.shape[1] #number of available stocks
A = sparse.bmat([[np.ones((1, n)),  None],
                 [sparse.eye(n),    None]], format='csc')
l = np.hstack([1., np.zeros(n)])
u = np.hstack([1., np.ones(n)])

# Cardinality requirement
Number_Stocks = 50 #the maximum number of stock included in our portfolio
Trading_Stocks = n #the number of stocks with non-zero allocation
w_mask = np.ones(n) 
weightings = [] 
itr = 1
cardinality = []

while Trading_Stocks > Number_Stocks: 
    u = np.hstack([1., np.ones(n)*w_mask])
    Trading_Stocks = sum(w_mask)
    cardinality.append(Trading_Stocks)
    prob = osqp.OSQP()
    prob.setup(P_sparse, q, A, l, u, verbose=False, eps_abs = 1e-5, eps_rel = 1e-5)
    res = prob.solve()
    w_opt_i = res.x
    weightings.append(w_opt_i)
    
    median_w_icutoff = np.median(w_opt_i[abs(w_opt_i)>1e-5])

    w_mask = np.ones(n)*(w_opt_i > median_w_icutoff)
    print(Trading_Stocks)

    

#how many iterations we required
len(weightings) 


# In Sample Tracking Portfolio Performance
plt.figure(figsize=(12, 5))
for i, w in enumerate(weightings):
    plt.plot(np.arange(len(r)), R @ w, label=f"Fit Iter {i+1}, with {int(cardinality[i])} stocks" )
# Add the benchmark
plt.plot(np.arange(len(r)), r, label="Benchmark", linewidth=2, color='black', linestyle='--')

# Labels and legend
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.title("Tracker Portfolio Fits Over Iterations")
plt.legend()
plt.tight_layout()
plt.show()



# Visualise the change in weight allocation from iteration to iteration 
# for the final portfolio 

W = np.vstack(weightings)  # shape (n_iter, n_stocks)
sorted_indices = np.argsort(-W[-1])  # descending order based on last iteration
top_n = 60
top_indices = sorted_indices[:top_n]
top_weights = W[:, top_indices]  # shape: (n_iter, top_n)

bar_width = 0.15
x = np.arange(top_n)  # one group per stock

plt.figure(figsize=(20, 10))

for i in range(len(weightings)):
    offset = i * bar_width
    plt.bar(x + offset, top_weights[i], width=bar_width, label=f'Iter {i+1}')

plt.xlabel("Stock (sorted by final iteration weight)")
plt.ylabel("Portfolio Weight")
plt.title("Top Stock Portfolio Weights Across Iterations")
plt.xticks(x + bar_width * (len(weightings) - 1) / 2, [f" {i}" for i in top_indices], rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()



#Mean Squares Error between the tracking and the original pricings
mse = np.mean((R @ w_opt - r)**2)
print(f"Tracking MSE: {mse:.4f}")

# comparing the MSE between solutions
for i, w in enumerate(weightings):
    print(f"Iter {i+1} error: {np.mean((R @ w - r)**2):.6f}")


nonzero_counts = [np.sum(w > 1e-6) for w in weightings]
plt.plot(range(1,len(weightings)+1), nonzero_counts)
plt.xlabel("Iteration Number")
plt.ylabel("Number os stocks with non-negative weightings")
plt.title("Reduction in Solution Cardinality across Iterations")
plt.xticks(np.arange(len(weightings))+1)
plt.show()

np.mean(weightings[-1])

plt.bar(np.arange(len(weightings[-1])), weightings[-1], label= "fit") 


#### showing the top 200 weighted stocks for each iteration

top_n = 200  # show top 20 weights
iterations_to_plot = 4  # first 4 iterations

fig, axes = plt.subplots(2, 2, figsize=(16,10))
axes = axes.flatten()  # easier indexing

for i in range(iterations_to_plot):
    w = weightings[i]
    # get indices of top_n weights
    top_indices = np.argsort(-w)[:top_n]
    top_weights = w[top_indices]
    
    axes[i].bar(np.arange(top_n), top_weights, color='skyblue')
    axes[i].set_title(f'Iteration {i+1} - Top {top_n} Weights')
    axes[i].set_xlabel('Stock Rank')
    axes[i].set_ylabel('Weight')
    axes[i].grid(True, axis='y')

plt.tight_layout()
plt.show()



###### Final Tracking Portfolio Allocations

#Match the weightings to the stocks

# Get stock names (excluding Date and index column)
StockPR.columns = StockPR.columns.str.replace("_US_USD_USD", "", regex=False)
stock_names = StockPR.columns.to_numpy()

# Pick final weights (example: last iteration from your loop)
final_weights = weightings[-1]  # or use w_opt3

# Combine into a DataFrame for easier viewing
portfolio = pandas.DataFrame({
    "SNP 500 Stock": stock_names,
    "Weight": final_weights
})

# Keep only nonzero weights (or above a tolerance)
portfolio = portfolio[portfolio["Weight"] > 1e-6]

# Sort by weight descending
portfolio = portfolio.sort_values(by="Weight", ascending=False)

print(portfolio)


