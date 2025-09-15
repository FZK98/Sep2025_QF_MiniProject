# Sep2025_QF_MiniProject
This project implements a cardinality-constrained index tracking strategy in Python.

**'In_Sample_SNP_Tracker.py'** :  
Historical stock price data, the workflow processes returns, applies quadratic 
programming with OSQP to construct sparse portfolios, using an iterative heutistic to 
enforce cardinality consraints.

**'Out_of_Sample_SNP_Tracker.py'** :  
Evaluates the performance of the iterative heuristic portfolio construction strategy
through backtesting real SNP index data across the period of 2010-2025. This performance
is compared to the real SNP index returns. The success of the tracking is evaluated 
through the MSE of the series. A rolling two-year window is produced to rebalance
the sparse portfolio once a year. 

**'in_sample_portfolio_tracking.ipynb'** and **'out_of_sample_tracking.ipynb'** :  
Markdown files of corresponding material, with the main results summarised. 
