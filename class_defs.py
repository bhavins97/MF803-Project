from pandas_datareader import data
import pandas as pd
import numpy as np

class stock_history_finder:
    def __init__(self, tickers):
        self.tickers =  tickers
    
    def get_returns_data(self):
        prices = data.DataReader(self.tickers,'yahoo', '2019-01-01','2019-11-30')
        adj_close_prices = prices['Adj Close']
        return adj_close_prices.pct_change(1)
        
    def get_vols(self):
        returns = self.get_returns_data()
        vols = returns.std()
        ann_vols = vols * np.sqrt(252)
        return ann_vols
    
    def get_cov_mat(self):
        returns = self.get_returns_data()
        cov_mat = returns.cov()
        return cov_mat

test = stock_history_finder(['AAPL','MSFT'])
vols = test.get_vols()
print(vols)
cov_mat = test.get_cov_mat()
print(np.matrix(cov_mat))