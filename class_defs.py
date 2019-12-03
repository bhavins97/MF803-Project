from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime

class stock_history_finder:
    """
    Get stock returns, realized volatility and correlation structures
    """
    def __init__(self, tickers):
        """
        tickers: list
        """
        self.tickers =  tickers
    
    def get_prices(self):
        today = datetime.datetime.today()
        today = today.strftime('%Y-%m-%d')
        prices = data.DataReader(self.tickers,'yahoo', '2015-01-01', today )
        return prices['Adj Close']

    def get_latest_prices(self):
        prices = self.get_prices()
        return prices.tail(1)

    def get_returns_data(self):
        adj_close_prices = self.get_prices()
        return adj_close_prices.pct_change(1)

    def get_ann_returns(self):
        returns = self.get_returns_data()
        avg_returns = returns.mean()
        ann_avg_returns = avg_returns * 252
        return ann_avg_returns

    def get_vols(self):
        returns = self.get_returns_data()
        vols = returns.std()
        ann_vols = vols * np.sqrt(252)
        return ann_vols
    
    def get_cov_mat(self):
        returns = self.get_returns_data()
        cov_mat = returns.cov()
        return cov_mat

class BS_sim:

    def __init__(self, initial_stock_prices, returns, cov_mat, maturity):
        self.cov_mat = cov_mat
        self.initial_stock_prices = initial_stock_prices
        self.returns = returns
        self.maturity = maturity
        self.steps = int(252*maturity)
    
    def simulate(self, sim_number):
        dt = self.maturity/self.steps
        n_stocks = len(self.initial_stock_prices) 
        prices = np.zeros((n_stocks,self.steps,sim_number))  #np array of zeros where the simulated prices will go
        # structure of prices is the following: 
        # prices[i] gives you the price paths of the stock at index i
        # the next level, prices[i][j] gives all the simulated prices of the stock at index i at time j (so if you want the prices at maturity, type prices[i][-1]) 
        
        for stock in range(n_stocks):
            prices[stock][0] = self.initial_stock_prices[stock]  #populating with the initial stock prices
        
        #write actual simulation 
        
        return prices


#testing things out
tickers = ['AAPL', 'MSFT','F']
test = stock_history_finder(tickers)
returns = test.get_ann_returns()
print(returns)
vols = test.get_vols()
print(vols)
cov_mat = test.get_cov_mat()
print(np.matrix(cov_mat))
latest_prices = test.get_latest_prices()
latest_prices = [latest_prices[ticker] for ticker in tickers]
print(latest_prices)
print("------------------")
test_sim = BS_sim(latest_prices, [0,0,0], cov_mat, 0.5)
test_output = test_sim.simulate(5)
print(test_output)
# test_mat = np.zeros((2,3,3))
# print(test_mat)
# print(test_mat[0])
# print(test_mat[0][0])