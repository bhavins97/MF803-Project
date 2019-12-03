from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

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
    
    def get_corr_mat(self):
        returns = self.get_returns_data()
        corr_mat = returns.corr()
        return corr_mat

class BS_sim:

    def __init__(self, initial_stock_prices, returns, vols, corr_mat, maturity):
        self.corr_mat = np.array(corr_mat)
        self.initial_stock_prices = initial_stock_prices
        self.returns = returns
        self.maturity = maturity
        self.steps = int(252*maturity)
        self.vols = vols
    
    def simulate(self, sim_number = 1000):
        dt = self.maturity/self.steps
        n_stocks = len(self.initial_stock_prices) 
        prices = np.zeros((n_stocks,self.steps,sim_number))  #np array of zeros where the simulated prices will go
        # structure of prices is the following: 
        # prices[i] gives you the price paths of the stock at index i
        # the next level, prices[i][j] gives all the simulated prices of the stock at index i at time j (so if you want the prices at maturity, type prices[i][-1]) 
        
        for stock in range(n_stocks):
            prices[stock][0] = self.initial_stock_prices[stock]  #populating with the initial stock prices
        
        for time in range(1,self.steps):
            random_variables = np.random.multivariate_normal(self.returns, self.corr_mat, sim_number) #generating correlated random variables
            for stock in range(n_stocks):
                prices[stock][time] = prices[stock][time-1] + self.returns[stock]*prices[stock][time-1]*dt + self.vols[stock]*prices[stock][time-1]*np.sqrt(dt)*random_variables[:,stock]
        return prices


#testing things out
tickers = ['AAPL', 'MSFT','F']
test = stock_history_finder(tickers)
returns = test.get_ann_returns()
vols = test.get_vols()
cov_mat = test.get_cov_mat()
corr_mat = test.get_corr_mat()
latest_prices = test.get_latest_prices()
latest_prices = [latest_prices[ticker] for ticker in tickers]
#testing out a simulation
test_sim = BS_sim([100,200], [0,0], [0.25,0.25] ,[[1,-0.5],[-0.5,1]], 1)
test_output = test_sim.simulate(sim_number = 5000)
plt.plot(test_output[0])
plt.show()
#testing out put price, should be near 9.75
payoffs = np.maximum(100-test_output[0][-1],0)
print(sum(payoffs)/5000)