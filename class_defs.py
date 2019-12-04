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
        tickers: list (last element of the list should be the index on which the payoff will be based)
        """
        self.tickers =  tickers
    
    def get_prices(self):
        today = datetime.datetime.today()
        today = today.strftime('%Y-%m-%d')
        prices = data.DataReader(self.tickers,'yahoo', '2015-01-01', today )
        return prices['Adj Close']

    def get_latest_prices(self):
        prices = self.get_prices()
        last_prices = prices.tail(1)
        return [last_prices[ticker] for ticker in self.tickers]

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
    
    def simulate(self, sim_number = 10000):
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

class AtlasOption:

    def __init__(self, n1, n2, strike, paths):
        self.paths = paths
        self.index_prices = paths[-1][-1]
        self.stock_prices = paths[:-1]
        self.n1 = n1
        self.n2 = n2
        self.strike = strike
        self.maturity = len((paths[0].transpose())[0])/252
    
    def get_price(self):
        initial_prices = np.array([self.stock_prices[i][0] for i in range(len(self.stock_prices))])
        final_prices = np.array([self.stock_prices[i][-1] for i in range(len(self.stock_prices))])
        returns = final_prices/initial_prices

        sim_grouped_returns = returns.transpose() #grouping returns from each simulation together
        sorted_grouped_returns  = np.array([np.sort(sim_grouped_returns[i]) for i in range(len(sim_grouped_returns))])    #sorting each subarray
        
        #below, taking out n1 stocks from the top and n2 from the bottom
        if self.n1 != 0:
            filtered_returns = np.array([sorted_grouped_returns[i][self.n2:-self.n1] for i in range(len(sorted_grouped_returns))])
        elif self.n1 == 0 and self.n2 ==0:
            filtered_returns = sorted_grouped_returns
        elif self.n1 == 0:
            filtered_returns = np.array([sorted_grouped_returns[i][self.n2:] for i in range(len(sorted_grouped_returns))])

        returns_minus_strike = filtered_returns - self.strike #subtracting the strike

        sum_term = np.array([np.sum(returns_minus_strike[i]) for i in range(len(returns_minus_strike))]) #summing up the remaining stocks in the basket
        
        payoff_term_1 = sum_term/(len(self.stock_prices)-(self.n1+self.n2)) #dividing by the number of remaining stocks
        
        payoffs = np.maximum(payoff_term_1, np.zeros(len(payoff_term_1))) #taking rhe max of 0 and the other payoff term
        
        payoffs_dollars = payoffs * self.index_prices #converting perecentage payoffs to dollar amounts
        
        ### TEMPORARY DEFS###############
        r = 0
        #########################
        
        discounted_payoffs = np.exp(-r*self.maturity) * payoffs_dollars
        avg_disc_payoff = np.average(discounted_payoffs)
        
        return avg_disc_payoff, self.maturity

#testing things out
tickers = ['AAPL','MSFT','AMZN','VOO']
test = stock_history_finder(tickers)
returns = test.get_ann_returns()
vols = test.get_vols()
cov_mat = test.get_cov_mat()
corr_mat = test.get_corr_mat()
latest_prices = test.get_latest_prices()
#latest_prices = [latest_prices[ticker] for ticker in tickers]
#testing out a simulation
test_sim = BS_sim(latest_prices,len(tickers)*[0] , vols ,corr_mat, 0.5)
test_output = test_sim.simulate(sim_number = 5000)
# plt.plot(test_output[0])
# plt.show()
# plt.plot(test_output[1])
# plt.show()
# plt.plot(test_output[2])
# plt.show()

# testing at the money put prices
# for stock in range(len(tickers)):
#     payoffs = np.maximum(float(latest_prices[stock])-test_output[stock][-1],0)
#     print(sum(payoffs)/5000)

test_pricer_00 = AtlasOption(0,0,1,test_output)
print(test_pricer_00.get_price())
test_pricer_01 = AtlasOption(0,1,1,test_output)
print(test_pricer_01.get_price())
test_pricer_10 = AtlasOption(1,0,1,test_output)
print(test_pricer_10.get_price())
test_pricer_11 = AtlasOption(1,1,1,test_output)
print(test_pricer_11.get_price())
test_pricer_20 = AtlasOption(2,0,1,test_output)
print(test_pricer_20.get_price())
test_pricer_02 = AtlasOption(0,2,1,test_output)
print(test_pricer_02.get_price())