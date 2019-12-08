from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

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
        prices = prices['Adj Close']
        prices = prices[self.tickers]
        return prices

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
        #could add premium over realized vol
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

    def __init__(self, initial_stock_prices, rf_rate, vols, corr_mat, maturity):
        self.corr_mat = np.array(corr_mat)
        self.initial_stock_prices = initial_stock_prices
        self.rf_rate = rf_rate
        self.maturity = maturity
        self.steps = int(252*maturity)
        self.vols = vols
    
    def simulate(self, sim_number = 10000):

        if self.maturity == 0:
            """ If maturity is 0, then it just returns the inital prices"""
            return np.array([sim_number*[self.initial_stock_prices[stock]] for stock in range(len(self.initial_stock_prices)) ])
        
        dt = self.maturity/self.steps
        n_stocks = len(self.initial_stock_prices) 
        prices = np.zeros((n_stocks,self.steps,sim_number))  #np array of zeros where the simulated prices will go
        # structure of prices is the following: 
        # prices[i] gives you the price paths of the stock at index i
        # the next level, prices[i][j] gives all the simulated prices of the stock at index i at time j (so if you want the prices at maturity, type prices[i][-1]) 
        
        for stock in range(n_stocks):
            prices[stock][0] = self.initial_stock_prices[stock]  #populating with the initial stock prices
        
        for time in range(1,self.steps):
            #random_variables = np.random.multivariate_normal(n_stocks*[self.rf_rate], self.corr_mat, sim_number) #generating correlated random variables
            random_variables = np.random.multivariate_normal(n_stocks*[0], self.corr_mat, sim_number) #generating correlated random variables
            for stock in range(n_stocks):
                #prices[stock][time] = prices[stock][time-1] + self.rf_rate*prices[stock][time-1]*dt + self.vols[stock]*prices[stock][time-1]*np.sqrt(dt)*random_variables[:,stock]
                prices[stock][time] = prices[stock][time-1] * np.exp((self.rf_rate - ((self.vols[stock]**2)/2) ) * dt + self.vols[stock]*np.sqrt(dt)* random_variables[:,stock])
        
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
        else:
            filtered_returns = np.array([sorted_grouped_returns[i][self.n2:] for i in range(len(sorted_grouped_returns))])

        returns_minus_strike = filtered_returns - self.strike #subtracting the strike

        sum_term = np.array([np.sum(returns_minus_strike[i]) for i in range(len(returns_minus_strike))]) #summing up the remaining stocks in the basket
        
        payoff_term_1 = sum_term/(len(self.stock_prices)-(self.n1+self.n2)) #dividing by the number of remaining stocks
        
        payoffs = np.maximum(payoff_term_1, np.zeros(len(payoff_term_1))) #taking rhe max of 0 and the other payoff term
        
        payoffs_dollars = payoffs * self.index_prices #converting perecentage payoffs to dollar amounts
        
        ### TEMPORARY DEF ########
        r = 0.0184
        ##########################
        
        discounted_payoffs = np.exp(-r*self.maturity) * payoffs_dollars
        avg_disc_payoff = np.average(discounted_payoffs)
        
        return avg_disc_payoff
    
class AtlasPlot:
    
    def __init__(self, ticker_list, sim):
        self.tickers = ticker_list
        self.simulation = sim
        
    def plot_n1_n2_price(self): #3d plot of n1, n2, and price
        
        #n1 and n2 are the x and y axes
        x = range(len(self.tickers)//2 + 1) #n1 and n2 shouldn't be bigger than half of the basket, right?
        y = range(len(self.tickers)//2 + 1) #we don't want n1+n2 > len(tickers)
        #price is the z axis
        z = np.zeros((len(x),len(y)))
        #z is a two dimensional array of prices
        for i in range(len(x)):
            for j in range(len(y)):
                atlas_option_ij = AtlasOption(i,j,1,self.simulation)
                z[i][j] = atlas_option_ij.get_price()
        
        #plot commands
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)
        ax.set_title('Relationship of price to choices of n1 and n2')
        ax.set_xlabel('n2')
        ax.set_ylabel('n1')
        ax.set_zlabel('price')
        ax.plot_surface(X,Y,z) #we can experiment with colors, shading, etc.
        plt.show()
        
    def plot_strike_price(self): #2d plot of strike and price
        
        #strike is the x axis, price is the y axis
        x = np.linspace(0.5,1.5,20) #using a bunch of different arbitrary values for strike
        y = np.zeros(len(x))
        for i in range(len(x)):
            atlas_option_i = AtlasOption(5,5,x[i],self.simulation) #arbitrarily picked n1=n2=5
            y[i] = atlas_option_i.get_price()
        #plot commands
        plt.xlabel('strike')
        plt.ylabel('price')
        plt.title('Relationship of strike to price')
        plt.plot(x,y)
        plt.show()

##### THINGS TO DO! #######
# 1. Validation (testing), try to get as many different test cases as possible (where we know what the output should be). Price a basket on OVME on Bloomberg
#DONE# 2. Get an example going. Play around with changing n1 and n2, get a 3d graph of price with different n1 and n2. Plot price against strike
# 3. Look into premium of implied over realized vol and maybe add a premium to our calculation
# 4. Look at what happens when the basket is made up of assets that are highly correlated vs low/negative(?) correlation
#DONE# 5. Try using risk free rate as the expected return in the BS model

# ADD ANYTHING ELSE YOU CAN THINK OF THAT COULD BE USEFUL



#testing things out

# tickers = ['AAPL','MSFT', 'F', 'AMZN','VOO']
# tickers = ['AAPL','TSLA','BND','VOO']
# test = stock_history_finder(tickers)
# vols = test.get_vols()
# print(vols)
# corr_mat = test.get_corr_mat()
# print(corr_mat)
# latest_prices = test.get_latest_prices()
# #testing out a simulation
# test_sim = BS_sim(latest_prices, 0.0184 , vols ,corr_mat, 1)
# test_output = test_sim.simulate(sim_number = 5000)

# test_atlas = AtlasOption(0,0,1,test_output)
# print(test_atlas.get_price())

# plt.plot(test_output[0])
# plt.show()
# plt.plot(test_output[1])
# plt.show()
# plt.plot(test_output[2])
# plt.show()

# testing at the money put prices
# for stock in range(len(tickers)):
#     payoffs = np.exp(-0.0184) * np.maximum(float(latest_prices[stock])-test_output[stock][-1],0)
#     print(sum(payoffs)/5000)


#plots
"""
#re-running a simulation with more tickers to get more n1/n2 values
tickers_plot = ['SNY', 'TJX', 'STT', 'RTN', 'SAM', 'TRIP', 'DNKN', 'CVS', 'EV', 'BFAM', 'W', 'THG', 'IRM', 'AKAM', 'IRBT', 'HMHC', 'GOLF', 'ATHN', 'VOO'] 
test_plot = stock_history_finder(tickers_plot)
sim_plot = BS_sim(test_plot.get_latest_prices(), 0.0184, test_plot.get_vols(), test_plot.get_corr_mat(), 1)
output_plot = sim_plot.simulate(5000)

#display the plots - only display one at a time!
atlas_plot = AtlasPlot(tickers_plot, output_plot)
atlas_plot.plot_n1_n2_price() #3d plot of n1, n2, and price
atlas_plot.plot_strike_price() #2d plot of strike and price
###others?
"""