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
        if prices.isnull().values.any() == True:
            prices.ffill()
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
        self.steps = int(252*maturity)+1
        self.vols = vols
    
    def simulate(self, sim_number = 10000):
        
        n_stocks = len(self.initial_stock_prices)

        if self.maturity == 0:
            """ If maturity is 0, then it just returns the inital prices"""
            prices = np.zeros((n_stocks,1,sim_number))
            for stock in range(n_stocks):
                prices[stock][0] = self.initial_stock_prices[stock]
            return prices
        
        dt = self.maturity/(self.steps-1) 
        prices = np.zeros((n_stocks,self.steps,sim_number))  #np array of zeros where the simulated prices will go
        # structure of prices is the following: 
        # prices[i] gives you the price paths of the stock at index i
        # the next level, prices[i][j] gives all the simulated prices of the stock at index i at time j (so if you want the prices at maturity, type prices[i][-1]) 
        
        for stock in range(n_stocks):
            prices[stock][0] = self.initial_stock_prices[stock]  #populating with the initial stock prices
        
        for time in range(1,self.steps):
            random_variables = np.random.multivariate_normal(n_stocks*[0], self.corr_mat, sim_number) #generating correlated random variables
            for stock in range(n_stocks):
                #prices[stock][time] = prices[stock][time-1] + self.rf_rate*prices[stock][time-1]*dt + self.vols[stock]*prices[stock][time-1]*np.sqrt(dt)*random_variables[:,stock]
                prices[stock][time] = prices[stock][time-1] * np.exp((self.rf_rate - ((self.vols[stock]**2)/2) ) * dt + self.vols[stock]*np.sqrt(dt)* random_variables[:,stock])
        
        return prices


class AtlasOption:

    def __init__(self, rf_rate, n1, n2, strike, paths):
        self.paths = paths
        self.index_prices = paths[-1][-1]
        self.stock_prices = paths[:-1]
        self.rf = rf_rate
        self.n1 = n1
        self.n2 = n2
        self.strike = strike
        self.maturity = (len((paths[0].transpose())[0])-1)/252
    
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
        
        payoffs = np.maximum(payoff_term_1, np.zeros(len(payoff_term_1))) #taking the max of 0 and the other payoff term
        
        payoffs_dollars = payoffs * self.index_prices #converting perecentage payoffs to dollar amounts
        
        discounted_payoffs = np.exp(-self.rf*self.maturity) * payoffs_dollars
        avg_disc_payoff = np.average(discounted_payoffs)
        
        return avg_disc_payoff

class AtlasGreeks:

    def __init__(self,tickers,rf_rate,n1,n2,strike,maturity):
        self.tickers = tickers
        self.rf_rate = rf_rate
        self.n1 = n1
        self.n2 = n2
        self.strike = strike
        self.maturity = maturity
        self.stock_history = stock_history_finder(tickers)
        self.vols = self.stock_history.get_vols()
        self.corr = self.stock_history.get_corr_mat()
        self.initial_prices = self.stock_history.get_latest_prices()
    
    def get_index_delta(self):
        sim_object_1 = BS_sim(self.initial_prices,self.rf_rate,self.vols,self.corr,self.maturity)
        sim_1 = sim_object_1.simulate(sim_number = 20000)
        atlas_object_1 = AtlasOption(self.rf_rate, self.n1,self.n2,self.strike,sim_1)
        price_1 = atlas_object_1.get_price()
        
        deltas = []

        for h in np.arange(0.5,10,0.5):
            new_initial_prices = np.append(self.initial_prices[:-1], self.initial_prices[-1] + h)
            sim_object_2 = BS_sim(new_initial_prices,self.rf_rate,self.vols,self.corr,self.maturity)
            sim_2 = sim_object_2.simulate(sim_number = 20000)
            atlas_object_2 = AtlasOption(self.rf_rate, self.n1,self.n2,self.strike,sim_2)
            price_2 = atlas_object_2.get_price()
            
            delta = (price_2 - price_1)/h
            deltas.append(delta)
        return np.average(deltas)
    
    def get_index_vega(self):
        
        sim_object_1 = BS_sim(self.initial_prices,self.rf_rate,self.vols,self.corr,self.maturity)
        sim_1 = sim_object_1.simulate(sim_number = 20000)
        atlas_object_1 = AtlasOption(self.rf_rate, self.n1,self.n2,self.strike,sim_1)
        price_1 = atlas_object_1.get_price()

        vegas = []

        for h in np.arange(0.002,0.04,0.002):
            new_vols = np.append(self.vols[:-1], self.vols[-1] + h)
            sim_object_2 = BS_sim(self.initial_prices,self.rf_rate,new_vols,self.corr,self.maturity)
            sim_2 = sim_object_2.simulate(sim_number = 20000)
            atlas_object_2 = AtlasOption(self.rf_rate, self.n1,self.n2,self.strike,sim_2)
            price_2 = atlas_object_2.get_price()
            
            vega = (price_2 - price_1)/h
            vegas.append(vega)
        
        return np.average(vega)/100

    def get_index_theta(self):
        
        sim_object_1 = BS_sim(self.initial_prices,self.rf_rate,self.vols,self.corr,self.maturity)
        sim_1 = sim_object_1.simulate(sim_number = 20000)
        atlas_object_1 = AtlasOption(self.rf_rate, self.n1,self.n2,self.strike,sim_1)
        price_1 = atlas_object_1.get_price()

        thetas = []

        for h in np.arange(12/252,180/252,21/252):
            new_mat = self.maturity + h
            sim_object_2 = BS_sim(self.initial_prices,self.rf_rate,self.vols,self.corr,new_mat)
            sim_2 = sim_object_2.simulate(sim_number = 20000)
            atlas_object_2 = AtlasOption(self.rf_rate, self.n1,self.n2,self.strike,sim_2)
            price_2 = atlas_object_2.get_price()
            
            theta = (price_2 - price_1)/h
            thetas.append(theta)
        
        return -np.average(thetas)/252


class AtlasPlot:
    
    def __init__(self, ticker_list):
        self.tickers = ticker_list
        self.history = stock_history_finder(self.tickers)
        
    def plot_n1_n2_price(self): #3d plot of n1, n2, and price
        
        #first, we need a simulation of the tickers
        sim_plot = BS_sim(self.history.get_latest_prices(), 0.0184, self.history.get_vols(), self.history.get_corr_mat(), 1)
        output_plot = sim_plot.simulate(10000)
        #n1 and n2 are the x and y axes
        x = range(len(self.tickers))
        y = range(len(self.tickers)) 
        #price is the z axis
        z = np.zeros((len(x),len(y)))
        #z is a two dimensional array of prices
        for i in range(len(x)):
            for j in range(len(y)):
                if i+j < (len(self.tickers)-1): #we need n1+n2 < len(tickers)
                    atlas_option_ij = AtlasOption(0.0184,i,j,1,output_plot)
                    z[i][j] = atlas_option_ij.get_price()
                else:
                    z[i][j] = None
        
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
        
    def plot_strike_price(self,n1,n2): #2d plot of strike and price
        
        #first, we need a simulation of the tickers
        sim_plot = BS_sim(self.history.get_latest_prices(), 0.0184, self.history.get_vols(), self.history.get_corr_mat(), 1)
        output_plot = sim_plot.simulate(5000)
        #strike is the x axis, price is the y axis
        x = np.linspace(0.5,1.5,20) #using a bunch of different arbitrary values for strike
        y = np.zeros(len(x))
        for i in range(len(x)):
            atlas_option_i = AtlasOption(0.0184,n1,n2,x[i],output_plot) #arbitrarily picked n1=n2=5
            y[i] = atlas_option_i.get_price()
            
        #plot commands
        plt.xlabel('strike')
        plt.ylabel('price')
        plt.title('Relationship of strike to price')
        plt.plot(x,y)
        plt.show()
        
    def plot_maturity_price(self,n1,n2): #2d plot of maturity and price
        
        #in order to plot price against maturity, we need a bunch of different simulations
        #arbitrarily picking 0.5,1,2,5,10,30 year maturities
        maturities = [0.5, 1, 2, 5, 10, 30]
        
        #maturity is the x axis, price is the y axis
        x = np.linspace(0,30,6)
        y = np.zeros(len(x))
        for i in range(len(x)):
            sim_plot = BS_sim(self.history.get_latest_prices(), 0.0184, self.history.get_vols(), self.history.get_corr_mat(), maturities[i])
            output_plot = sim_plot.simulate(5000)
            atlas_option_i = AtlasOption(0.0184,n1,n2,1,output_plot) #arbitrarily picked n1=n2=5
            y[i] = atlas_option_i.get_price()
        
        #plot commands
        plt.xlabel('maturity')
        plt.ylabel('price')
        plt.title('Relationship of maturity to price')
        plt.plot(x,y)
        plt.show()


test_sim = BS_sim([100,150,100], 0.0184, [0.25,0.25,0.1], [[1,0,0],[0,1,0],[0,0,1]], 1).simulate(sim_number = 40000)
test_sim_1 = BS_sim([100,150,100], 0.0184, [0.25,0.25,0.1], [[1,0.5,0],[0.5,1,0],[0,0,1]], 1).simulate(sim_number = 40000)
test_sim_2 = BS_sim([100,150,100], 0.0184, [0.25,0.25,0.1], [[1,-0.5,0],[-0.5,1,0],[0,0,1]], 1).simulate(sim_number = 40000)
test_price = AtlasOption(0.0184,0,0,1,test_sim).get_price()
test_price_1 = AtlasOption(0.0184,0,0,1,test_sim_1).get_price()
test_price_2 = AtlasOption(0.0184,0,0,1,test_sim_2).get_price()
print(test_price)
print(test_price_1)
print(test_price_2)

#plots

#new ticker list to get more n1/n2 values
#Update: I moved the simulations inside the AtlasPlot class
#in order for the maturity plot to be able to handle multiple simulations
# tickers_plot = ['SNY', 'TJX', 'STT', 'RTN', 'SAM', 'TRIP', 'DNKN', 'CVS', 'EV', 'BFAM', 'W', 'THG', 'IRM', 'AKAM', 'IRBT', 'HMHC', 'GOLF', 'ATHN', 'VOO'] 

#display the plots - only display one at a time!
# atlas_plot = AtlasPlot(tickers_plot)
#atlas_plot.plot_n1_n2_price() #3d plot of n1, n2, and price
#atlas_plot.plot_strike_price() #2d plot of strike and price
# atlas_plot.plot_maturity_price() #2d plot of maturity and price