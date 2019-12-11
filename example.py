from class_defs import *

"""
Let's get all the data and graphs we need for the example in this file
"""
#example_tickers = ['SNY', 'TJX', 'STT', 'RTN', 'SAM', 'TRIP', 'DNKN', 'CVS','EV', 'BFAM', 'W', 'THG', 'IRM', 'AKAM', 'IRBT', 'HMHC','GOLF', 'SPY'] 
example_tickers = ['AAPL','AMZN','AAL','C','SPY']
example = stock_history_finder(example_tickers)
example_vols = example.get_vols()
print(example_vols)
example_corr = example.get_corr_mat()
print(example_corr)
example_init_prices = example.get_latest_prices()
print(example_init_prices)
example_paths = BS_sim(example_init_prices, 0.0184, example_vols, example_corr, 1).simulate()
example_price = AtlasOption(0.0184,0,0,1,example_paths).get_price()
print("Option price: $",round(example_price,2))
example_plot = AtlasPlot(example_tickers)
example_plot.plot_n1_n2_price()
example_plot.plot_strike_price(0,0)
