from class_defs import *

"""
Testing Stock History finder
"""
tickers = ['AAPL','MSFT', 'F', 'AMZN','VOO']
#tickers = ['AAPL','TSLA','BND','VOO']
test = stock_history_finder(tickers)
vols = test.get_vols()
print("Volatilities")
print(vols)
corr_mat = test.get_corr_mat()
print("Correlation matrix")
print(corr_mat)
latest_prices = test.get_latest_prices()
print("Current prices")
print(latest_prices)


"""
Testing BS_sim 
"""
test_sim = BS_sim(latest_prices, 0.0184 , vols ,corr_mat, 1)
test_paths = test_sim.simulate()

# plt.plot(test_paths[3]) #looking at plot of paths of AMZN
# plt.show() 




"""
Testing out simple call prices for these stocks -  comparing to theoretical BS price
They are all very similar (used https://www.erieri.com/blackscholes to compare)
"""
def call_and_put_px(tickers, paths, latest_prices):
    for i in range(len(tickers)):
        px_at_mat = paths[i][-1]
        payoffs_call = np.maximum(px_at_mat - float(latest_prices[i]), 0)
        px_call = np.exp(-0.0184*1) * payoffs_call
        payoffs_put = np.maximum(-px_at_mat + float(latest_prices[i]), 0)
        px_put = np.exp(-0.0184*1) * payoffs_put
        print(tickers[i],"| call:",np.average(px_call), "| put:", np.average(px_put))

call_and_put_px(tickers, test_paths, latest_prices)

"""
Setting maturity to 0 and vols to 0
"""
test_sim_0_mat = BS_sim(latest_prices, 0.0184 , vols ,corr_mat, 0)
test_paths_0_mat = test_sim_0_mat.simulate()
call_and_put_px(tickers, test_paths_0_mat, latest_prices)

test_sim_0_vol = BS_sim(latest_prices, 0.0184 , len(latest_prices)*[0] ,corr_mat, 1)
test_paths_0_vol = test_sim_0_mat.simulate()
call_and_put_px(tickers, test_paths_0_vol, latest_prices)
# test_atlas = AtlasOption(0,0,1,test_output)
# print(test_atlas.get_price())