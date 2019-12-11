from class_defs import *

"""
Testing Stock History finder
"""
tickers = ['AAPL','MSFT', 'F', 'AMZN','VOO']
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
Testing out simple at the money call and put prices for these stocks -  comparing to theoretical BS price
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
Setting maturity to 0 and vols to 0. Should see put and call prices go to 0.
"""
print("Should see 0s")
test_sim_0_mat = BS_sim(latest_prices, 0.0184 , vols ,corr_mat, 0)
test_paths_0_mat = test_sim_0_mat.simulate()
call_and_put_px(tickers, test_paths_0_mat, latest_prices)

print("Should see 0s for puts, small price for calls")
test_sim_0_vol = BS_sim(latest_prices, 0.0184 , len(latest_prices)*[0] ,corr_mat, 1)
test_paths_0_vol = test_sim_0_vol.simulate()
call_and_put_px(tickers, test_paths_0_vol, latest_prices)

print("Should see 0s")
test_sim_0_ret_vol = BS_sim(latest_prices, 0, len(latest_prices)*[0] ,corr_mat, 1)
test_paths_0_ret_vol = test_sim_0_ret_vol.simulate()
call_and_put_px(tickers, test_paths_0_ret_vol, latest_prices)

"""
Testing atlas prices now that we know the simulation class is working
"""
print("Should see decreasing numbers:")
print(AtlasOption(0,0,1,test_paths).get_price())
print(AtlasOption(1,0,1,test_paths).get_price())
print(AtlasOption(2,0,1,test_paths).get_price())
print(AtlasOption(3,0,1,test_paths).get_price())
print("Should see increasing numbers:")
print(AtlasOption(0,0,1,test_paths).get_price())
print(AtlasOption(0,1,1,test_paths).get_price())
print(AtlasOption(0,2,1,test_paths).get_price())
print(AtlasOption(0,3,1,test_paths).get_price())
print("Should see 0")
print(AtlasOption(0,0,1,test_paths_0_mat).get_price())
print("Should see approximately 5.37 = 289.19 * (np.exp(0.0184)-1)")
print(AtlasOption(0,0,1,test_paths_0_vol).get_price())
print("Should see price of the index")
print(AtlasOption(0,0,0,test_paths_0_mat).get_price())
print("Should see 294.56 = 289.19 * np.exp(0.0184)")
print(AtlasOption(0,0,0,test_paths_0_vol).get_price())
