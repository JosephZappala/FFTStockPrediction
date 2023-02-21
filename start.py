import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy import optimize, interpolate
from datetime import datetime



msft = yf.Ticker("AMC")
hist = msft.history(period="6hr", interval="1m")
storedPrices = np.array(hist['Close'][-50:])
times = np.array(range(0, len(hist)-50))
prices = np.array(hist['Close'][:-50])

def func1(x, a, b, c, d):
    return a*x**3+b*x**2+c*x + d


params, _ = optimize.curve_fit(func1, times, prices)
a, b, c, d = params[0], params[1], params[2], params[3]
yfit = a*times**3+b*times**2+c*times + d


plt.figure(figsize = (10,8))
# print(prices)
out = fft(prices)
indices = np.abs(out) > 2
out_clean = indices * out
new_prices_clean = ifft(out_clean)

times = times
prices = prices


V_spline = np.linspace(max(times), min(times))
spline = interpolate.make_interp_spline(times, new_prices_clean, k=2 )
pricesSplined = spline(V_spline)
#yfit += abs(pricesSplined)
postTimes = np.array(range(times[-1],times[-1] + 100, 2))
yfitAfter = a*postTimes**3+b*postTimes**2+c*postTimes + d


plt.plot(times, new_prices_clean, 'b')
plt.plot(times, prices, 'g')
plt.plot(times, yfit, 'r')
plt.plot(postTimes, pricesSplined, 'y')
plt.plot(postTimes, storedPrices, 'g')
plt.plot(postTimes, yfitAfter, 'r')



plt.show()

