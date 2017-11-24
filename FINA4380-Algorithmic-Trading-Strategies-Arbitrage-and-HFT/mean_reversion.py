"""
  Python 3 script for Group Asg of FINA4380.
  The Script does following:
    1. used rolling ordinary least squares regression method to compute beta.
    2. by the concept mean-reseversion, long an asset X and short another one y with ratio beta to do pair trading.
  In this asg, Cryptocurrency Bitcoin, Ethereum and Litecoin had been considered.
"""

# Import:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import statsmodels.api as sm

symbols = ['ltc', 'btc'] # 'eth', 'btc', 'lte'

def read_data(symbols=symbols):
  path0 = 'Data Bitfinex {0}USD 1t.csv'.format(symbols[0].upper())
  print('  Reading CSV data from %s...' % path0)
  df0 = pd.read_csv(path0, index_col=0)

  path1 = 'Data Bitfinex {0}USD 1t.csv'.format(symbols[1].upper())
  print('  Reading CSV data from %s...' % path1)
  df1 = pd.read_csv(path1, index_col=0)

  # create a pd dataframe with close price of each cryptocurrency
  # correctly aligend and dropping missing rows
  print('  Constructing Dual Matrix for {0} and {1}...'.format(symbols[0].upper(), symbols[1].upper()))
  pairs = pd.DataFrame(index=df1.index)
  pairs['{0}_close'.format(symbols[0])] = df0['close']
  pairs['{0}_close'.format(symbols[1])] = df1['close']
  pairs = pairs.dropna()
  pairs.index.name = 'date'
  return pairs

def calculate_spread_zscore(pairs, symbols=symbols, window=100):
  sym1 = '{0}_close'.format(symbols[0])
  sym2 = '{0}_close'.format(symbols[1])
  a = np.array([np.nan] * len(pairs))
  b = [np.nan] * len(pairs)
  y_ = pairs[sym1].values
  x_ = pairs[[sym2]].assign(constant=1).values

  print('  Calculating Betas by Rolling Linear Regression...')
  for n in range(window, len(pairs)):
    y = y_[(n-window):n]
    X = x_[(n-window):n]
    # betas = Inverse(X'.X).X'.y
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_hat = betas.dot(x_[n, :])
    a[n] = y_hat
    b[n] = betas.tolist()
    if n % 100000 == 0:
      print('  Now is reading %d-th lines...' % n)
  pairs['hedge_ratio'] = [i[0] if isinstance(i, list) else np.nan for i in b]
  pairs = pairs.dropna()
  print('  Finish Rolling Linear Regression and Now is computing Spread and Z Score...')
  pairs['spread'] = pairs[sym1] - pairs['hedge_ratio'] * pairs[sym2]
  pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread'])) / np.std(pairs['spread'])
  print('  Generating Spread by Kalmen Filter......')
  return pairs

def create_long_short_market_signals(pairs, z_entry_threshold=2.0, z_exit_threshold=1.0):
    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    # These signals are needed because we need to propagate a
    # position forward, i.e. we need to stay long if the zscore
    # threshold is less than z_entry_threshold by still greater
    # than z_exit_threshold, and vice versa for shorts.
    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    # These variables track whether to be long or short while
    # iterating through the bars
    long_market = 0
    short_market = 0

    print("Calculating when to be in the market (long and short)...")
    for i, b in enumerate(pairs.iterrows()):        
        if (i+1) % 100000 == 0:
          print('  Now is reading %d th row...' % (i+1))

        # Calculate longs
        if b[1]['longs'] == 1.0:
            long_market = 1            
        # Calculate shorts
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calculate exists
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0
        # This directly assigns a 1 or 0 to the long_market/short_market
        # columns, such that the strategy knows when to actually stay in!
        pairs.ix[i]['long_market'] = long_market
        pairs.ix[i]['short_market'] = short_market

    return pairs

def create_portfolio_returns(pairs, symbols=symbols):
    # Construct the portfolio object with positions information
    # Note that minuses to keep track of shorts!
    sym1 = '{0}_close'.format(symbols[0])
    sym2 = '{0}_close'.format(symbols[1])

    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio['sym1'] = -1.0 * pairs[sym1] * portfolio['positions']
    portfolio['sym2'] = pairs[sym2] * portfolio['positions']
    portfolio['total'] = portfolio['sym1'] + portfolio['sym2']

    # Construct a percentage returns stream and eliminate all 
    # of the NaN and -inf/+inf cells
    print("Constructing the equity curve...")
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calculate the full equity curve
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio
