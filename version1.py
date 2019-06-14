# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:34:30 2019

@author: CarrollG
"""
import pandas as pd
import pandas_datareader as pdr
from stockstats import StockDataFrame as sdf
import datetime
#import quandl
import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import csv

# Data handling section
#######################
# Get data from the file
companyData = pd.read_csv('companylist.csv', header=0, index_col='Symbol')
df_companyData = pd.DataFrame(companyData)
# Drop rows with NaN or missing data
df_companyData = df_companyData.dropna()
tickers = list(df_companyData.index)
df_stockData = pd.DataFrame()

# Set data window
start_year = 2013
start_month = 6
start_day = 12

end_year = 2019
end_month = 6
end_day = 12

#tickers = "A"
# Iterate through ticker list to grab data from Yahoo finance
allStocks_df = pd.DataFrame()
for i in tickers:
    ticker = i
    print(ticker)
    failed_tickers = []
    good_tickers = []
    try:
        # Grab from internet
        stock = pdr.get_data_yahoo(ticker,
                                  start=datetime.datetime(start_year, start_month, start_day),
                                  end=datetime.datetime(end_year, end_month, end_day))
    except:
        print("Issue getting data for ticker: ", ticker)
        # Track failed tickers for later review
        failed_tickers.append(ticker)
        # Add routine to track all failed tickers so it can be reviewed
    else: 
        good_tickers.append(ticker)
        # Rename some columns 
        stock.rename(columns={'Close':'NonAdjClose', 'Adj Close':'Close'})
        # Plot the closing prices 
        stock['Close'].plot(grid=True)
        # Show the plot
        print("Plot the closing prices ", ticker)
        plt.show()         
        # Add a column 'diff' to 'stock'
        stock['Diff'] = stock['Close'].shift(-1) - stock['Close']
        print(ticker)    
        #print("Today close: ", stock['NextClose'])
        #print("Output Diff of today close and tomorrow close: ", stock['Diff'])
        ##### RSI Routine
        # Recast pandas df to stockstats df
        stockstats_df = sdf.retype(stock)
        # Calculate RSI for 14 day lookback window and add to df
        stock['RSI']=stockstats_df['rsi_14']
        #print("RSI: ", stock['RSI'])
        ##### SMA - using fibonacci periods
        stock['SMA13']=stockstats_df['open_13_sma']
        stock['SMA21']=stockstats_df['open_21_sma']
        stock['SMA55']=stockstats_df['open_55_sma']
        stock['SMA89']=stockstats_df['open_89_sma']
        stock['SMA144']=stockstats_df['open_144_sma']
        stock['SMA233']=stockstats_df['open_233_sma']
        #print("SMA13: ", stock['SMA13'])
        #print("SMA21: ", stock['SMA21'])
        #print("SMA55: ", stock['SMA55'])
        #print("SMA89: ", stock['SMA89'])
        #print("SMA144: ", stock['SMA144'])
        #print("SMA233: ", stock['SMA233'])
        ##### EMA - using fibonacci periods
        stock['EMA13']=stockstats_df['open_13_ema']
        stock['EMA21']=stockstats_df['open_21_ema']
        stock['EMA55']=stockstats_df['open_55_ema']
        stock['EMA89']=stockstats_df['open_89_ema']
        stock['EMA144']=stockstats_df['open_144_ema']
        stock['EMA233']=stockstats_df['open_233_ema']
        #print("EMA13: ", stock['EMA13'])
        #print("EMA21: ", stock['EMA21'])
        #print("EMA55: ", stock['EMA55'])
        #print("EMA89: ", stock['EMA89'])
        #print("EMA144: ", stock['EMA144'])
        #print("EMA233: ", stock['EMA233'])
        ##### MACD
        stock['MACD']=stockstats_df['macd']
        #print("MACD: ", stock['MACD'])
        ##### KDJ stochastic oscillator - default 9 days
        stock['KDJK']=stockstats_df['kdjk']
        stock['KDJD']=stockstats_df['kdjd']
        stock['KDJJ']=stockstats_df['kdjj']
        #print("KDJK: ", stock['KDJK'])
        #print("KDJD: ", stock['KDJD'])
        #print("KDJJ: ", stock['KDJJ'])
        ##### Bollinger bands
        stock['BOLL']=stockstats_df['boll']
        stock['BOLLUB']=stockstats_df['boll_ub']
        stock['BOLLLB']=stockstats_df['boll_lb']
        #print("BOLL: ", stock['BOLL'])
        #print("BOLLUB: ", stock['BOLLUB'])
        #print("BOLLLB: ", stock['BOLLLB'])
        ##### DMI _ Diretional Movement Indicator - default 14 days
        stock['DMIP']=stockstats_df['pdi']
        stock['DMIM']=stockstats_df['mdi']
        stock['DMIX']=stockstats_df['dx']
        #print("DMIP: ", stock['DMIP'])
        #print("DMIM: ", stock['DMIM'])
        #print("DMIX: ", stock['DMIX'])
        ##### Volatility Volume Ratio - default 26 days
        stock['VVR']=stockstats_df['vr']
        #print("VVR: ", stock['VVR'])
        # Remove unneeded data from stock df created by stockstats
        del stock['close_-1_s']
        del stock['close_-1_d']
        del stock['rs_14']
        del stock['rsi_14']
        del stock['open_13_sma']
        del stock['open_21_sma']
        del stock['open_55_sma']
        del stock['open_89_sma']
        del stock['open_144_sma']
        del stock['open_233_sma']
        del stock['open_13_ema']
        del stock['open_21_ema']
        del stock['open_55_ema']
        del stock['open_89_ema']
        del stock['open_144_ema']
        del stock['open_233_ema']
        del stock['close_26_ema']
        del stock['close_12_ema']
        del stock['macd']
        del stock['macds']
        del stock['macdh']
        del stock['rsv_9']
        del stock['kdjk_9']
        del stock['kdjk']
        del stock['kdjd_9']
        del stock['kdjd']
        del stock['kdjj_9']
        del stock['kdjj']
        del stock['close_20_sma']
        del stock['close_20_mstd']
        del stock['boll']
        del stock['boll_ub']
        del stock['boll_lb']
        del stock['high_delta']
        del stock['low_delta']
        del stock['um']
        del stock['dm']
        del stock['pdm']
        del stock['pdm_14_ema']
        del stock['pdm_14']
        del stock['tr']
        del stock['atr_14']
        del stock['pdi_14']
        del stock['pdi']
        del stock['mdm']
        del stock['mdm_14_ema']
        del stock['mdm_14']
        del stock['mdi']
        del stock['mdi_14']
        del stock['dx_14']
        del stock['dx']
        del stock['dx_6_ema']
        del stock['adx_6_ema']
        del stock['adxr']
        del stock['adx']
        del stock['change']
        del stock['vr']
        del stock['open']
        del stock['high']
        del stock['low']
        del stock['adj close']
        del stock['volume']
        ##### Would like to add On-board volume
        ##### Need to add sentiment analysis
        # Create output column
        stock['NextClose'] = stock['close'].shift(-1)
        #print("OUTPUT - Tomorrow close: ", stock['NextClose'])
        # Put data in csv file in case you need it later
        print("Saving data for ", ticker, " to CSV.")
        stock.to_csv(ticker + '.csv')
        # Creste consolidated dataframe with all ticker data
        print("Stock DF header: ")
        print(stock.head(1))
        # Can toggle output
        del stock['diff']
        #del stock['nextclose']
        # Clean up some rows with no data based on time calculations
        stock.drop(stock.tail(5).index, inplace=True)
        stock.drop(stock.head(5).index, inplace=True)
        # Final dataframe
        allStocks_df = pd.concat([allStocks_df, stock], ignore_index=True)
        print("Check allStocks DataFrame")
        print(allStocks_df.head(n=20))
        # Get data from the file for DataFrame 
        try:
            df = pd.read_csv(ticker + '.csv', header=0, index_col='Date', parse_dates=True)        
        except:
            print("Failed to save data for ", ticker, " to CSV.")
        else:
            print("Success saving data for ", ticker, " to CSV.")
    # Save list of failed tickers to CSV file
    #failed_tickers_df = pd.DataFrame(failed_tickers)
    #good_tickers_df = pd.DataFrame(good_tickers)
    #failed_tickers_df.to_csv('failed_tickers.csv')
    #good_tickers_df.to_csv('good_tickers.csv')
######################### END OF DATA ROUTINE #################################

######################## START NEURAL NETWORK #################################
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf

# Make data numpy array - remove Date column - print first row
data = allStocks_df
print("allStocks DF header: ")
print(data.head(1))

# Dimensions of dataset
# Rows
n = data.shape[0]
print("n Rows = ", n)
# Columns
p = data.shape[1]
print("p Columns = ", p)

data = data.values
print("After .value: ")
print(data[0])

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
# Output y is last column
X_train = data_train[:, 0:24]
print("x_train: ")
print(X_train[0])
y_train = data_train[:, 25]
print("y_train: ")
print(y_train[0])
X_test = data_test[:, 0:24]
y_test = data_test[:, 25]

# Model architecture parameters
# Not understanging n_stocks number
n_stocks = 24
n_neurons_1 = 56
n_neurons_2 = 28
n_neurons_3 = 14
n_neurons_4 = 7
n_target = 1

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", 
                                                     distribution="uniform", 
                                                     scale=sigma)
bias_initializer = tf.zeros_initializer()

# Number of epochs and batch size
epochs = 10
batch_size = 256

# Placeholder
# The None argument indicates that at this point we do not yet know the number 
# of observations that flow through the neural net graph in each batch, so we 
# keep if flexible. We will later define the variable batch_size that controls 
# the number of observations per training batch.
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
print("Interactive Plot")
plt.show()

for e in range(epochs):
    print("Epochs: ", e)
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    print("CHECKING")
    p
    print(range(0, len(y_train) // batch_size))

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        print("Mini-batch: ", i)
        start = i * batch_size
        print("Mini-batch size: ", start)
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 5) == 0:
            print("Show progress")
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print("mse final: ")
print(mse_final)

"""
RSI (Relative Strength Index)
Avg gain or loss during a look back period
Standard 14 periods for look-back
Data needed: Close

SMA (Simple Moving Average)
Sum of stocks closing prices for the number of time periods in question divided by that period
Data needed: Close

EMA (Exponential Moving Average)
Weighted towards most recent data points

MACD (Moving Average Convergence Divergence)
emaslow, emafast, emafast - emaslow
Subtract the 26 day exponential moving average from the 12 period EMA

On-Balance Volume
Data needed:  Close and Volume

Sentiment Score

1. Script the data grab
	Get a list of tickets to loop through to grab data
2. Determine back testing window in years
3. Script the data handling/cleanup
4. Create loop to generate above calculations per stock with existing code
5. Determine locations for news articles
6. Build senitment analysis routine
7. Create loop to generate sentiment scores for stock list
8. Build neural network with 6 inputs
9. Dtermine other requirements for RNN
10.Feed the outcomes of computations into RNN inputs
"""