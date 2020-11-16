import pandas as pd
import numpy as np

def calculate_annualized(begin, end):
    return ((end/begin)**(365/(20*365+285)))-1

# function to calculate simple moving average profit
def calculate_sma_profit(long_period, short_period, df):
    long_ma = list(df['Price'].rolling(long_period).mean())
    short_ma = list(df['Price'].rolling(short_period).mean())
    results = pd.DataFrame(columns = ['Date','Action', 'Price', 'Bank', 'Shares'])

    
    bank = 1000
    shares = 0
    
    if short_ma[long_period-1] < long_ma[long_period-1]:
        short_ma_under = True
    else:
        short_ma_under = False
    
    for x in range(long_period, len(long_ma)):
        if short_ma_under == True and short_ma[x] > long_ma[x]:
            # buy
            short_ma_under = False
            if bank > 0:
                shares = bank / df['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df['Date'].loc[x], 'Action': 'Buy', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
                
                
        elif short_ma_under == False and short_ma[x] < long_ma[x]:
            # sell
            short_ma_under = True
            if bank == 0:
                bank = shares * df['Price'].loc[x]
                shares = 0
                results = results.append({'Date':df['Date'].loc[x], 'Action': 'Sell', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
    
    
    if shares > 0:
        bank = shares * df['Price'].loc[x]
        
    return bank, results


#Input is array of price changes
def calculate_rsi(x):
    sum_of_gains = sum([a for a in x if a > 0])
    sum_of_changes = sum([abs(a) for a in x])
    return (sum_of_gains/sum_of_changes)*100

# function to calculate weighted moving average profit
def calculate_wma_profit(long_period, short_period, df):
    weights_long = np.arange(1,long_period+1)
    weights_short = np.arange(1,short_period+1)
    long_ma = list(df['Price'].rolling(long_period).apply(lambda x: np.dot(x,weights_long)/weights_long.sum()))
    short_ma = list(df['Price'].rolling(short_period).apply(lambda x : np.dot(x, weights_short)/weights_short.sum()))
    results = pd.DataFrame(columns = ['Date','Action', 'Price', 'Bank', 'Shares'])
    
    bank = 1000
    shares = 0
    
    if short_ma[long_period-1] < long_ma[long_period-1]:
        short_ma_under = True
    else:
        short_ma_under = False
    
    for x in range(long_period, len(long_ma)):
        if short_ma_under == True and short_ma[x] > long_ma[x]:
            # buy
            short_ma_under = False
            if bank > 0:
                shares = bank / df['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df['Date'].loc[x], 'Action': 'Buy', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
                
                
        elif short_ma_under == False and short_ma[x] < long_ma[x]:
            # sell
            short_ma_under = True
            if bank == 0:
                bank = shares * df['Price'].loc[x]
                shares = 0
                results = results.append({'Date':df['Date'].loc[x], 'Action': 'Sell', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
    
    
    if shares > 0:
        bank = shares * df['Price'].loc[x]
        
    return bank, results



# function to calculate exponential moving average profit
def calculate_ema_profit(long_period, short_period, df):
    long_ma = list(df['Price'].ewm(span=long_period, min_periods=long_period).mean())
    short_ma = list(df['Price'].ewm(span=short_period, min_periods=short_period).mean())
    results = pd.DataFrame(columns = ['Date','Action', 'Price', 'Bank', 'Shares'])

    
    bank = 1000
    shares = 0
    
    if short_ma[long_period-1] < long_ma[long_period-1]:
        short_ma_under = True
    else:
        short_ma_under = False
    
    for x in range(long_period, len(long_ma)):
        if short_ma_under == True and short_ma[x] > long_ma[x]:
            # buy
            short_ma_under = False
            if bank > 0:
                shares = bank / df['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df['Date'].loc[x], 'Action': 'Buy', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
                
                
        elif short_ma_under == False and short_ma[x] < long_ma[x]:
            # sell
            short_ma_under = True
            if bank == 0:
                bank = shares * df['Price'].loc[x]
                shares = 0
                results = results.append({'Date':df['Date'].loc[x], 'Action': 'Sell', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
    
    
    if shares > 0:
        bank = shares * df['Price'].loc[x]
        
    return bank, results
    
    
    
#This method calculates the profit using the RSI crossover 50 strategy
# Input is a dataframe of prices and a period that represents the 
# number of days to use to calculate the RSI value
def calculate_rsi_profit_50(df, period):
    df_copy = df.copy()
    df_copy['Price_change'] = df_copy['Price'].rolling(2).apply(lambda x : x[1]-x[0], raw=True)
    df_copy['rsi'] = df_copy['Price_change'].rolling(period).apply(calculate_rsi)
    results = pd.DataFrame(columns = ['Date','Action', 'Price', 'Bank', 'Shares'])
    
    bank = 1000
    shares = 0
    
    if df_copy['rsi'].loc[period] < 50:
        below_50 = True
    else:
        below_50 = False
    
    for x in range(period+1, len(df_copy)):
        if below_50 == True and df_copy['rsi'].loc[x] > 50:
            #buy
            below_50 = False
            if bank > 0:
                shares = bank / df_copy['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df_copy['Date'].loc[x], 'Action': 'Buy', 'Price': df_copy['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
        
        elif below_50 == False and df_copy['rsi'].loc[x] < 50:
            #sell
            below_50 = True
            if bank == 0:
                bank = shares * df_copy['Price'].loc[x]
                shares = 0
                results = results.append({'Date': df_copy['Date'].loc[x], 'Action': 'Sell', 'Price': df_copy['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
        
    if shares > 0:
        bank = shares * df_copy['Price'].loc[x]
        
    return bank, results
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    