from pyomo.environ import *
import pandas as pd
import numpy as np
import statsmodels.api as sm


def calculate_annualized(begin, end):
    return ((end/begin)**(365/(20*365+285)))-1

# function to calculate simple moving average profit
def calculate_sma_profit(long_period, short_period, df):
    df_copy = df.copy()
    df_copy['long'] = df['Price'].rolling(long_period).mean()
    df_copy['short'] = df['Price'].rolling(short_period).mean()
    results = pd.DataFrame(columns = ['Date','Action', 'Price', 'Bank', 'Shares'])

    bank = 1000
    shares = 0
    
    for x in range(long_period, len(df_copy)):
        if df_copy['short'].loc[x-1] <= df_copy['long'].loc[x-1] and df_copy['short'].loc[x] > df_copy['long'].loc[x]:
            # buy
            if bank > 0:
                shares = bank / df['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df['Date'].loc[x], 'Action': 'Buy', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
                                
        elif df_copy['short'].loc[x-1] >= df_copy['long'].loc[x-1] and df_copy['short'].loc[x] < df_copy['long'].loc[x]:
            # sell
            if bank == 0:
                bank = shares * df['Price'].loc[x]
                shares = 0
                results = results.append({'Date':df['Date'].loc[x], 'Action': 'Sell', 'Price': df['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
    
    if shares > 0:
        bank = shares * df['Price'].loc[x]
        
    return bank, results


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
    

#Input is array of price changes
def calculate_rsi(x):
    sum_of_gains = sum([a for a in x if a > 0])
    sum_of_changes = sum([abs(a) for a in x])
    return (sum_of_gains/sum_of_changes)*100    
    
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
    
    for x in range(period+1, len(df_copy)):
        if df_copy['rsi'].loc[x-1] <= 50 and df_copy['rsi'].loc[x] > 50:
            #buy
            if bank > 0:
                shares = bank / df_copy['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df_copy['Date'].loc[x], 'Action': 'Buy', 'Price': df_copy['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
        
        elif df_copy['rsi'].loc[x-1] >= 50 and df_copy['rsi'].loc[x] < 50:
            #sell
            if bank == 0:
                bank = shares * df_copy['Price'].loc[x]
                shares = 0
                results = results.append({'Date': df_copy['Date'].loc[x], 'Action': 'Sell', 'Price': df_copy['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
        
    if shares > 0:
        bank = shares * df_copy['Price'].loc[x]
        
    return bank, results
    

def calculate_rsi_profit_70_30(df, period):
    df_copy = df.copy()
    df_copy['Price_change'] = df_copy['Price'].rolling(2).apply(lambda x : x[1]-x[0], raw=True)
    df_copy['rsi'] = df_copy['Price_change'].rolling(period).apply(calculate_rsi)
    results = pd.DataFrame(columns = ['Date','Action', 'Price', 'Bank', 'Shares'])
    
    bank = 1000
    shares = 0
        
    for x in range(period+1, len(df_copy)):
        if df_copy['rsi'].loc[x-1] <= 30 and df_copy['rsi'].loc[x] > 30:
            #buy
            if bank > 0:
                shares = bank / df_copy['Price'].loc[x]
                bank = 0
                results = results.append({'Date': df_copy['Date'].loc[x], 'Action': 'Buy', 'Price': df_copy['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
                
        elif df_copy['rsi'].loc[x-1] >= 70 and df_copy['rsi'].loc[x] < 70:
            #sell
            if bank == 0:
                bank = shares * df_copy['Price'].loc[x]
                shares = 0
                results = results.append({'Date': df_copy['Date'].loc[x], 'Action': 'Sell', 'Price': df_copy['Price'].loc[x], 'Bank': bank, 'Shares': shares}, 
                                         ignore_index=True)
                
    if shares > 0:
        bank = shares * df_copy['Price'].loc[x]
        
    return bank, results


def find_best_intervals(df,tx):
    df_copy = df.copy()
    
    model = ConcreteModel()
    model.buy = Var(range(len(df_copy)), domain=Boolean)
    model.sell = Var(range(len(df_copy)), domain=Boolean)
    
    
    model.profit = Objective(expr=sum(-1*model.buy[i]*df_copy['Price'].loc[i] + model.sell[i]*df_copy['Price'].loc[i] for i in range(len(df_copy))),sense=maximize)
    model.limits = ConstraintList()
    
    model.limits.add(sum(model.buy[i] for i in range(len(df_copy))) == tx)
    model.limits.add(sum(model.sell[i] for i in range(len(df_copy))) == tx)
    
    for x in range(len(df_copy)):
        model.limits.add(sum(model.buy[i] for i in range(x+1)) - sum(model.sell[i] for i in range(x+1)) >= 0)
        model.limits.add(sum(model.buy[i] for i in range(x+1)) - sum(model.sell[i] for i in range(x+1)) <= 1)
    
    
    solver = SolverFactory("glpk")
    solver.solve(model)
    
    
    buy_idx_list = [x for x in range(len(df_copy)) if model.buy[x]() > 0]
    sell_idx_list = [x for x in range(len(df_copy)) if model.sell[x]() > 0]
    
    
    df_copy['State'] = 0

    for x in range(len(buy_idx_list)):
        df_copy['State'].loc[buy_idx_list[x]:sell_idx_list[x]-1] = 1

        
    return df_copy    
        
        

def prepare_for_ml(df):
    df_copy = df.copy()
    
    #Calculate simple moving averages
    df_copy['sma10'] = df_copy['Price'].rolling(10).mean()
    df_copy['sma25'] = df_copy['Price'].rolling(25).mean()
    df_copy['sma50'] = df_copy['Price'].rolling(50).mean()
    
    #Calculate exponential moving averages
    df_copy['ema10'] = df_copy['Price'].ewm(span=10, min_periods=10).mean()
    df_copy['ema25'] = df_copy['Price'].ewm(span=25, min_periods=25).mean()
    df_copy['ema50'] = df_copy['Price'].ewm(span=50, min_periods=50).mean()
    
    #Calculate slopes for simple moving averages
    
    df_copy['sma10_slope'] = np.nan
    df_copy['ema10_slope'] = np.nan
    df_copy['sma25_slope'] = np.nan
    df_copy['ema25_slope'] = np.nan
    df_copy['sma50_slope'] = np.nan
    df_copy['ema50_slope'] = np.nan
    
    x = [x for x in range(10)]
    x = sm.add_constant(x)
    
    for i in range(18,len(df_copy)):
        y = list(df_copy['sma10'].loc[i-9:i])
        line = sm.OLS(y,x).fit()
        df_copy['sma10_slope'].loc[i] = line.params[1]
        
        y = list(df_copy['ema10'].loc[i-9:i])
        line = sm.OLS(y,x).fit()
        df_copy['ema10_slope'].loc[i] = line.params[1]
        
    for i in range(24,len(df_copy)):
        y = list(df_copy['sma25'].loc[i-9:i])
        line = sm.OLS(y,x).fit()
        df_copy['sma25_slope'].loc[i] = line.params[1]
        
        y = list(df_copy['ema25'].loc[i-9:i])
        line = sm.OLS(y,x).fit()
        df_copy['ema25_slope'].loc[i] = line.params[1]
        
    for i in range(49,len(df_copy)):
        y = list(df_copy['sma50'].loc[i-9:i])
        line = sm.OLS(y,x).fit()
        df_copy['sma50_slope'].loc[i] = line.params[1]
        
        y = list(df_copy['ema50'].loc[i-9:i])
        line = sm.OLS(y,x).fit()
        df_copy['ema50_slope'].loc[i] = line.params[1]
            
    
    df_copy = df_copy[58:len(df_copy)-1]
    
    
    return df_copy
    
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    