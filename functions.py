from pyomo.environ import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split


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
    df_copy.index = [x for x in range(len(df_copy))]
    
    
    return df_copy


def plot_best_intervals(df):
    line_list = []
    colors = []
    line_size = []
    
    x = 0
    
    while x < len(df):
        
        state = df['State'].loc[x]
        line = []
        
        if state == 0:
            colors.append(mcolors.to_rgb("black"))
            line_size.append(0.5)
        else:
            colors.append(mcolors.to_rgb("green"))
            line_size.append(1.0)
            
        while  x < len(df) and df['State'].loc[x] == state:
            line.append((df.index[x],df['Price'].loc[x]))
            x = x + 1
            
        line_list.append(line)
        
    
    
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()
    ax.set_facecolor('whitesmoke')
    line_collection = LineCollection(line_list, linewidths=line_size,colors=colors)
    ax.add_collection(line_collection)
    ax.autoscale()
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks([0,500,1004,1508,2010,2515,3019,3521,4025,4528,5031],[y for y in range(2000,2021,2)])
    plt.show()
    
    
def calculate_interval_profit(df):
    bank = 1000
    shares = 0
    state = 0
    
    for x in range(len(df)):
        if state == 0 and df['State'].loc[x] == 1:
            #buy
            if bank > 0:
                shares = bank / df['Price'].loc[x]
                bank = 0
        elif state == 1 and df['State'].loc[x] == 0:
            #sell
            if bank == 0:
                bank = shares * df['Price'].loc[x]
                shares = 0
                
        state = df['State'].loc[x]
                
    if shares > 0:
        bank = shares * df['Price'].loc[x]
        
    return bank

def prepare_for_svm_sequentially(df):
    df_copy = prepare_for_ml(df)
    
    df_train = df_copy[:4136]
    df_test = df_copy[4136:]
    df_test.index = [i for i in range(len(df_test))]
    
    df_train = find_best_intervals(df_train, 64)
    df_test = find_best_intervals(df_test, 16)
    
    df_train['sma10_scaled'] = pp.scale(df_train['sma10'])
    df_train['sma25_scaled'] = pp.scale(df_train['sma25'])
    df_train['sma50_scaled'] = pp.scale(df_train['sma50'])
    df_train['ema10_scaled'] = pp.scale(df_train['ema10'])
    df_train['ema25_scaled'] = pp.scale(df_train['ema25'])
    df_train['ema50_scaled'] = pp.scale(df_train['ema50'])
    
    df_test['sma10_scaled'] = df_test['sma10'].apply(lambda x : (x-df_train['sma10'].mean())/df_train['sma10'].std())
    df_test['sma25_scaled'] = df_test['sma25'].apply(lambda x : (x-df_train['sma25'].mean())/df_train['sma25'].std())
    df_test['sma50_scaled'] = df_test['sma50'].apply(lambda x : (x-df_train['sma50'].mean())/df_train['sma50'].std())
    df_test['ema10_scaled'] = df_test['ema10'].apply(lambda x : (x-df_train['ema10'].mean())/df_train['ema10'].std())
    df_test['ema25_scaled'] = df_test['ema25'].apply(lambda x : (x-df_train['ema25'].mean())/df_train['ema25'].std())
    df_test['ema50_scaled'] = df_test['ema50'].apply(lambda x : (x-df_train['ema50'].mean())/df_train['ema50'].std())
    
    x_train = np.array(df_train.drop(['Date','Price','State', 'sma10', 'sma25', 'sma50', 'ema10', 'ema25', 'ema50'],axis=1))
    y_train = np.array(df_train['State'])
    
    x_test = np.array(df_test.drop(['Date','Price','State', 'sma10', 'sma25', 'sma50', 'ema10', 'ema25', 'ema50'],axis=1))
    y_test = np.array(df_test['State'])
    
    return x_train, x_test, y_train, y_test

def prepare_for_svm_random(df):
    df_svm = prepare_for_ml(df)

    x = np.array(df_svm.drop(['Date','Price','State'],axis=1))
    y = np.array(df_svm['State'])
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)
    
    for i in range(0,6):
        x_test[:,i] = (x_test[:,i]-np.mean(x_train[:,i]))/np.std(x_train[:,i])
        
    for i in range(0,6):
        x_train[:,i] = pp.scale(x_train[:,i])
        
    return x_train, x_test, y_train, y_test
    
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    