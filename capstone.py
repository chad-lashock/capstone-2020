import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cat = pd.read_csv("./data/CAT.csv")
cat.index = pd.to_datetime(cat.Date)
cat['Price'] = cat['Close']
cat = cat.drop(['Open','High','Low','Close','Adj Close'], axis=1)

cat['sma200'] = cat['Price'].rolling(200).mean()
cat['sma50'] = cat['Price'].rolling(50).mean()

plt.figure(figsize = (12, 6))
plt.plot(cat['Price'], label="Price", color="red")
plt.legend()
plt.grid()
plt.show()


def calculate_sma_profit(long_period, short_period, df):
    long_ma = list(df['Price'].rolling(long_period).mean())
    short_ma = list(df['Price'].rolling(short_period).mean())
    
    bank = 1000
    
    if short_ma[long_period-1] < long_ma[long_period-1]:
        short_ma_under = True
    else:
        short_ma_under = False
    
    for x in range(long_period, len(long_ma)):
        if short_ma_under == True and short_ma[x] > long_ma[x]:
            # buy
            short_ma_under = False
        elif short_ma_under == False and short_ma[x] < long_ma[x]:
            # sell
            short_ma_under = True
            




mmm = pd.read_csv("./data/MMM.csv")
mmm.index = pd.to_datetime(mmm.Date)
mmm['Price'] = mmm['Close']
mmm = mmm.drop(['Open','High','Low','Close','Adj Close'], axis=1)

mmm['sma200'] = mmm['Price'].rolling(200).mean()
mmm['sma50'] = mmm['Price'].rolling(50).mean()

weights200 = np.arange(1,201)
weights50 = np.arange(1,51)
mmm['wma200'] = mmm['Price'].rolling(200).apply(lambda x: np.dot(x,weights200)/weights200.sum())
mmm['wma50'] = mmm['Price'].rolling(50).apply(lambda x : np.dot(x, weights50)/weights50.sum())

plt.figure(figsize = (12,8))
plt.plot(mmm['wma50'], label="50-day WMA", color="grey")
plt.plot(mmm['wma200'], label="200-day WMA", color="green")
plt.legend()
plt.grid()
plt.title("MMM Weighted Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

vz = pd.read_csv("./data/VZ.csv")
vz.index = pd.to_datetime(vz.Date)
vz['Price'] = vz['Close']
vz = vz.drop(['Open','High','Low','Close','Adj Close'], axis=1)

vz['ema50'] = vz['Price'].ewm(span=50, min_periods=50).mean()
vz['ema200'] = vz['Price'].ewm(span=200, min_periods=200).mean()

plt.figure(figsize = (12,8))
plt.plot(vz['ema50'], label='50-day EMA',color='grey')
plt.plot(vz['ema200'],label='200-day EMA',color='red')
plt.legend()
plt.grid()
plt.title("VZ Exponential Moving Averages")
plt.ylabel("Price ($)")
plt.xlabel("Date")
plt.show()


#Relative strength

#Input is array of price changes
def calculate_rsi(x):
    sum_of_gains = sum([a for a in x if a > 0])
    sum_of_changes = sum([abs(a) for a in x])
    return (sum_of_gains/sum_of_changes)*100

cat['Change'] = cat['Price'].rolling(2).apply(lambda x : x[1]-x[0])
cat['rsi30'] = cat['Change'].rolling(30).apply(calculate_rsi)

plt.figure(figsize=(12,8))
plt.plot(cat['rsi30'].loc['2020-01-01':'2020-03-31'], label="30-day RSI", color="blue")
#plt.xticks(['2020-01-01','2020-02-01', '2020-03-01', '2020-04-01'],['Jan 1','Feb 1','Mar 1', 'Apr 1'])
plt.legend()
plt.grid()
plt.title("CAT RSI Quarter 1 of 2020")
plt.ylabel("RSI")
plt.xlabel("Date")
plt.ylim(bottom=20, top=80)
plt.show()































