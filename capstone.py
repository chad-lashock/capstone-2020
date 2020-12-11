import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import calculate_sma_profit
from functions import calculate_wma_profit
from functions import calculate_ema_profit
from functions import calculate_rsi
from functions import calculate_rsi_profit_50
from functions import calculate_rsi_profit_70_30

#Load and organize data

#---CAT---
cat = pd.read_csv("./data/CAT.csv")
cat['Price'] = cat['Close']
cat = cat.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1)


#----MMM----
mmm = pd.read_csv("./data/MMM.csv")
#mmm.index = pd.to_datetime(mmm.Date)
mmm['Price'] = mmm['Close']
mmm = mmm.drop(['Open','High','Low','Close','Adj Close'], axis=1)

mmm['sma200'] = mmm['Price'].rolling(200).mean()
mmm['sma50'] = mmm['Price'].rolling(50).mean()

weights200 = np.arange(1,201)
weights50 = np.arange(1,51)
mmm['wma200'] = mmm['Price'].rolling(200).apply(lambda x: np.dot(x,weights200)/weights200.sum())
mmm['wma50'] = mmm['Price'].rolling(50).apply(lambda x : np.dot(x, weights50)/weights50.sum())

#----VZ----
vz = pd.read_csv("./data/VZ.csv")
#vz.index = pd.to_datetime(vz.Date)
vz['Price'] = vz['Close']
vz = vz.drop(['Open','High','Low','Close','Adj Close'], axis=1)

vz['ema50'] = vz['Price'].ewm(span=50, min_periods=50).mean()
vz['ema200'] = vz['Price'].ewm(span=200, min_periods=200).mean()


# Plots to show the stock prices and how some of the strategies work


#----CAT----
plt.figure(figsize = (12, 6))
plt.plot(cat['Price'], label="Price", color="red")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (12,6))
plt.plot(cat['sma200'], label='200-day', color='blue')
plt.plot(cat['sma50'], label='50-day', color='red')
plt.legend()
plt.grid()
plt.show()

#----MMM----

plt.figure(figsize = (12,8))
plt.plot(mmm['wma50'], label="50-day WMA", color="grey")
plt.plot(mmm['wma200'], label="200-day WMA", color="green")
plt.legend()
plt.grid()
plt.title("MMM Weighted Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()


#----VZ----
plt.figure(figsize = (12,8))
plt.plot(vz['ema50'], label='50-day EMA',color='grey')
plt.plot(vz['ema200'],label='200-day EMA',color='red')
plt.legend()
plt.grid()
plt.title("VZ Exponential Moving Averages")
plt.ylabel("Price ($)")
plt.xlabel("Date")
plt.show()


            
#Calculate sma profit for standard 200-day and 50-day
calculate_sma_profit(200, 50, cat)


#The following calculates the results of each CAT, MMM, and VZ for each of the 
# short-term and long-term value pairs using the simple moving average crossover
# strategy. The results are saved to a data frame.

#----CAT----
sma_results_cat = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_sma_profit(y, x, cat)
        sma_results_cat = sma_results_cat.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
sma_results_cat = sma_results_cat.sort_values(by="bank", ascending=False)

#----MMM----
sma_results_mmm = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_sma_profit(y, x, mmm)
        sma_results_mmm = sma_results_mmm.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
sma_results_mmm = sma_results_mmm.sort_values(by="bank", ascending=False)


#----VZ----

sma_results_vz = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_sma_profit(y, x, vz)
        sma_results_vz = sma_results_vz.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
sma_results_vz = sma_results_vz.sort_values(by="bank", ascending=False)

#The following calculates the results of each CAT, MMM, and VZ for each of the 
# short-term and long-term value pairs using the weighted moving average crossover
# strategy. The results are saved to a data frame.

#----CAT----
wma_results_cat = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_wma_profit(y, x, cat)
        wma_results_cat = wma_results_cat.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
wma_results_cat = wma_results_cat.sort_values(by="bank", ascending=False)

#----MMM----
wma_results_mmm = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_wma_profit(y, x, mmm)
        wma_results_mmm = wma_results_mmm.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
wma_results_mmm = wma_results_mmm.sort_values(by="bank", ascending=False)


#----VZ----

wma_results_vz = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_wma_profit(y, x, vz)
        wma_results_vz = wma_results_vz.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
wma_results_vz = wma_results_vz.sort_values(by="bank", ascending=False)

#The following calculates the results of each CAT, MMM, and VZ for each of the 
# short-term and long-term value pairs using the exponential moving average crossover
# strategy. The results are saved to a data frame.

#----CAT----
ema_results_cat = pd.DataFrame(columns=["short-period", "long-period", "bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_ema_profit(y, x, cat)
        ema_results_cat = ema_results_cat.append({"short-period":x,"long-period":y,"bank":bank}, ignore_index=True)
    
ema_results_cat = ema_results_cat.sort_values(by="bank", ascending=False)

#----MMM----
ema_results_mmm = pd.DataFrame(columns=["Short-period", "Long-period", "Bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_ema_profit(y, x, mmm)
        ema_results_mmm = ema_results_mmm.append({"Short-period":x,"Long-period":y,"Bank":bank}, ignore_index=True)
    
ema_results_mmm = ema_results_mmm.sort_values(by="Bank", ascending=False)


#----VZ----

ema_results_vz = pd.DataFrame(columns=["Short-period", "Long-period", "Bank"])

for x in range(10,201,5):
    for y in range(x+5,201,5):
        bank, result = calculate_ema_profit(y, x, vz)
        ema_results_vz = ema_results_vz.append({"Short-period":x,"Long-period":y,"Bank":bank}, ignore_index=True)
    
ema_results_vz = ema_results_vz.sort_values(by="Bank", ascending=False)

#Relative strength

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


#The following calculates the results of each CAT, MMM, and VZ  
# using the relative strength index crossover 50 strategy. 

#CAT

rsi50_results_cat = pd.DataFrame(columns=["Period", "Bank"])

for x in range(10,201,5):
    bank, result = calculate_rsi_profit_50(cat, x)
    rsi50_results_cat = rsi50_results_cat.append({"Period": x, "Bank": bank}, ignore_index=True)
    
rsi50_results_cat = rsi50_results_cat.sort_values(by="Bank",ascending=False)


#MMM

rsi50_results_mmm = pd.DataFrame(columns=["Period", "Bank"])

for x in range(10,201,5):
    bank, result = calculate_rsi_profit_50(mmm, x)
    rsi50_results_mmm = rsi50_results_mmm.append({"Period": x, "Bank": bank}, ignore_index=True)
    
rsi50_results_mmm = rsi50_results_mmm.sort_values(by="Bank",ascending=False)

#VZ

rsi50_results_vz = pd.DataFrame(columns=["Period", "Bank"])

for x in range(10,201,5):
    bank, result = calculate_rsi_profit_50(vz, x)
    rsi50_results_vz = rsi50_results_vz.append({"Period": x, "Bank": bank}, ignore_index=True)
    
rsi50_results_vz = rsi50_results_vz.sort_values(by="Bank",ascending=False)


#The following calculates the results of each CAT, MMM, and VZ  
# using the relative strength index crossover 70/30 strategy. 

#CAT

rsi7030_results_cat = pd.DataFrame(columns=["Period", "Bank"])

for x in range(10,201,5):
    bank, result = calculate_rsi_profit_70_30(cat, x)
    rsi7030_results_cat = rsi7030_results_cat.append({"Period": x, "Bank": bank}, ignore_index=True)
    
rsi7030_results_cat = rsi7030_results_cat.sort_values(by="Bank", ascending = False)


#MMM
rsi7030_results_mmm = pd.DataFrame(columns=["Period", "Bank"])

for x in range(10,201,5):
    bank, result = calculate_rsi_profit_70_30(mmm, x)
    rsi7030_results_mmm = rsi7030_results_mmm.append({"Period": x, "Bank": bank}, ignore_index=True)
    
rsi7030_results_mmm = rsi7030_results_mmm.sort_values(by="Bank", ascending = False)

#VZ
rsi7030_results_vz = pd.DataFrame(columns=["Period", "Bank"])

for x in range(10,201,5):
    bank, result = calculate_rsi_profit_70_30(vz, x)
    rsi7030_results_vz = rsi7030_results_vz.append({"Period": x, "Bank": bank}, ignore_index=True)
    
rsi7030_results_vz = rsi7030_results_vz.sort_values(by="Bank", ascending = False)




















































