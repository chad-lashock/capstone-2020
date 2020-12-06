from pyomo.environ import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


cat = pd.read_csv("./data/CAT.csv")
cat['Price'] = cat['Close']
cat = cat.drop(['Open','High','Low','Close','Adj Close', 'Volume'], axis=1)


##------Initial exploration and plotting--------------




##------------Computing best intervals------------
model = ConcreteModel()
model.buy = Var(range(len(cat)), domain=Boolean)
model.sell = Var(range(len(cat)), domain=Boolean)


model.profit = Objective(expr=sum(-1*model.buy[i]*cat['Price'].loc[i] + model.sell[i]*cat['Price'].loc[i] for i in range(len(cat))),sense=maximize)
model.limits = ConstraintList()

tx = 80 #Number of allowed transactions

model.limits.add(sum(model.buy[i] for i in range(len(cat))) == tx)
model.limits.add(sum(model.sell[i] for i in range(len(cat))) == tx)

for x in range(len(cat)):
    model.limits.add(sum(model.buy[i] for i in range(x+1)) - sum(model.sell[i] for i in range(x+1)) >= 0)
    model.limits.add(sum(model.buy[i] for i in range(x+1)) - sum(model.sell[i] for i in range(x+1)) <= 1)


solver = SolverFactory("glpk")
solver.solve(model)


buy_idx_list = [x for x in range(len(cat)) if model.buy[x]() > 0]
sell_idx_list = [x for x in range(len(cat)) if model.sell[x]() > 0]


##------------Manipulate data for modeling-------------

cat['Action'] = 'Hold'
cat['Action'].loc[buy_idx_list] = 'Buy'
cat['Action'].loc[sell_idx_list] = 'Sell'

#Calculate simple moving averages
cat['sma10'] = cat['Price'].rolling(10).mean()
cat['sma25'] = cat['Price'].rolling(25).mean()
cat['sma50'] = cat['Price'].rolling(50).mean()

#Calculate exponential moving averages
cat['ema10'] = cat['Price'].ewm(span=10, min_periods=10).mean()
cat['ema25'] = cat['Price'].ewm(span=25, min_periods=25).mean()
cat['ema50'] = cat['Price'].ewm(span=50, min_periods=50).mean()

#Calculate slopes for simple moving averages

cat['sma10_slope'] = np.nan
cat['ema10_slope'] = np.nan
cat['sma25_slope'] = np.nan
cat['ema25_slope'] = np.nan
cat['sma50_slope'] = np.nan
cat['ema50_slope'] = np.nan

x = [x for x in range(10)]
x = sm.add_constant(x)

for i in range(18,len(cat)):
    y = list(cat['sma10'].loc[i-9:i])
    line = sm.OLS(y,x).fit()
    cat['sma10_slope'].loc[i] = line.params[1]
    
    y = list(cat['ema10'].loc[i-9:i])
    line = sm.OLS(y,x).fit()
    cat['ema10_slope'].loc[i] = line.params[1]
    
for i in range(24,len(cat)):
    y = list(cat['sma25'].loc[i-9:i])
    line = sm.OLS(y,x).fit()
    cat['sma25_slope'].loc[i] = line.params[1]
    
    y = list(cat['ema25'].loc[i-9:i])
    line = sm.OLS(y,x).fit()
    cat['ema25_slope'].loc[i] = line.params[1]
    
for i in range(49,len(cat)):
    y = list(cat['sma50'].loc[i-9:i])
    line = sm.OLS(y,x).fit()
    cat['sma50_slope'].loc[i] = line.params[1]
    
    y = list(cat['ema50'].loc[i-9:i])
    line = sm.OLS(y,x).fit()
    cat['ema50_slope'].loc[i] = line.params[1]
        


##------------Plot best intervals--------------
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('whitesmoke')

plt.plot(cat['Price'], color="red", linewidth="0.5")

for x in range(len(buy_idx_list)):
    plt.plot(cat['Price'].loc[buy_idx_list[x]:sell_idx_list[x]], label="Long", color="green")

plt.legend(["Out of the Market", "Long"])
plt.grid()
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks([200,400,600], ["Jan", "Feb", "Mar"])
plt.show()
plt.close(fig)



