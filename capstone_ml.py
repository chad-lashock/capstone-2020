from pyomo.environ import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cat = pd.read_csv("./data/CAT.csv")
cat['Price'] = cat['Close']
cat = cat.drop(['Open','High','Low','Close','Adj Close'], axis=1)


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

print("Profit: {}".format(model.profit()))
print([model.buy[i]() for i in range(len(cat))])
print([model.sell[i]() for i in range(len(cat))])

buy_idx_list = [x for x in range(len(cat)) if model.buy[x]() > 0]
sell_idx_list = [x for x in range(len(cat)) if model.sell[x]() > 0]


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



