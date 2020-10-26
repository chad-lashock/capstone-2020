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


