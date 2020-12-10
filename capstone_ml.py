import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from functions import prepare_for_ml
from functions import find_best_intervals
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp


##--------------Load Data--------------------

cat = pd.read_csv("./data/CAT.csv")
cat['Price'] = cat['Close']
cat = cat.drop(['Open','High','Low','Close','Adj Close', 'Volume'], axis=1)

mmm = pd.read_csv("./data/MMM.csv")
mmm['Price'] = mmm['Close']
mmm = mmm.drop(['Open','High','Low','Close','Adj Close', 'Volume'], axis=1)

vz = pd.read_csv("./data/VZ.csv")
vz['Price'] = vz['Close']
vz = vz.drop(['Open','High','Low','Close','Adj Close', 'Volume'], axis=1)

##------Initial exploration and plotting--------------



##-----------Find best intervals-----------

cat = find_best_intervals(cat)
mmm = find_best_intervals(mmm)
vz = find_best_intervals(vz)


##----------CAT SVM------------

cat_svm = prepare_for_ml(cat)

x = np.array(cat_svm.drop(['Date', 'State'], axis=1))
y = np.array(cat_svm['State'])

x = pp.scale(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(370,400,5)], 
                           "gamma": [x/100 for x in range(230,270,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=38,gamma=2.35)
svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

conf_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]),
                 index=['true:Out', 'true:Long'],
                 columns=['pred:Out','pred:Long'])



##-----------------MMM SVM---------------------

mmm_svm = prepare_for_ml(mmm)

x = np.array(mmm_svm.drop(['Date', 'State'], axis=1))
y = np.array(mmm_svm['State'])

x = pp.scale(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(460,510,5)], 
                           "gamma": [x/10 for x in range(28,38,2)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=47.5,gamma=3.2)
svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

conf_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]),
                 index=['true:Out', 'true:Long'],
                 columns=['pred:Out','pred:Long'])


##----------------VZ SVM--------------------

vz_svm = prepare_for_ml(vz)

x = np.array(vz_svm.drop(['Date', 'State'], axis=1))
y = np.array(vz_svm['State'])

x = pp.scale(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [23,23.5,24,24.5], 
                           "gamma": [2.9,2.95,3,3.05,3.1,3.15]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=24,gamma=3.1)
svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

conf_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]),
                 index=['true:Out', 'true:Long'],
                 columns=['pred:Out','pred:Long'])

x_train = x[:4135]
y_train = y[:4135]
x_test = x[4136:]
y_test = y[4136:]

svm_model = svm.SVC(C=24,gamma=3.1)
svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

vz_pred = pd.DataFrame({'Date':pd.to_datetime(vz_svm[3103:]['Date']), 'Predict': y_pred})


##-----------------Random Forest-------------------

cat_rf = prepare_for_ml(cat)

x = np.array(cat_rf.drop(['State', 'Date'], axis=1))
y = np.array(cat_rf['State'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)


rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(x_train, y_train)
##------------Plot best intervals--------------
fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('whitesmoke')

plt.plot(cat['Price'], color="red", linewidth="0.5")

# for x in range(len(buy_idx_list)):
#     plt.plot(cat['Price'].loc[buy_idx_list[x]:sell_idx_list[x]], label="Long", color="green")

plt.legend(["Out of the Market", "Long"])
plt.grid()
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks([200,400,600], ["Jan", "Feb", "Mar"])
plt.show()
plt.close(fig)







    








