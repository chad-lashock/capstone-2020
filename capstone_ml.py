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

x = np.array(cat_svm.drop(['Date','Price','State'],axis=1))
y = np.array(cat_svm['State'])

x[:,0] = pp.scale(x[:,0])
x[:,1] = pp.scale(x[:,1])
x[:,2] = pp.scale(x[:,2])
x[:,3] = pp.scale(x[:,3])
x[:,4] = pp.scale(x[:,4])
x[:,5] = pp.scale(x[:,5])



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)

svm_model = svm.SVC()

# grid_search = GridSearchCV(svm_model, param_grid = {"C": [333,333.5,334,334.5,335,335.5], 
#                            "gamma": [72,72.5,73,73.5,74]},
#                            cv=5,verbose=1,n_jobs=8,return_train_score=True)


grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(520,561,5)], 
                           "gamma": [x/10 for x in range(100,141,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)

grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=54.5,gamma=12)
svm_model.fit(x_train,y_train)

y_pred = pd.DataFrame({'Pred': svm_model.predict(x_test)})

conf_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]),
                 index=['true:0', 'true:1'],
                 columns=['pred:0','pred:1'])


cat_train = cat[:4183]
cat_test = cat[4183:]
cat_test.index = [i for i in range(len(cat_test))]

cat_train = find_best_intervals(cat_train, 64)
cat_test = find_best_intervals(cat_test, 16)

cat_train = prepare_for_ml(cat_train)
cat_test = prepare_for_ml(cat_test)

x_train = np.array(cat_train.drop(['Date','Price','State'],axis=1))
y_train = np.array(cat_train['State'])

x_train[:,0] = pp.scale(x_train[:,0])
x_train[:,1] = pp.scale(x_train[:,1])
x_train[:,2] = pp.scale(x_train[:,2])
x_train[:,3] = pp.scale(x_train[:,3])
x_train[:,4] = pp.scale(x_train[:,4])
x_train[:,5] = pp.scale(x_train[:,5])

x_test = np.array(cat_test.drop(['Date','Price','State'],axis=1))
y_test = np.array(cat_test['State'])

x_test[:,0] = pp.scale(x_test[:,0])
x_test[:,1] = pp.scale(x_test[:,1])
x_test[:,2] = pp.scale(x_test[:,2])
x_test[:,3] = pp.scale(x_test[:,3])
x_test[:,4] = pp.scale(x_test[:,4])
x_test[:,5] = pp.scale(x_test[:,5])

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(550,571,5)], 
                           "gamma": [x for x in range(30,61,2)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)

grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=55.5,gamma=44)
svm_model.fit(x_train,y_train)



cat_test = cat_test.join(y_pred)

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

x_train = x[:3876]
y_train = y[:3876]
x_test = x[3877:]
y_test = y[3877:]

svm_model = svm.SVC(C=24,gamma=3.1)
svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

vz_pred = pd.DataFrame({'Date':pd.to_datetime(vz_svm[3103:]['Date']), 'Predict': y_pred})


##-----------------Random Forest-------------------

cat_rf = prepare_for_ml(cat)

x = x = np.array(cat_svm[['sma10_slope','sma25_slope','sma50_slope','ema10_slope','ema25_slope','ema50_slope']])
y = np.array(cat_rf['State'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)


rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
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







    








