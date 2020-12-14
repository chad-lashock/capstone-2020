import pandas as pd
import numpy as np
from sklearn import svm
from functions import prepare_for_ml
from functions import find_best_intervals
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from functions import plot_best_intervals
from functions import calculate_interval_profit
from functions import calculate_annualized
from functions import prepare_for_svm_sequentially
from functions import prepare_for_svm_random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

cat = find_best_intervals(cat,80)
mmm = find_best_intervals(mmm,80)
vz = find_best_intervals(vz,80)
 

##----------Plot best intervals-----------

plot_best_intervals(cat)
plot_best_intervals(mmm)
plot_best_intervals(vz)

##----------Calculate best profit-----------

bank = calculate_interval_profit(cat)
calculate_annualized(1000,bank)

bank = calculate_interval_profit(mmm)
calculate_annualized(1000,bank)

bank = calculate_interval_profit(vz)
calculate_annualized(1000,bank)

##----------CAT SVM------------

x_train, x_test, y_train, y_test = prepare_for_svm_random(cat)


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

svm_model = svm.SVC(C=54,gamma=12)
svm_model.fit(x_train,y_train)

svm_model.score(x_test,y_test)

##---------------Analysis with sequentially ordered data-----------------

x_train, x_test, y_train, y_test = prepare_for_svm_sequentially(cat)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(220,261,5)], 
                           "gamma": [x/10 for x in range(510,541,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)

grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=25.5,gamma=53.5)
svm_model.fit(x_train,y_train)

svm_model.score(x_test,y_test)

y_pred = svm_model.predict(x_test)

cat_pred = cat[4194:5227][['Date','Price']]
cat_pred.index = [x for x in range(len(cat_pred))]
cat_pred['State'] = y_pred

##-----------------MMM SVM---------------------

x_train, x_test, y_train, y_test = prepare_for_svm_random(mmm)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(630,701,5)], 
                           "gamma": [x/10 for x in range(160,191,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=49.5,gamma=17)
svm_model.fit(x_train,y_train)

svm_model.score(x_test,y_test)

y_pred = svm_model.predict(x_test)

conf_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]),
                 index=['true:Out', 'true:Long'],
                 columns=['pred:Out','pred:Long'])

##---------------Analysis with sequentially ordered data-----------------

x_train, x_test, y_train, y_test = prepare_for_svm_sequentially(mmm)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(540,571,5)], 
                           "gamma": [x/10 for x in range(370,396,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=45.5,gamma=38.5)
svm_model.fit(x_train,y_train)

svm_model.score(x_test,y_test)

y_pred = svm_model.predict(x_test)

##----------------VZ SVM--------------------

x_train, x_test, y_train, y_test = prepare_for_svm_random(vz)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(630,701,5)], 
                           "gamma": [x/10 for x in range(160,191,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=49.5,gamma=17)
svm_model.fit(x_train,y_train)

svm_model.score(x_test,y_test)

y_pred = svm_model.predict(x_test)

mmm_pred = vz[4194:5227][['Date','Price']]
mmm_pred.index = [x for x in range(len(mmm_pred))]
mmm_pred['Pred'] = y_pred

##---------------Analysis with sequentially ordered data-----------------

x_train, x_test, y_train, y_test = prepare_for_svm_sequentially(vz)

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid = {"C": [x/10 for x in range(690,721,5)], 
                           "gamma": [x/10 for x in range(220,251,5)]},
                           cv=5,verbose=1,n_jobs=8,return_train_score=True)
grid_search.fit(x_train,y_train)

grid_search.best_params_
grid_search.best_score_

svm_model = svm.SVC(C=69.5,gamma=23)
svm_model.fit(x_train,y_train)

svm_model.score(x_test,y_test)

y_pred = svm_model.predict(x_test)

vz_pred = vz[4194:5227][['Date','Price']]
vz_pred.index = [x for x in range(len(vz_pred))]
vz_pred['Pred'] = y_pred



##-------------Calculate trading performance of SVM models-----------

bank = calculate_interval_profit(cat_pred)
(bank/1000)**(365/1498)
bank = calculate_interval_profit(mmm_pred)
(bank/1000)**(365/1498)
bank = calculate_interval_profit(vz_pred)
(bank/1000)**(365/1498)

##-----------------Random Forest-------------------


##----------------CAT RF--------------------------
cat_rf = prepare_for_ml(cat)

x = np.array(cat_rf.drop(['Date','Price','State'],axis=1))
y = np.array(cat_rf['State'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)


rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
rf_model.fit(x_train, y_train)
rf_model.score(x_test,y_test)

##---------------Analysis with sequentially ordered data-----------------

df_train = cat_rf[:4136]
df_test = cat_rf[4136:]
df_test.index = [i for i in range(len(df_test))]

df_train = find_best_intervals(df_train, 64)
df_test = find_best_intervals(df_test, 16)

x_train = np.array(df_train.drop(['Date','Price','State'],axis=1))
y_train = np.array(df_train['State'])
x_test = np.array(df_test.drop(['Date','Price','State'],axis=1))
y_test = np.array(df_test['State'])

rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
rf_model.fit(x_train, y_train)
rf_model.score(x_test,y_test)

y_pred = rf_model.predict(x_test)

cat_rf_pred = cat[4194:5227][['Date','Price']]
cat_rf_pred.index = [x for x in range(len(cat_rf_pred))]
cat_rf_pred['State'] = y_pred

bank = calculate_interval_profit(cat_rf_pred)
(bank/1000)**(365/1498)

##-----------------------MMM RF-----------------------------------
mmm_rf = prepare_for_ml(mmm)

x = np.array(mmm_rf.drop(['Date','Price','State'],axis=1))
y = np.array(mmm_rf['State'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)


rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
rf_model.fit(x_train, y_train)
rf_model.score(x_test,y_test)

##---------------Analysis with sequentially ordered data-----------------

df_train = mmm_rf[:4136]
df_test = mmm_rf[4136:]
df_test.index = [i for i in range(len(df_test))]

df_train = find_best_intervals(df_train, 64)
df_test = find_best_intervals(df_test, 16)

x_train = np.array(df_train.drop(['Date','Price','State'],axis=1))
y_train = np.array(df_train['State'])
x_test = np.array(df_test.drop(['Date','Price','State'],axis=1))
y_test = np.array(df_test['State'])

rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
rf_model.fit(x_train, y_train)
rf_model.score(x_test,y_test)

y_pred = rf_model.predict(x_test)

mmm_rf_pred = mmm[4194:5227][['Date','Price']]
mmm_rf_pred.index = [x for x in range(len(mmm_rf_pred))]
mmm_rf_pred['State'] = y_pred

bank = calculate_interval_profit(mmm_rf_pred)
(bank/1000)**(365/1498)

##-----------------------VZ RF-----------------------------------
vz_rf = prepare_for_ml(vz)

x = np.array(vz_rf.drop(['Date','Price','State'],axis=1))
y = np.array(vz_rf['State'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)


rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
rf_model.fit(x_train, y_train)
rf_model.score(x_test,y_test)

##---------------Analysis with sequentially ordered data-----------------

df_train = vz_rf[:4136]
df_test = vz_rf[4136:]
df_test.index = [i for i in range(len(df_test))]

df_train = find_best_intervals(df_train, 64)
df_test = find_best_intervals(df_test, 16)

x_train = np.array(df_train.drop(['Date','Price','State'],axis=1))
y_train = np.array(df_train['State'])
x_test = np.array(df_test.drop(['Date','Price','State'],axis=1))
y_test = np.array(df_test['State'])

rf_model = RandomForestClassifier(random_state=0,n_estimators=500)
rf_model.fit(x_train, y_train)
rf_model.score(x_test,y_test)

y_pred = rf_model.predict(x_test)

vz_rf_pred = vz[4194:5227][['Date','Price']]
vz_rf_pred.index = [x for x in range(len(vz_rf_pred))]
vz_rf_pred['State'] = y_pred

bank = calculate_interval_profit(vz_rf_pred)
(bank/1000)**(365/1498)
    








