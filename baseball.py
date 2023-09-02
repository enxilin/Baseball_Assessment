import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
 
#clean the data and drop rows with missing values
training_data=pd.read_csv("/Users/enxilin/Desktop/training.csv")
testing_data=pd.read_csv("/Users/enxilin/Desktop/deploy.csv")
missing_data=training_data.isnull().sum()
print(missing_data)
training_data.dropna(inplace=True)
#drop the duplicate value
training_data.drop_duplicates(inplace=True)
#check the outliers, no clear outliers, 
fig,axs=plt.subplots(2,2,figsize=(12,10))
axs[0,0].hist(training_data['Velo'])
axs[0,1].hist(training_data['SpinRate'])
axs[1,0].hist(training_data['HorzBreak'])
axs[1,1].hist(training_data['InducedVertBreak'])# a little left-skewed
plt.show()
#since logistics regression doesn't make any assumptions about the distriburion
#so transforming them to normal distriburion isn't striclly necessary
correlation_matrix=training_data[['InPlay','Velo','SpinRate','HorzBreak','InducedVertBreak']].corr()
print(correlation_matrix)


###Logistics Model
#prepare the features and labels
x=training_data[['Velo','SpinRate','HorzBreak','InducedVertBreak']]
y=training_data['InPlay']
log_model=LogisticRegression()
log_model.fit(x,y)
print(log_model.intercept_)
print(log_model.coef_)
#make predictions on the deploy data
testing_data.dropna(inplace=True)
test_x=testing_data[['Velo','SpinRate','HorzBreak','InducedVertBreak']]
log_predict=log_model.predict_proba(test_x)[:,1] #return the possibilities of the class 1
#if you want the result of 0/1 use log_model.predict(test_x)
print(log_predict)
#add predictions and save as csv
testing_data['PredictionChance']=log_predict
testing_data.to_csv('PredictionsChance_log.csv',index=False)


###Random Forest Model
rf=RandomForestClassifier(random_state=95)
para_rang={'n_estimators':[50,100,200,300,400,500]}
grid_search=GridSearchCV(estimator=rf, param_grid=para_rang,cv=5,n_jobs=-1, verbose=2)
grid_search.fit(x,y)
print(grid_search.best_params_)
random_model=RandomForestClassifier(n_estimators=300,random_state=95)
random_model.fit(x,y)
random_predict=random_model.predict_proba(test_x)[:,1]
print(random_predict)
testing_data['PredictionChance']=random_predict
testing_data.to_csv('PredictionsChance_random.csv',index=False)