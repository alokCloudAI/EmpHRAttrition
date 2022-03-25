
import pandas as pd
import numpy as np
import seaborn as sns 

sns.set(rc = {'figure.figsize':(15,8)})

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'

# setting the pandas to view all the available coulmns

pd.set_option('display.max_columns', None)

# Reading the data from CSV files

data = pd.read_csv("/Users/alokkumar/Desktop/HRAttirition/HR-Employee-Attrition.csv", sep=',')
print(data.head(2))
print(data.shape)
print(data.info)

# For data processing, Create a copy dataframe for our processing so that some data we dont lost

df = data.copy()
df.head(2)
print(df.head(2))

# Data Seperation based on types :- Before Separting the Data Column based on its type, Seperating Dependant and Indepandant Feature Seperately
df_target = df[['Attrition']]
df.drop(columns=['Attrition'],axis=1,inplace=True)
df_target.head(2)
print(df_target.head(2))

df_numerical = df.select_dtypes(include=['number'])
df_numerical.head(2)
print(df_numerical.head(2))

df_categorical = df.select_dtypes(include=['object'])
print(df_categorical)

print("Numerical dataframe Cross data type Validation")
for col in df_numerical.columns:
    print(df_numerical[col].value_counts().shape , f"Feature {col}")
    
print("Categorical dataframe Cross data type Validation")
for col in df_categorical.columns:
    print(df_categorical[col].value_counts().shape, f"Feature {col}")
    
#Looks Good for Both Dataframe, Still we can see Education,EmployeeCount,EnvironmentSatisfaction,JobInvolvement,JobLevel,JobSatisfaction,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StandardHours,StockOptionLevel,TrainingTimesLastYear,WorkLifeBalance are acting like indicator variables, So we will treat like this only.
indicator_columns = ['Education','EnvironmentSatisfaction','JobSatisfaction','PerformanceRating','StockOptionLevel','WorkLifeBalance']
df_indicator = df_numerical[indicator_columns]
df_numerical.drop(columns=indicator_columns,axis=1,inplace=True)

# Dropping the Id features
df_numerical.drop(columns=['EmployeeNumber'],axis=1,inplace=True)

# Data Cleaning, outliner detection and Capping for numerical features

df_numerical.describe([0.01,0.1,0.25,0.5,0.75,0.95,0.99])

print(df_numerical.describe([0.01,0.1,0.25,0.5,0.75,0.95,0.99]))

def outlayer_capping(data=None,numerical_columns=None,mode='6sigmaa',multiplier=3,inplace=False):
    if data is None or numerical_columns is None:
        raise NotImplemented("No DataFrame passed or No Numerical Columns provided")
        return
    if inplace:
        df = data
    else:
        df = data.copy()
    for column in numerical_columns:
        stat = df[column].describe()
        if mode != '6sigma':
            df[column]=df[column].clip(lower=df[column].quantile(0.01))
            df[column]=df[column].clip(upper=df[column].quantile(0.99))
        else:
            mask  = (df[column]).le(stat['mean']-multiplier*stat['std'])
            (df[column])[mask] = stat['min']

            mask  = (df[column]).ge(stat['mean']+multiplier*stat['std'])
            (df[column])[mask] = stat['75%']
    if inplace==False:
        return df

outlayer_capping(data=df_numerical,numerical_columns=df_numerical.columns.to_list(),inplace=True)

df_numerical.isna().sum()
df_categorical.isna().sum()
df_indicator.isna().sum()

# Correlation analysis

df_numerical.corr()

def correlation_analysis(dataframe,threshold=(-0.75,0.75),console_output=False,inplace=False):
    """
    Multivariet Analaysis:
    
    This will keep only columns which has low correlation on specified threshold
    Console Output: True=> It print the internal process, False=> It will not print any processing
    threhold: ask the min and max threshold to retain attribure
    -1------------- -0.75---------__attrib retained__-----------0.75----------------1
    """
    if inplace:
        data = dataframe
    else:
        data = dataframe.copy()


    index = 0
    cols = data.columns.to_list()
    while cols[index] is not data[cols[-1]].name:
        new_corr = data.corr()
        mask = new_corr.loc[cols[index]].between(threshold[0],threshold[1],inclusive='both')
        removed_cols = mask[~mask].index.to_list()
        if console_output:
            print("===================================================================================================")
            print("For Feature:  ",cols[index])
            print("```````````````````````````````````````````````````````````````````````````````````````````````````")
            temp = [i for i in removed_cols]
            temp.append(cols[index])
            print(np.round(data[temp].corr(),2).convert_dtypes('str'))
#             print("===================================================================================================")

        data.drop(columns=removed_cols[1:],axis=1,inplace=True)
        cols = data.columns.to_list()
        index = index + 1
    if inplace==False:
        return data
    
[col for col in df_numerical.columns.to_list() if col not in correlation_analysis(dataframe=df_numerical,console_output=False).columns.to_list()]

# Feature Enginering 

df_engineered = pd.DataFrame()
# Numerical Features
df_numerical.head()

# Categorical Feature

for col in df_categorical.columns:
    print(f"Feature======>{col}")
    print(df_categorical[col].value_counts())

df_categorical.drop(columns='Over18',inplace=True,axis=1)

# indicator Features

df_indicator.head(2)

for col in df_indicator.columns:
    print("Feature=====>", col)
    print(df_indicator[col].value_counts())

df_indicator = df_indicator.astype("object",copy=False,errors='raise')
df_indicator.info()

# Zero Variance check for numerical features

print(df_numerical)

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()  #5%-35% 
selector.fit_transform(df_numerical)

selector.get_support(indices=True)

# Univariant analysis

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_target['Attrition'] = le.fit_transform(df_target)
df_target.head()
print(df_target.head())

#  Bi Variate Analysis For numerical categorical features

from sklearn.preprocessing import KBinsDiscretizer

discrete=KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')
num_binned=pd.DataFrame(discrete.fit_transform(df_numerical),index=df_numerical.index, columns=df_numerical.columns).add_suffix('_Rank')
num_binned.head()
print(num_binned.head())

df_numerical_binned = pd.concat([num_binned,df_target],axis=1)

for col in df_numerical_binned.columns:
    plt.figure()
    sns.barplot(x=col,y='Attrition',data=df_numerical_binned,estimator=np.mean)
plt.show()
df_numerical.drop(columns='RelationshipSatisfaction',axis=1,inplace=True)
df_categorical_merged = pd.concat([df_indicator,df_engineered,df_categorical],axis=1)

df_categorical_mergedY = pd.concat([df_categorical_merged,df_target],axis=1)

for col in (df_categorical_mergedY.columns):
    plt.figure()
    sns.barplot(x=col, y="Attrition",data=df_categorical_mergedY, estimator=np.mean )
plt.show()

df_categorical_merged.drop(columns=['Gender','PerformanceRating'],axis=1,inplace=True)

df_numerical.shape

df_categorical_encoded = pd.get_dummies(df_categorical_merged, drop_first = True)
df_categorical_encoded.head()
df_categorical_encoded.shape

df_categorical_encoded_selected = df_categorical_encoded

X = pd.concat([df_numerical,df_categorical_encoded_selected],axis=1)
y = df_target['Attrition']

# For Avoiding all other process
pd.concat([X,y],axis=1).to_csv("processed_raw.csv",index=False)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, shuffle=True, random_state=37)

print(X)

# With Standerdized
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled = train_test_split(X_scaled,y,test_size=0.3,shuffle=True,random_state=29)
CV_results = pd.DataFrame()


# Random Forest Classifier 
# Number of estimator selection OOB

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=17)

rf.set_params(warm_start=True, oob_score=True)

min_estimators = 15
max_estimators = 500

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    rf.set_params(n_estimators=i)
    rf.fit(X_train, y_train)

    oob_error = 1 - rf.oob_score_
    error_rate[i] = oob_error

oob_series = pd.Series(error_rate)
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',color = 'red')
plt.axhline(0.055, color='#875FDB',linestyle='--')
plt.axhline(0.154, color='#875FDB',linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')

rf.set_params(n_estimators=400,oob_score=False)

from sklearn.model_selection import GridSearchCV

param_dist = {'bootstrap': [True,False], 'criterion': ['gini','entropy'],'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 250],'min_samples_leaf': [10,15,30,45,60] }

tree_grid = GridSearchCV(rf, cv = 5, param_grid=param_dist,n_jobs = 3,verbose=4)
tree_grid.fit(X_train,y_train) 

print('Best Parameters using grid search: \n', tree_grid.best_params_)
print("Best Score", tree_grid.best_score_)

param_dist = {'bootstrap': [False], 'criterion': ['entropy'],'max_depth': [7], 'min_samples_split': [40,45,50],'min_samples_leaf': [10] }

tree_grid = GridSearchCV(rf, cv = 3, param_grid=param_dist,n_jobs = 3,verbose=4,return_train_score=True,scoring='accuracy')
tree_grid.fit(X_train,y_train) 

print('Best Parameters using grid search: \n', tree_grid.best_params_)
print("Best Score", tree_grid.best_score_)

rf = RandomForestClassifier(bootstrap=False,criterion='entropy',max_depth=6,min_samples_leaf=10,min_samples_split=50,n_estimators=480)
rf.fit(X_train,y_train)

def feature_importance(fitted_model=None,
                       dataFrame=None,
                       return_till_feature=None,
                       display=True,
                       height=400):
    """
    Purpose
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on information gain for CART model.
    Parameters
    ----------
    * return_top_x(int): Array of Features till top X which are contributing more 
    * dataframe: Dataframe required to calculated columns : This dataframe only used for training
    * fitted_model: Trained Model required

    Returns
    ----------
    Return top X feature if parameter passed
    """
    import plotly.express as px

    if fitted_model is None or dataFrame is None:
        raise EOFError("Failed to Load model or Root DataFrame")
    ranking = pd.DataFrame({
        'features': dataFrame.columns[:-1].to_list(),
        #             'importance': fitted_model.feature_importance
        "importance": fitted_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    ranking['cumsum'] = ranking['importance'].cumsum()
    ranking.index = ranking['features']
    if display:
        fig = px.bar(
            ranking.sort_values(by='importance'),
            y='features',
            x='importance',
            text='importance',
            orientation='h',
            title="Feature Importance Table",
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=10,
                          uniformtext_mode='hide',
                          height=height)
        fig.show()
    if return_till_feature is not None and return_till_feature in ranking.features.to_list(
    ):
        #         return ranking.sort_values(by='importance',ascending=False)['features'].to_list()[:return_top_x]
        return ranking
    importance = feature_importance(fitted_model=rf,dataFrame=pd.concat([X,y],axis=1),height=800,return_till_feature='Age')
    importance.sort_values(by='cumsum',ascending=True)
    important_columns = importance[:'EducationField_Life Sciences'].index.to_list()
    
    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X[important_columns], y, test_size=0.3, shuffle=True, random_state=37)


rf=RandomForestClassifier(random_state=17)
rf.set_params(warm_start=False, 
                  oob_score=True)

min_estimators = 400
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    rf.set_params(n_estimators=i)
    rf.fit(X_train, y_train)

    oob_error = 1 - rf.oob_score_
    error_rate[i] = oob_error

oob_series = pd.Series(error_rate)

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',color = 'red')
plt.axhline(0.055, color='#875FDB',linestyle='--')
plt.axhline(0.154, color='#875FDB',linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')

rf.set_params(n_estimators=550,oob_score=False)


param_dist = {'bootstrap': [True,False], 'criterion': ['gini','entropy'],'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 250],'min_samples_leaf': [10,15,30,45,60] }

tree_grid = GridSearchCV(rf, cv = 5, param_grid=param_dist,n_jobs = 3,verbose=4)
tree_grid.fit(X_train,y_train) 

print('Best Parameters using grid search: \n', tree_grid.best_params_)
print("Best Score", tree_grid.best_score_)

rf = RandomForestClassifier(bootstrap=False,
                            criterion='entropy',
                            max_depth=7,
                            min_samples_leaf=15,
                            min_samples_split=50,
                            n_estimators=550)
rf.fit(X_train,y_train)
cols = [f'split{i}_test_score' for i in range(5)]

pd.DataFrame((pd.DataFrame(tree_grid.cv_results_))[cols].mean()).reset_index()

# Decission tree classifier

from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(random_state=0)

from sklearn.model_selection import GridSearchCV
param_dist = {'criterion': ['gini','entropy'],'max_depth': [6,7,9,10,12], 'min_samples_split': [50, 100, 150],'min_samples_leaf': [10,15,20,30,45] }
tree_grid = GridSearchCV(dtree, cv = 5, param_grid=param_dist,n_jobs = 3,verbose=4,return_train_score=True,scoring='accuracy')
tree_grid.fit(X_train,y_train) 
print('Best Parameters using grid search: \n', tree_grid.best_params_)
print("Best Score", tree_grid.best_score_)
dtree=DecisionTreeClassifier(
    criterion='gini',
    random_state=0,
    max_depth=6,
    min_samples_split=50,
    min_samples_leaf=30)

dtree.fit(X_train,y_train)

from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
plt.figure(figsize=[50,10])
tree.plot_tree(dtree,filled=True,fontsize=15,rounded=True,feature_names=X.columns)
plt.show()

print(plt.show())

# Logistic regression 

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(random_state=0)
logreg.fit(X_train_scaled,y_train_scaled)

coeff_df=pd.DataFrame(X.columns)
coeff_df.columns=['features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
coeff_df
print(coeff_df)

# GradientBoosing Classifier (Bagging)

from sklearn.ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier(random_state=0,n_estimators=850)

from sklearn.model_selection import GridSearchCV

# param_dist = {'learning_rate': [0.001, 0.01, 0.1], 'loss': ['deviance', 'exponential'],'max_depth': [3, 5, 6, 7,10,15], 'min_samples_split': [50, 100, 150, 200, 250, 500],'min_samples_leaf': [10,15,20,30,45,60] }
param_dist = {'learning_rate': [0.1], 'loss': ['deviance'],'max_depth': [6,7,10], 'min_samples_split': [50, 100, 150],'min_samples_leaf': [20,30,45,60] }

tree_grid = GridSearchCV(gbm, cv = 3, param_grid=param_dist,n_jobs = 3,verbose=4,return_train_score=True,scoring='accuracy')
tree_grid.fit(X_train,y_train) 

print('Best Parameters using grid search: \n', tree_grid.best_params_)
print("Best Score", tree_grid.best_score_)

gbm=GradientBoostingClassifier(
    learning_rate=0.1,
    random_state=0,
    loss='deviance',
    max_depth=6,
    min_samples_split=100,
    min_samples_leaf=45)

gbm.fit(X_train,y_train)

# SVC Classifer

from sklearn.svm import SVC
svm_clf = SVC(gamma='auto')

# Stacking Classifier

base_learners = [('rf',
                  RandomForestClassifier(bootstrap=False,
                                         criterion='entropy',
                                         max_depth=7,
                                         min_samples_leaf=10,
                                         min_samples_split=50,
                                         n_estimators=580)),
                 ('gbm',
                  GradientBoostingClassifier(learning_rate=0.1,
                                             random_state=0,
                                             loss='deviance',
                                             max_depth=6,
                                             min_samples_split=100,
                                             min_samples_leaf=45))]

from sklearn.ensemble import StackingClassifier
stack_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

stack_clf.fit(X_train, y_train)

# Model Evaluation Metric Evaluation (Accuracy, Precision, Recall, F1-score, ROC Curve)

from sklearn.metrics import accuracy_score,recall_score,precision_score

models = [dtree,rf,gbm,stack_clf]


from sklearn import metrics
response = []
for model in models:
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1_score = metrics.f1_score(y_test,y_pred)
    
    e = {'Model':str(model),'Y_pred':y_pred, 'accuracy':accuracy, 'precision':precision,'recall':recall,'f1_score':f1_score }
    response.append(e)
    
result = pd.DataFrame(response)
result
print(result)

import plotly.graph_objects as go
model_list = ['DecisionTree','RandomForest','GradientBoosting','StackCLF']

fig = go.Figure(data=[
    go.Bar(name=result[item].name, x=model_list, y=result[item]) for item in result.columns[2:]
])
# Change the bar mode
fig.update_layout(barmode='group',title="Different Model Metric Comparison ",height=700)
fig.show()

# Business interpretation 

data['prediction'] = gbm.predict(X[important_columns])

data['prediction_prob'] = pd.DataFrame(gbm.predict_proba(X[important_columns])).iloc[:,0]
sorted_data = data.sort_values(by='prediction_prob',ascending=False)
sorted_data
print(sorted_data)
