#!/usr/bin/env python
# coding: utf-8

# # NAME : HARISHA B S
# 

# # EMPLOYEE TURNOVER PREDICTION

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[27]:


pip install tabulate


# In[28]:


from tabulate import tabulate


# In[29]:


attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
attrition.columns


# In[30]:


attrition.head()


# In[31]:


attrition.shape


# In[32]:


attrition.info()


# In[33]:


sns.set_style('darkgrid')
sns.countplot(data=attrition,x='Attrition')
plt.show()

size = attrition.groupby("Attrition").size()
percent = size/1470
print(percent)


# In[34]:


attrition.hist(figsize=(25,15))
plt.show()


# In[35]:


print(attrition.StandardHours.value_counts())
print(attrition.EmployeeCount.value_counts())
print(attrition.EmployeeNumber.value_counts())


# In[36]:


attrition = attrition.drop('StandardHours', axis=1)


# In[37]:


attrition = attrition.drop('EmployeeCount', axis=1)


# In[38]:


attrition = attrition.drop('EmployeeNumber', axis=1)


# In[39]:


attrition.shape


# In[40]:


sns.countplot(x = attrition['DailyRate'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()


# In[41]:


sns.countplot(x = attrition['Age'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='Age', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Age': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[42]:


sns.countplot(x = attrition['BusinessTravel'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='BusinessTravel', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Business travel': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[43]:


sns.countplot(x = attrition['Department'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='Department', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Department': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[44]:


sns.countplot(x = attrition['DistanceFromHome'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='DistanceFromHome', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'DistanceFromHome': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[45]:


sns.countplot(x = attrition['Education'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='Education', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Education': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[46]:


sns.countplot(x = attrition['EducationField'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='EducationField', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'EducationField': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[47]:


sns.countplot(x = attrition['EnvironmentSatisfaction'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='EnvironmentSatisfaction', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'EnvironmentSatisfaction': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[48]:


sns.countplot(x = attrition['Gender'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='Gender', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Gender': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[49]:


sns.countplot(x = attrition['HourlyRate'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()


# In[50]:


sns.countplot(x = attrition['JobInvolvement'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='JobInvolvement', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Job Involvement': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[51]:


sns.countplot(x = attrition['JobLevel'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='JobLevel', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'JobLevel': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[52]:


sns.countplot(x = attrition['JobRole'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='JobRole', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Job Role': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[53]:


sns.countplot(x = attrition['JobSatisfaction'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='JobSatisfaction', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Job Satisfaction': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[54]:


sns.countplot(x = attrition['MaritalStatus'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='MaritalStatus', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Marital Status': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[55]:


sns.countplot(x = attrition['MonthlyIncome'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()


# In[56]:


sns.countplot(x = attrition['MonthlyRate'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()


# In[57]:


sns.countplot(x = attrition['NumCompaniesWorked'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='NumCompaniesWorked', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Number of Companies Worked': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[58]:


sns.countplot(x = attrition['Over18'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='Over18', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Over 18': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[59]:


sns.countplot(x = attrition['OverTime'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='OverTime', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Over Time': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[60]:


sns.countplot(x = attrition['PercentSalaryHike'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='PercentSalaryHike', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Percent Salary Hike': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[61]:


sns.countplot(x = attrition['PerformanceRating'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='PerformanceRating', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Performance Rating': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[62]:


sns.countplot(x = attrition['RelationshipSatisfaction'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='RelationshipSatisfaction', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Relationship Satisfaction': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[63]:


sns.countplot(x = attrition['StockOptionLevel'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='StockOptionLevel', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'StockOptionLevel': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[64]:


sns.countplot(x = attrition['TotalWorkingYears'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='TotalWorkingYears', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Total Working Years': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[65]:


sns.countplot(x = attrition['TrainingTimesLastYear'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='TrainingTimesLastYear', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Training Times Last Year': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[66]:


sns.countplot(x = attrition['WorkLifeBalance'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='WorkLifeBalance', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Work life balance': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[67]:


sns.countplot(x = attrition['YearsAtCompany'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='YearsAtCompany', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'Years at Company': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[68]:


sns.countplot(x = attrition['YearsInCurrentRole'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='YearsInCurrentRole', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'YearsInCurrentRole': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[69]:


sns.countplot(x = attrition['YearsWithCurrManager'],hue=attrition['Attrition'])
plt.xticks(rotation=45)
plt.show()

pivot_table = attrition.pivot_table(index='YearsWithCurrManager', columns='Attrition', aggfunc='size', fill_value=0)

total = pivot_table.sum(axis=1)
percentages = (pivot_table.div(total, axis=0) * 100).reset_index()

percentages.rename(columns={'YearsWithCurrManager': '', 'No': 'Percentage of No', 'Yes': 'Percentage of Yes'}, inplace=True)

table = tabulate(percentages, headers='keys', tablefmt='grid', showindex=False)

print(table)


# In[70]:


sns.pairplot(attrition)


# In[71]:


attrition.groupby('Attrition').hist(figsize=(9,9))


# In[72]:


plt.figure(figsize=(20,20))
sns.heatmap(attrition.corr(),annot=True)


# In[73]:


sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.countplot(data=attrition, x='Department', hue='EducationField')

plt.xlabel('Department')
plt.ylabel('Count')
plt.title('Distribution of Education Fields in Each Department')

plt.xticks(rotation=45)
plt.legend(title='Education Field')
plt.tight_layout()
plt.show()


# In[76]:


sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.countplot(data=attrition, x='JobLevel', hue='Education')

plt.xlabel('Job Level')
plt.ylabel('Count')
plt.title('Distribution of Education in Each Job Level')

plt.xticks(rotation=45)
plt.legend(title='Education')
plt.tight_layout()
plt.show()


# In[77]:


sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.countplot(data=attrition, x='JobRole', hue='EducationField')

plt.xlabel('Job Role')
plt.ylabel('Count')
plt.title('Distribution of Education Fields in Each Job Role')

plt.xticks(rotation=45)
plt.legend(title='Education Field')
plt.tight_layout()
plt.show()


# In[78]:


sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.countplot(data=attrition, x='JobRole', hue='Education')

plt.xlabel('Job Level')
plt.ylabel('Count')
plt.title('Distribution of Education in Each Job Role')

plt.xticks(rotation=45)
plt.legend(title='Education')
plt.tight_layout()
plt.show()


# In[79]:


sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.countplot(data=attrition, x='JobRole', hue='JobLevel')

plt.xlabel('JobRole')
plt.ylabel('Count')
plt.title('Distribution of Job Role in Each Job Level')

plt.xticks(rotation=45)
plt.legend(title='Job Level')
plt.tight_layout()
plt.show()


# In[80]:


attrition.columns


# In[81]:


attrition.columns


# In[82]:


columns_to_drop = ['Age', 'DailyRate', 'Department','DistanceFromHome','Education','HourlyRate','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','PercentSalaryHike', 'PerformanceRating','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
attrition_mod = attrition.drop(columns=columns_to_drop)


# In[83]:


attrition_mod.columns


# In[84]:


attrition_mod.info()


# In[85]:


for col in attrition_mod[['BusinessTravel', 'EducationField','Gender',
       'JobRole','MaritalStatus','OverTime']].columns:
  new_df = pd.get_dummies(attrition_mod[col])
  attrition_mod = pd.concat([attrition_mod, new_df], axis=1)
  attrition_mod = attrition_mod.drop([col], axis=1)
  print(attrition_mod.shape)


# In[86]:


attrition_mod.head()


# In[87]:


x = attrition_mod.drop('Attrition', axis=1)
y = attrition_mod[['Attrition']]
print(x.head())
print("\n\n")
print(y.head())
print()
print(x.shape)
print(y.shape)


# In[88]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[89]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier(n_estimators=10)))
models.append(('GB', GradientBoostingClassifier()))


# In[90]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score


# In[91]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify = attrition_mod.Attrition, random_state=123)


# In[92]:


names = []
scores = []
precisions = []
f1 =[]

for name, model in models:
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, pos_label='Yes'))
    f1.append(f1_score(y_test, y_pred, pos_label='Yes'))
    
    names.append(name)


# In[93]:


print(scores)
print(names)
print(precisions)
print(f1)


# In[94]:


models_comparison = pd.DataFrame({'Name': names, 'Score': scores,'Precision': precisions,'F1':f1})
models_comparison.sort_values(by='Score', ascending = False, inplace = True)
print(models_comparison)


# In[95]:


modelChosen = models[-1][1]
modelChosen.feature_importances_


# In[97]:


features = attrition_mod.drop('Attrition', axis=1).columns


# In[98]:


features_weight = list(zip(features, modelChosen.feature_importances_))
features_weight


# In[99]:


selectedFeatures = []

for item in features_weight:
  if item[1] > 0.02:
    selectedFeatures.append(item)
    
selectedFeatures


# In[100]:


from sklearn.model_selection import cross_val_score


# In[101]:


from sklearn.model_selection import KFold 

names = []
scores = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=123, shuffle=True) 
    score = cross_val_score(model, x, y.values.ravel(), cv=kfold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)
    precisions.append(precision_score(y_test,y_pred, pos_label= "Yes"))
    f1.append(f1_score(y_test,y_pred, pos_label= "Yes"))
    
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
kf_cross_val


# In[102]:


kf_cross_val.sort_values(by='Score', ascending=False)

