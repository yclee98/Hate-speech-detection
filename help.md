# Basic Libraries
```
import numpy as np 
import pandas as pd 
import seaborn as sb 
import matplotlib.pyplot as plt 
sb.set() 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import plot_tree 
from sklearn.metrics import confusion_matrix 
```

# Pandas extraction
``` 
dataset1 = pd.read_csv('train.csv') 
dataset1 = pd.read_html('https://en.wikipedia.org/wiki/2016_Summer_Olympics_medal_table') 
txt_data = pd.read_table('data/somedata.txt', sep = "\s+", header = None) 
xls_data = pd.read_excel('data/somedata.xlsx', sheet_name = 'Sheet1', header = None) 
json_data = pd.read_json('data/somedata.json') 
#extracting only certain columns 
lotarea = pd.DataFrame(dataset1 ['LotArea']) 
```

# Basic exploring  
```
dataset1.head() 
dataset1.shape 
dataset1.dtypes 
dataset1.info() 

# statistics information 
dataset1.describe()  
dataset1['CentralAir'].describe() 

#to count the outlier which are abvove 1 step of Q1 and Q3 
Q1=houseNumData.quantile(0.25) 
Q3=houseNumData.quantile(0.75) 
rule = ((houseNumData < (Q1 - 1.5 * (Q3 - Q1))) | (houseNumData > (Q3 + 1.5 * (Q3 - Q1)))) 
rule.sum() 

#count outlier for many variable  
for var in dataset1: 
    Q1=dataset1[var].quantile(0.25) 
    Q3=dataset1[var].quantile(0.75) 
    rule = ((dataset1[var] < (Q1 - 1.5 * (Q3 - Q1))) | (dataset1[var] > (Q3 + 1.5 * (Q3 - Q1)))) 
    print("Outlier of "+var+" is " +str(rule.sum())) 
```

# Pandas functions  
```
#sample data
df.sample(10)

#to get specified row
dataset1=DataFrame.iloc[10]  

#to get row with label = 0
df.loc[df['label'] == 0].head()

#get all row with dtypes = int64 
dataset1 = dataset1.select_dtypes(include = np.int64) 

#dropping columns  
dataset1 = dataset1.drop(['MSSubClass'], axis = 1) 

#if drop row by index then remove axis 
df.drop([0, 1]) 

#adding columns to the data 
houseCatSale = pd.concat([houseCatData, saleprice], sort = False, axis = 1).reindex(index=houseCatData.index) 

#concate row 
df5=pd.concat([df2, df3, df4], axis=0, ignore_index=True) 

#changing the variable type 
houseCatData['MSSubClass'] = houseCatData['MSSubClass'].astype('category') 

#export to csv 
df5.to_csv(r'movie.csv', index = False) 

#check for duplicate 
df_dup = df5[df5.duplicated("ID", keep = False)] 

#check for unique 
dupids_clean = dupid_data_clean["ID"].unique() 

#sort 
pkmndata_clean[pkmndata_clean["GENERATION"] == generation].sort_values('TOTAL', ascending=False).head(10) 
```

# Basic figure
```
#box plot 
f = plt.figure(figsize=(24, 4)) 
sb.boxplot(data = saleprice, orient = "h") 

#histogram with KDE 
f = plt.figure(figsize=(24, 12)) 
sb.histplot(data = saleprice, kde = True) 

#voilin plot 
f = plt.figure(figsize=(24, 12)) 
sb.violinplot(data = saleprice, orient = "h") 

#all 3 plot together, can increase size (1,3) 
f, axes = plt.subplots(1, 3, figsize=(24, 12)) 
sb.boxplot(data = saleprice, orient = "h", ax = axes[0,0]) 
sb.histplot(data = saleprice, ax = axes[0,1]) 
sb.violinplot(data = saleprice, orient = "h", ax = axes[0,2]) 

#loop to plot multiple variable with the 3 plot 
f, axes = plt.subplots(4, 3, figsize=(18, 20)) 
count = 0 
for var in dataset1: 
    sb.boxplot(data = dataset1[var], orient = "h", ax = axes[count,0]) 
    sb.histplot(data = dataset1[var], ax = axes[count,1], kde = True) 
    sb.violinplot(data = dataset1[var], orient = "h", ax = axes[count,2]) 
    count += 1 
```

# Comparing multi variables figure  
```
#scatter plot of 2 variables 
jointDF = pd.concat([lotarea, saleprice], axis = 1).reindex(lotarea.index) 
sb.jointplot(data = jointDF, x = "LotArea", y = "SalePrice", height = 12) 

#no need to join if the dataframe have both the x and y  
sb.jointplot(data = dataset1, x = “GrLivArea”, y = “SalePrice”, height = 12) 

#another scatterplot format 
f, axes = plt.subplots(1, 1, figsize=(16, 8)) 
plt.scatter(dataset1[‘Height’], dataset1[‘Length’]) 

#pair plot, which does a scatter plot for all the variables 
sb.pairplot(data = houseNumData) 
```

# Correlation heatmap 
```
#work for multiple variable table as well 
dataset1.corr()  #to get the correlation matrix 
sb.heatmap(dataset1.corr(), vmin = -1, vmax = 1, annot = True, fmt=".2f") 

#check correlation coefficient 
dataset1.SalePrice.corr(dataset1.GrLivArea) 
```

# Categorical data exploring 
```
#describe work here as well 
houseCatData.describe() 

#countplot
p = sb.countplot(x='label', data=df)
p.bar_label(p.containers[0])
plt.show()

#catplot which count the number of element in each of the category inside 
#something like histogram for categorical 
sb.catplot(y = 'MSSubClass', data = houseCatData, kind = "count", height = 8) 

#heatmap of 2 variables 
# Distribution of BldgType across MSSubClass 
f = plt.figure(figsize=(20, 8)) 
sb.heatmap(houseCatData.groupby(['BldgType', 'MSSubClass']).size().unstack(),  
           linewidths = 1, annot = True, fmt = 'g', annot_kws = {"size": 18}, cmap = "BuGn") 

#box plot of a categorical variable against a numeric variable  
f = plt.figure(figsize=(16, 8)) 
sb.boxplot(x = 'MSSubClass', y = 'SalePrice', data = houseCatSale) 

#swarm plot 
f = plt.figure(figsize=(16, 8)) 
sb.swarmplot(x = 'SalePrice', y = 'CentralAir', data = dataset1) 

#stripplot, a more condensed version of swarm plot  
f = plt.figure(figsize=(16, 8)) 
sb.stripplot(x = 'SalePrice', y = 'CentralAir', data = dataset1) 

#counting the value of inside the categorical 
countY, countX = houseData.CentralAir.value_counts() 
```

# Linear regression
``` 
linreg = LinearRegression() 
#extract the response and predictor to x and y 
y = pd.DataFrame(dataset1['Length']) 
x= pd.DataFrame(dataset1['Height']) 
#split the data to test and train 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 360) 
# Check the sample sizes 
print("Train Set :", x_train.shape, y_train.shape) 
print("Test Set  :", x_test.shape, y_test.shape) 
#perform linear regression on the train set 
linreg.fit(x_train, y_train) 
#check the a and b of the linear regression 
print('Intercept \t: b = ', linreg.intercept_) 
print('Coefficients \t: a = ', linreg.coef_) 
# Formula for the Regression line 
regline_x = x_train 
regline_y = linreg.intercept_ + linreg.coef_ * x_train 
# Plot the Linear Regression line 
f, axes = plt.subplots(1, 1, figsize=(16, 8)) 
plt.scatter(x_train, y_train) 
plt.plot(regline_x, regline_y, 'r-', linewidth = 3) 
plt.show() 
# Explained Variance in simply the "Score" 
print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train)) 
# Predict the response on the train set 
y_train_pred = linreg.predict(x_train) 
# Compute MSE on the train set 
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred)) 
# Predict SalePrice values corresponding to GrLivArea 
y_test_pred = linreg.predict(x_test) 
# Plot the Predictions on a Scatterplot 
f = plt.figure(figsize=(16, 8)) 
plt.scatter(X_test, y_test, color = "green") #actual 
plt.scatter(X_test, y_test_pred, color = "red") #predicted  
plt.show() 
# Compute MSE on the test set 
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred)) 
```
```
#consolidated 
#fitting the model 
linreg = LinearRegression() 
x = pd.DataFrame(dataset1['Height']) 
y = pd.DataFrame(dataset1['Length']) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1200) 
print("Train Set :", x_train.shape, y_train.shape) 
print("Test Set  :", x_test.shape, y_test.shape) 
linreg.fit(x_train, y_train) 
print('Intercept \t: b = ', linreg.intercept_) 
print('Coefficients \t: a = ', linreg.coef_) 
regline_x = x_train 
regline_y = linreg.intercept_ + linreg.coef_ * x_train 
f, axes = plt.subplots(1, 1, figsize=(24,12)) 
plt.scatter(x_train, y_train) 
plt.plot(regline_x, regline_y, 'r-', linewidth = 3) 
plt.show() 
#predicting using the model 
y_train_pred = linreg.predict(x_train) 
print("Goodness of Fit of Model \tTrain Dataset") 
print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train)) 
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred)) 
print() 
y_test_pred = linreg.predict(x_test) 
print("Goodness of Fit of Model \tTest Dataset") 
print("Explained Variance (R^2) \t:", linreg.score(x_test, y_test)) 
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred)) 
f ,axes = plt.subplots(1,2,figsize=(24,12)) 
axes[0].set_title("Train Dataset") 
axes[0].scatter(x_train, y_train, color = "green") 
axes[0].scatter(x_train,y_train_pred, color = "red") 
axes[0].set_xlabel(x.columns[0]) 
axes[0].set_ylabel(y.columns[0]) 
axes[1].set_title("Test Dataset") 
axes[1].scatter(x_test, y_test, color = "green") 
axes[1].scatter(x_test,y_test_pred, color = "red") 
axes[1].set_xlabel(x.columns[0]) 
axes[1].set_ylabel(y.columns[0]) 
#Multi variable linear regression 
# Extract Response and Predictors 
y = pd.DataFrame(dataset1['SalePrice']) 
X = pd.DataFrame(dataset1[['GrLivArea','LotArea','TotalBsmtSF','GarageArea']]) 
# Split the Dataset into random Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 360) 
# Check the sample sizes 
print("Train Set :", X_train.shape, y_train.shape) 
print("Test Set  :", X_test.shape, y_test.shape) 
# Create a Linear Regression object 
linreg = LinearRegression() 
# Train the Linear Regression model 
linreg.fit(X_train, y_train) 
print('Intercept \t: b = ', linreg.intercept_) 
print('Coefficients \t: a = ', linreg.coef_) 
#cannot plot the regression line since it is not 2d 
# Predict SalePrice values corresponding to Predictors 
y_train_pred = linreg.predict(X_train) 
y_test_pred = linreg.predict(X_test) 
# Plot the Predictions vs the True values 
f, axes = plt.subplots(1, 2, figsize=(24, 12)) 
axes[0].scatter(y_train, y_train_pred, color = "blue") 
axes[0].plot(y_train, y_train, 'w-', linewidth = 1) 
axes[0].set_xlabel("True values of the Response Variable (Train)") 
axes[0].set_ylabel("Predicted values of the Response Variable (Train)") 
axes[1].scatter(y_test, y_test_pred, color = "green") 
axes[1].plot(y_test, y_test, 'w-', linewidth = 1) 
axes[1].set_xlabel("True values of the Response Variable (Test)") 
axes[1].set_ylabel("Predicted values of the Response Variable (Test)") 
plt.show() 
#Goddness of fit 
print("Explained Variance (R^2) on Train Set \t:", linreg.score(X_train, y_train)) 
print("Mean Squared Error (MSE) on Train Set \t:", mean_squared_error(y_train, y_train_pred)) 
print("Mean Squared Error (MSE) on Test Set \t:", mean_squared_error(y_test, y_test_pred)) 
#consolidated 
linreg = LinearRegression() 
y = pd.DataFrame(dataset1['Length']) 
x = pd.DataFrame(dataset1[['Height','Weight','Diameter']]) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1200) 
print("Train Set :", x_train.shape, y_train.shape) 
print("Test Set  :", x_test.shape, y_test.shape) 
linreg.fit(x_train, y_train) 
print('Intercept \t: b = ', linreg.intercept_) 
print('Coefficients \t: a = ', linreg.coef_) 
y_train_pred = linreg.predict(x_train) 
print("Goodness of Fit of Model \tTrain Dataset") 
print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train)) 
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred)) 
print() 
y_test_pred = linreg.predict(x_test) 
print("Goodness of Fit of Model \tTest Dataset") 
print("Explained Variance (R^2) \t:", linreg.score(x_test, y_test)) 
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred)) 
f, axes = plt.subplots(1, 2, figsize=(24,12)) 
axes[0].set_title("Train Dataset") 
axes[0].scatter(y_train, y_train_pred, color = "blue") 
axes[0].plot(y_train, y_train, 'w-', linewidth = 1) 
axes[0].set_xlabel("True values") 
axes[0].set_ylabel("Predicted values") 
axes[1].set_title("Test Dataset") 
axes[1].scatter(y_test, y_test_pred, color = "green") 
axes[1].plot(y_test, y_test, 'w-', linewidth = 1) 
axes[1].set_xlabel("True values") 
axes[1].set_ylabel("Predicted values") 
plt.show() 
```

# Classification tree 
``` 
#do a box plot to visualize the mutual relationship 
#do a swarm plot or stripplot as well 
#max depth tells how dip the tree to go 
dectree = DecisionTreeClassifier(max_depth = 2) 

# Extract Response and Predictors 
y = pd.DataFrame(dataset1['CentralAir']) 
x= pd.DataFrame(dataset1['SalePrice']) 

# Split the Dataset into random Train and Test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 360) 

# Check the sample sizes 
print("Train Set :", x_train.shape, y_train.shape) 
print("Test Set  :", x_test.shape, y_test.shape) 

#fit the decision tree with train data 
dectree.fit(x_train, y_train) 

#visual representation of the tree 
f = plt.figure(figsize=(12,12)) 
out=plot_tree(dectree, filled=True, rounded=True,  
          feature_names=x_train.columns,  
          class_names=["N","Y"]) 
for o in out: 
    arrow = o.arrow_patch 
    if arrow is not None: 
        arrow.set_edgecolor('red') 
        arrow.set_linewidth(3) 

#predicting on the train data 
y_train_pred = dectree.predict(x_train) 
# Plot the two-way Confusion Matrix/the TN/TP chart 
sb.heatmap(confusion_matrix(y_train, y_train_pred),  
           annot = True, fmt=".0f", annot_kws={"size": 18}) 

# Print the Classification Accuracy 
print("Train Data") 
print("Accuracy  :\t", dectree.score(x_train, y_train)) 
print() 

# Print the Accuracy Measures from the Confusion Matrix 
cmTrain = confusion_matrix(y_train, y_train_pred) 
tpTrain = cmTrain[1][1] # True Positives : Y (1) predicted Y (1) 
fpTrain = cmTrain[0][1] # False Positives : N (0) predicted Y (1) 
tnTrain = cmTrain[0][0] # True Negatives : N (0) predicted N (0) 
fnTrain = cmTrain[1][0] # False Negatives : Y (1) predicted N (0) 
print("TPR Train :\t", (tpTrain/(tpTrain + fnTrain))) 
print("TNR Train :\t", (tnTrain/(tnTrain + fpTrain))) 
print() 
print("FPR Train :\t", (fpTrain/(tnTrain + fpTrain))) 
print("FNR Train :\t", (fnTrain/(tpTrain + fnTrain))) 

# Predict the Response corresponding to Predictors 
y_test_pred = dectree.predict(x_test) 

# Plot the two-way Confusion Matrix 
sb.heatmap(confusion_matrix(y_test, y_test_pred),  
           annot = True, fmt=".0f", annot_kws={"size": 18}) 
```

```
#consolidated 
#fiting the tree 
y = pd.DataFrame(dataset1['CentralAir']) 
x= pd.DataFrame(dataset1['SalePrice']) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 360) 
print("Train Set :", x_train.shape, y_train.shape) 
print("Test Set  :", x_test.shape, y_test.shape) 
dectree = DecisionTreeClassifier(max_depth = 2) 
dectree.fit(x_train, y_train) 
f = plt.figure(figsize=(12,12)) 
out=plot_tree(dectree, filled=True, rounded=True,  
          feature_names=x_train.columns,  
          class_names=["N","Y"]) 
for o in out: 
    arrow = o.arrow_patch 
    if arrow is not None: 
        arrow.set_edgecolor('red') 
        arrow.set_linewidth(3) 
#predicting for train 
y_train_pred = dectree.predict(x_train) 
sb.heatmap(confusion_matrix(y_train, y_train_pred),  
           annot = True, fmt=".0f", annot_kws={"size": 18}) 
print("Goodness of Fit of Model \tTrain Dataset") 
print("Classification Accuracy \t:", dectree.score(x_train, y_train)) 
print() 
cmatrix = confusion_matrix(y_train, y_train_pred) 
tp = cmatrix[1][1]  
fp = cmatrix[0][1]  
tn = cmatrix[0][0] 
fn = cmatrix[1][0]  
print("True Positive rate \t: ", tp/(tp+fn)) 
print("True Negative rate \t: ", tn/(tn+fp)) 
print("False Positive rate \t: ", fp/(fp+tn)) 
print("False Negative rate \t: ", fn/(fn+tp)) 
#predicting for test 
y_test_pred = dectree.predict(x_test) 
sb.heatmap(confusion_matrix(y_test, y_test_pred),  
           annot = True, fmt=".0f", annot_kws={"size": 18}) 
print("Goodness of Fit of Model \tTrain Dataset") 
print("Classification Accuracy \t:", dectree.score(x_test, y_test)) 
print() 
cmatrix = confusion_matrix(y_test, y_test_pred) 
tp = cmatrix[1][1]  
fp = cmatrix[0][1]  
tn = cmatrix[0][0] 
fn = cmatrix[1][0]  
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel() 
print("True Positive rate \t: ", tp/(tp+fn)) 
print("True Negative rate \t: ", tn/(tn+fp)) 
print("False Positive rate \t: ", fp/(fp+tn)) 
print("False Negative rate \t: ", fn/(fn+tp)) 
#Multi variable decision tree 
# Extract Response and Predictors 
y = pd.DataFrame(houseData['CentralAir']) 
X = pd.DataFrame(houseData[['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt']]) 
# Split the Dataset into Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 360) 
# Decision Tree using Train Data 
dectree = DecisionTreeClassifier(max_depth = 2)  # create the decision tree object 
dectree.fit(X_train, y_train)                    # train the decision tree model 
# Plot the trained Decision Tree 
f = plt.figure(figsize=(24,24)) 
plot_tree(dectree, filled=True, rounded=True,  
          feature_names=X_train.columns,  
          class_names=["N","Y"]) 
```

```
#consolidateed 
y = pd.DataFrame(dataset1['CentralAir']) 
x= pd.DataFrame(dataset1[['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt']]) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 360) 
print("Train Set :", x_train.shape, y_train.shape) 
print("Test Set  :", x_test.shape, y_test.shape) 
dectree = DecisionTreeClassifier(max_depth = 2) 
dectree.fit(x_train, y_train) 
f = plt.figure(figsize=(12,12)) 
out=plot_tree(dectree, filled=True, rounded=True,  
          feature_names=x_train.columns,  
          class_names=["N","Y"], node_ids=True) 
for o in out: 
    arrow = o.arrow_patch 
    if arrow is not None: 
        arrow.set_edgecolor('red') 
        arrow.set_linewidth(3) 
        
######part 3 
y_train.CentralAir.value_counts() 
node=dectree.apply(x_train) 
x_train['nodes']=node 
x_train.loc[x_train['nodes'] == 5] 
x_train['node']=node 
x_train[‘Predicted’]=y_train_pred 
x_train[‘Actual’]=y_train 
x_train.loc[(x_train['node'] == 5) & (x_train['predict'] == 'Y') & (x_train['real'] == 'N')] 
def linearmodel(x,y,t_size): 
    x = pd.DataFrame(x) 
    y = pd.DataFrame(y) 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = t_size) 
    print("Train Set :", x_train.shape, y_train.shape) 
    print("Test Set  :", x_test.shape, y_test.shape) 
    scaler = StandardScaler() 
    x_train = scaler.fit_transform(x_train) 
    x_test = scaler.transform(x_test) 
    linreg = LinearRegression() 
    linreg.fit(x_train, y_train) 
    print('Intercept \t: b = ', linreg.intercept_) 
    print('Coefficients \t: a = ', linreg.coef_) 
    y_train_pred = linreg.predict(x_train) 
    print("Goodness of Fit of Model \tTrain Dataset") 
    print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train)) 
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred)) 
    print() 
    y_test_pred = linreg.predict(x_test) 
    print("Goodness of Fit of Model \tTest Dataset") 
    print("Explained Variance (R^2) \t:", linreg.score(x_test, y_test)) 
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred)) 
    f, axes = plt.subplots(1, 2, figsize=(24,12)) 
    axes[0].set_title("Train Dataset") 
    axes[0].scatter(y_train, y_train_pred, color = "blue") 
    axes[0].plot(y_train, y_train, 'w-', linewidth = 1) 
    axes[0].set_xlabel("True values") 
    axes[0].set_ylabel("Predicted values") 
    axes[1].set_title("Test Dataset") 
    axes[1].scatter(y_test, y_test_pred, color = "green") 
    axes[1].plot(y_test, y_test, 'w-', linewidth = 1) 
    axes[1].set_xlabel("True values") 
    axes[1].set_ylabel("Predicted values") 
    plt.show() 
```