# --------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# code starts here
df=pd.read_csv(path)
print(df.head())
X=df.drop(columns={'list_price'})
y=df['list_price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3, random_state = 6)
# code ends here



# --------------
import matplotlib.pyplot as plt
import seaborn as sns
# code starts here        
fig ,axes=plt.subplots(nrows = 3 , ncols = 3,figsize=(15,20))
cols=X_train.columns
for i in range(0,3):
    for j in range(0,3):
        col=cols[ i * 3 + j]
        axes[i,j].scatter(X_train[col],y_train)


# --------------
# Code starts here
corr=abs(X_train.corr())
print(corr)

# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
X_train=X_train.drop(columns=to_drop)
X_test=X_test.drop(columns=to_drop)


# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here

#Instantiate linear regression model
regressor=LinearRegression()

# fit the model
regressor.fit(X_train,y_train)

# predict the result
y_pred =regressor.predict(X_test)

# Calculate mse
mse = mean_squared_error(y_test, y_pred)

# print mse
print(mse)

# Calculate r2_score
r2 = r2_score(y_test, y_pred)

#print r2
print(r2)

# Code ends here


# --------------
# Code starts here


# calculate the residual
residual = (y_test - y_pred)

# plot the figure for residual
plt.figure(figsize=(15,8))
plt.hist(residual, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frequency")   
plt.title("Error Residual plot")
plt.show()

# Code ends here


