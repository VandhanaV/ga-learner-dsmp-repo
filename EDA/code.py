# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here

#Loading the data
data=pd.read_csv(path)

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')

plt.show()


#Subsetting the dataframe based on `Rating` column
data=data[data['Rating']<=5]

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')   

#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
percent_null=total_null/data.isnull().count()
missing_data=pd.concat([total_null, percent_null],keys=['Total','Percent'] , axis=1)
print(missing_data)
data=data.dropna()
total_null_1=data.isnull().sum()
percent_null_1=total_null_1/data.isnull().count()
missing_data_1=pd.concat([total_null_1, percent_null_1],keys=['Total','Percent'] , axis=1)
print(missing_data_1)
#data.columns
# code ends here


# --------------

#Code starts here
import seaborn as sns
ax=sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)
ax.set_xticklabels(rotation=30)
plt.title('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

#Removing `,` from the column
data['Installs']=data['Installs'].str.replace(',','')

#Removing `+` from the column
data['Installs']=data['Installs'].str.replace('+','')

#Converting the column to `int` datatype
data['Installs'] = data['Installs'].astype(int)

#Creating a label encoder object
le=LabelEncoder()

#Label encoding the column to reduce the effect of a large range of values
data['Installs']=le.fit_transform(data['Installs'])

#Setting figure size
plt.figure(figsize = (10,10))

#Plotting Regression plot between Rating and Installs
sns.regplot(x="Installs", y="Rating", color = 'teal',data=data)

#Setting the title of the plot
plt.title('Rating vs Installs[RegPlot]',size = 20)

#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price']=data['Price'].str.lstrip('$').astype('float') 
sns.regplot(x="Price", y="Rating", data=data)
plt.title("Rating vs Price [RegPlot]")
#Code ends here


# --------------
import numpy as np
#print(data['Genres'].value_counts())
data["Genres"]=data["Genres"].str.split(";", expand = True)[0]
gr_mean=data.groupby(['Genres'],as_index=False)[['Genres','Rating']].agg(np.mean)
print(gr_mean.describe())
gr_mean=gr_mean.sort_values('Rating')
print("Lowest rating game",gr_mean[:1])
print("Highest rating game",gr_mean[-1:])


# --------------

#Code starts here
data['Last Updated'].value_counts()
data['Last Updated']=pd.to_datetime(data['Last Updated'])
max_date=max(data['Last Updated'])
Diff_dates= max_date-data['Last Updated']
data['Last Updated Days']=Diff_dates.dt.days
sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title('Rating vs Last Updated [RegPlot]')

#Code ends here


