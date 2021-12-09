import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Load the Dataset

cust_df = pd.read_csv("data.csv")

# Pre-processing data
df = cust_df.drop('Gender',axis=1)

#Taking the features
X2=df[["Annual Income (k$)","Spending Score (1-100)"]]

#We choose the k 
km = KMeans(n_clusters=5)
y2 = km.fit_predict(X2)

df["label"] = y2
#The data with labels
print(df)

cust1=df[df["label"]==1]
print('Number of customer in 1st group=', len(cust1))
print('They are -', cust1["CustomerID"].values)
print("--------------------------------------------")
cust2=df[df["label"]==2]
print('Number of customer in 2nd group=', len(cust2))
print('They are -', cust2["CustomerID"].values)
print("--------------------------------------------")
cust3=df[df["label"]==0]
print('Number of customer in 3rd group=', len(cust3))
print('They are -', cust3["CustomerID"].values)
print("--------------------------------------------")
cust4=df[df["label"]==3]
print('Number of customer in 4th group=', len(cust4))
print('They are -', cust4["CustomerID"].values)
print("--------------------------------------------")
cust5=df[df["label"]==4]
print('Number of customer in 5th group=', len(cust5))
print('They are -', cust5["CustomerID"].values)
print("--------------------------------------------")

#Predict a value
y3 = km.predict([[15,39]])
print("the class of this cumtomer is : ",y3)

pickle.dump(km,open('modell.pkl','wb'))