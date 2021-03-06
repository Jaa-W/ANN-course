import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

df = pd.read_csv("./Data Files/House_Price.csv", header=0)

# Outlier treatment

uv = np.percentile(df.n_hot_rooms, [99])[0]
df.n_hot_rooms[(df.n_hot_rooms > 3 * uv)] = 3 * uv
#print(data_frame.n_hot_rooms[(data_frame.n_hot_rooms > uv)])

lv = np.percentile(df.rainfall, [1])[0]
df.rainfall[(df.rainfall < 0.3 * lv)] = 0.3 * lv

# Missing value

df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())

# Variable transformation

#sns.jointplot(x = "crime_rate", y = "price", data = df)
#plt.show()

df.crime_rate = np.log(1+df.crime_rate)
#sns.jointplot(x = "crime_rate", y = "price", data = df)
#plt.show() - more linear

df["avg_dist"] = (df.dist1 + df.dist2 + df.dist3 + df.dist4)/4
del df["dist1"]
del df["dist2"]
del df["dist3"]
del df["dist4"]

del df["bus_ter"]

# Dummy variable

df = pd.get_dummies(df)
#print(df.head())
del df["airport_NO"]
del df["waterbody_None"]
#print(df.head())

# Test-train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split