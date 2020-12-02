

###################################################################
# IMPORT LIBRARIES
###################################################################
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn import preprocessing






###################################################################
# Data importation and cleaning
###################################################################

os.chdir('C:\\Users\\hiban\\Desktop\\MN S9\\Machine Learning 3A\\final project\\ML_FinalProject_RadarTraffic')
data = pd.read_csv("Radar_Traffic_Counts.csv") 
data.shape
sample = data.sample(frac=0.01, random_state=1)


# Do we have nan values?
data.isna().sum() # no nan value
data.Direction.value_counts()  # we have 'None' values ~5.6%
data.Direction.replace("None", float('nan'), inplace=True)


# explore locations:
data.location_name.value_counts() # 23 distancts locations
data.location_name = data.location_name.apply(lambda x: x.strip()) # to remove useless leading and trailing whitespaces 
data.location_name.value_counts() # 18 distancts locations
data.location_longitude.nunique() # 18 distancts locations
data.location_latitude.nunique() # 18 distancts locations
data.Direction.value_counts(dropna=False).to_frame().transpose()
    # --> 18 distanct locations but records aren't uniformly distibuted


# explore dates:
data.Year.value_counts()
data.Year.plot.hist(bins=3)
data.Month.value_counts()
data.Month.plot.hist(bins=12)
data['Day of Week'].value_counts()
data['Day of Week'].plot.hist(bins=7)
data.Day.plot.hist()
data.Hour.value_counts()
data.Hour.plot.hist(bins=24)
    # dates are uniformly distributed between records


# Volume:
data.Volume.describe().to_frame().transpose()


###################################################################
# Predictors Selection
###################################################################

# Heat Correlation matrix
ax = sns.heatmap(data.corr(), linewidth=0.5)

# Boxplots: Volume-Date
ax = sns.boxplot(x="Year", y="Volume", data=data)
ax = sns.boxplot(x="Hour", y="Volume", data=data)
ax = sns.boxplot(x="Day", y="Volume", data=data)
ax = sns.boxplot(x="Day of Week", y="Volume", data=data)

# Boxplots: Volume-Location
data.location_latitude = data.location_latitude.apply(lambda x: round(x,3))
data.location_longitude = data.location_longitude.apply(lambda x: round(x,3))
ax = sns.boxplot(x="Direction", y="Volume", data=data)
ax = sns.boxplot(y="location_name", x="Volume", data=data, orient="h")
ax = sns.boxplot(y="location_longitude", x="Volume", data=data, orient="h")
ax = sns.boxplot(y="location_latitude", x="Volume", data=data, orient="h")


# Normalize longitude and latitude variables:
min_max_scaler = preprocessing.MinMaxScaler()

def normalize(df, column):
    x = df[column].values
    x = np.reshape(x, (-1,1))
    x = min_max_scaler.fit_transform(x)
    df[column] = pd.Series(np.reshape(x, (-1)))
    return df

data = normalize( data, "location_latitude")
data = normalize( data, "location_latitude")

# drop columns we won't use for prediction: 
data = data.drop(['Year', 'location_name', 'Day'], axis=1)

###################################################################
# 
###################################################################














###################################################################
# 
###################################################################














###################################################################
# 
###################################################################














###################################################################
# 
###################################################################














###################################################################
# 
###################################################################














###################################################################
# 
###################################################################














