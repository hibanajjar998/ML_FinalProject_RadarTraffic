

###################################################################
# IMPORT LIBRARIES
###################################################################
import pandas as pd
import os







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














###################################################################
# 
###################################################################














