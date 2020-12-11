

###################################################################
# IMPORT LIBRARIES
###################################################################
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn import preprocessing
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



###################################################################
# Data importation and cleaning
###################################################################

os.chdir('C:\\Users\\hiban\\Desktop\\MN S9\\Machine Learning 3A\\final project\\ML_FinalProject_RadarTraffic')
data = pd.read_csv("Radar_Traffic_Counts.csv") 
data.shape


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
# can PCA help fill missing Direction values ?
###################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# extract a sample (10%) of our data:
data = pd.read_csv("Radar_Traffic_Counts.csv") 
data.Direction.replace("None", float('nan'), inplace=True)
df = data.sample(frac=0.01, random_state=1)

# Separating out the features
features = [ 'location_latitude','location_longitude','Year', 'Month', 'Day','Day of Week', 'Hour', 'Minute', 'Volume']
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['Direction']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# apply PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents , columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
finalDf = pd.concat([principalDf, df[['Direction']]], axis = 1)

# plot a two dimensions projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['NB', 'SB', 'EB', 'WB']
colors = ['r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Direction'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c = color , s = 50)
ax.legend(targets)
ax.grid()

# explained variance
pca.explained_variance_ratio_

# drop records with missing Direction value:
data.dropna(inplace=True)




###################################################################
# Predictors Selection
###################################################################

data = pd.read_csv("Radar_Traffic_Counts.csv") 
data.Direction.replace("None", float('nan'), inplace=True)
data.dropna(inplace=True)

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
min_max_scaler = preprocessing.minmax_scale(feature_range=(-1, 1))

def normalize(df, column):
    x = df[column].values
    x = np.reshape(x, (-1,1))
    x = preprocessing.minmax_scale(x,feature_range=(-1, 1))
    df[column] = pd.Series(np.reshape(x, (-1)))
    return df

data = normalize( data, "location_latitude")
data = normalize( data, "location_longitude")

# drop columns we won't use for prediction: 
data = data.drop(['Year', 'location_name', 'Day'], axis=1)





###################################################################
# Full pre-processing
###################################################################

data = pd.read_csv("Radar_Traffic_Counts.csv") 

# drop records with missing direction:
data.Direction.replace("None", float('nan'), inplace=True)
data.dropna(inplace=True)

# remove useless leading and trailing whitespaces from the location_name
data.location_name = data.location_name.apply(lambda x: x.strip()) 

# correct the Minute variable using the Time Bin variable, to have all values in {0,15,30,45}
data.Minute = data['Time Bin'].apply(lambda x: int(x[3:]))

# drop columns we won't use for prediction: 
data = data.drop(['Year', 'location_latitude', 'location_longitude','Time Bin', 'Day'], axis=1)
data = data.astype(str)
data.Volume = data.Volume.astype(int)

# map location names to letters (for simplification) 
loc_letter, letter_loc = dict(), dict()
letters = 'ABCDEFGHIJKLMNOP'
names = data.location_name.unique()
for i in range(16):
    loc_letter[names[i]] = letters[i]
    letter_loc[letters[i]] = names[i]
data.location_name = data.location_name.apply(lambda x: loc_letter[x])

# the following function encode a dataframe batch into a one-hot representation matrix
def onehot_batch(df):
    one_hot_df = pd.get_dummies(df)
    one_hot_arrays = one_hot_df.to_numpy()
    return one_hot_arrays

data_OH = pd.get_dummies(data)
train_frac = 0.01
valid_frac = 0.01
train, valid, test = np.split(data_OH.sample(frac=1, random_state=7), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation





###################################################################
# Auxiliary Functions
###################################################################

# RNN for Regression Neural Network:
class RNN(nn.Module):
    def __init__(self, n_l1, n_l2, n_l3, n_feature=67, ):
        super(RNN, self).__init__()
        self.l1 = nn.Linear(n_feature, n_l1)
        self.l2 = nn.Linear(n_l1, n_l2)
        self.l3 = nn.Linear(n_l2, n_l3)
        self.output = nn.Linear(n_l3, 1)
 
    def forward(self, x):
        y = self.l1(x)
        y = y.relu()
        y = self.l2(y)
        y = y.relu()
        y = self.l3(y)
        y = y.relu()
        y = self.output(y)
        return y


# defin a training function, which returns training and validation loss lists
def train_model(model, df_train, df_valid, learning_rate, batch_size, num_epochs):
    # loss function (Mean Squared)  and optimizer (Adam):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_batches = np.array_split(df_train, 1+ train.shape[0]//batch_size)
    valid_batches = np.array_split(df_valid, 1+ valid.shape[0]//batch_size)
    
    train_loss, valid_loss = [], []
    train_N, valid_N = (valid.shape[0],train.shape[0])
    
    for epoch in range(num_epochs):
        
        error_train, error_valid = 0,0
        print("\n\n EPOCH ",epoch+1)
        
        print("    Training ...")
        for batch in train_batches:
            X, Y= batch.drop('Volume', axis=1).to_numpy(), batch.Volume.to_numpy()
            X, Y= torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)
            
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            # Forward pass 
            outputs = model(X)
            error = loss(outputs, Y)
            error_train += error.item()*X.shape[0]
            
            #Propagating the error backward
            error.backward()
            
            # Optimizing the parameters
            optimizer.step()
        
        print("    Validation ...")
        for batch2 in valid_batches:
            X, Y= batch2.drop('Volume', axis=1).to_numpy(), batch2.Volume.to_numpy()
            X, Y= torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)
            
            outputs = model(X)
            error = loss(outputs, Y)
            error_valid += error.item()*X.shape[0]
        
        train_loss.append(error_train/train_N)
        valid_loss.append(error_valid/valid_N)
        
    return train_loss, valid_loss
    

# Define a plot function using the training and validation list, essential to keep an eye on under/overfitting
def plot_graph(train_loss, valid_loss, num_epochs, num_model ):
    epochs = list(range(1, num_epochs+1))
    plt.figure(figsize=(10,10))
    font = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 15}
    matplotlib.rc('font', **font)
    plt.plot(epochs, train_loss )
    plt.plot(epochs, valid_loss)
    plt.title('Model '+str(num_model)+' - train and validation loss plots')
    plt.xlabel("Epochs")
    plt.ylabel("MSE loss")
    plt.legend(["Training", 'Validation'])
    plt.savefig( 'Loss_plots/model_'+ str(num_model)+'_loss_plot.png') 
    plt.show()




###################################################################
#  Model 1
###################################################################

# fix hyperparameters:
n_l1, n_l2, n_l3 = 64*2, 32*2, 16*2
learning_rate = 0.01
batch_size = 16
num_epochs = 5
num_model = 1

# intialize the model
model_1 = RNN(n_l1, n_l2, n_l3)

# train model and plot graph
train_loss, valid_loss = train_model(model_1, train, valid, learning_rate, batch_size, num_epochs)
plot_graph(train_loss, valid_loss, num_epochs, num_model )







###################################################################
# 
###################################################################














###################################################################
# 
###################################################################














