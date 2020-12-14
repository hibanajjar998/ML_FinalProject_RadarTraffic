

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



###################################################################
# Auxiliary Functions
###################################################################

# RNN for Regression Neural Network:
class RNN(nn.Module):
    def __init__(self, n_l1, n_l2, n_l3, n_feature=67 ):
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

# RNN for Regression Neural Network:
class RNN2(nn.Module):
    def __init__(self, n_l1, n_l2, n_l3, n_l4, n_l5, n_feature=67 ):
        super(RNN2, self).__init__()
        self.l1 = nn.Linear(n_feature, n_l1)
        self.l2 = nn.Linear(n_l1, n_l2)
        self.l3 = nn.Linear(n_l2, n_l3)
        self.l4 = nn.Linear(n_l3, n_l4)
        self.l5 = nn.Linear(n_l4, n_l5)
        self.output = nn.Linear(n_l5, 1)
 
    def forward(self, x):
        y = self.l1(x)
        y = y.relu()
        y = self.l2(y)
        y = y.relu()
        y = self.l3(y)
        y = y.relu()
        y = self.l4(y)
        y = y.relu()
        y = self.l5(y)
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
    train_N, valid_N = (df_train.shape[0], df_valid.shape[0])
    num_categories = df_train.shape[1]-1
    print('num_categories=', num_categories)
    
    for epoch in range(num_epochs):
        
        error_train, error_valid = 0,0
        print("\n\n EPOCH ",epoch+1, "/", num_epochs)
        
        print("    Training ...")
        model.train()
        for batch in train_batches:
            X, Y= batch.drop('Volume', axis=1).to_numpy(), batch.Volume.to_numpy()
            X, Y= torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)
            X, Y= X.reshape(-1, num_categories), Y.reshape(-1)
 
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            # Forward pass 
            outputs = model(X)
            error = loss(outputs, Y)
            error_train += error.item()*batch.shape[0]
            
            #Propagating the error backward
            error.backward()
            
            # Optimizing the parameters
            optimizer.step()
        
        print("    Train loss = ", round(error_train/train_N, 2))
        
        print("    Validation ...")
        model.eval()
        for batch2 in valid_batches:
            X, Y= batch2.drop('Volume', axis=1).to_numpy(), batch2.Volume.to_numpy()
            X, Y= torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)
            X, Y= X.reshape(-1, num_categories), Y.reshape(-1)
            
            outputs = model(X)
            error = loss(outputs, Y)
            error_valid += error.item()*batch2.shape[0]
        
        print("    Validation loss = ", round(error_valid/valid_N, 2))
        
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

data_OH = pd.get_dummies(data)
#train_frac = 0.025  # the equivalent of 100 000 records
#valid_frac = 0.01
#train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation

train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [100000, 200000]) # 60% train, 20% test and validation



# fix hyperparameters:
n_l1, n_l2, n_l3 = 64*2, 32*2, 16*2
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 1

# intialize the model
model_1 = RNN(n_l1, n_l2, n_l3)

# train model and plot graph
train_loss, valid_loss = train_model(model_1, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss, valid_loss, num_epochs, num_model )

## ==> Overfitting problem, we will try a less complex model:

    



###################################################################
# Model 2
###################################################################

data_OH = pd.get_dummies(data)

#train_frac = 0.025  # the equivalent of 100 000 records
#valid_frac = 0.01
#train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation

train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [100000, 200000]) # 60% train, 20% test and validation



# fix hyperparameters:
n_l1, n_l2, n_l3 = 64, 32, 16
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 2

# intialize the model
model_2 = RNN(n_l1, n_l2, n_l3)

# train model and plot graph
train_loss_2, valid_loss_2 = train_model(model_2, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_2, valid_loss_2, num_epochs, num_model )












###################################################################
#  Model 3
###################################################################
# keep location A and month of January
subdata = data[ (data['Month']=='1') & (data['location_name']=='A')]
subdata = subdata.drop(['Month', 'location_name'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3 = 64, 32, 16
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 3

# intialize the model
n_feature=data_OH.shape[1]-1
model_3 = RNN(n_l1, n_l2, n_l3, n_feature= n_feature)

# train model and plot graph
train_loss_3, valid_loss_3 = train_model(model_3, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_3, valid_loss_3, num_epochs, num_model )














###################################################################
#  Model 4
###################################################################
# keep location F only

subdata = data[  (data['location_name']=='F')]
subdata = subdata.drop([ 'location_name'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3 = 64, 32, 16
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 4

# intialize the model
n_feature = data_OH.shape[1]-1
model_4 = RNN(n_l1, n_l2, n_l3, n_feature= n_feature)

# train model and plot graph
train_loss_4, valid_loss_4 = train_model(model_4, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_4, valid_loss_4, num_epochs, num_model )









###################################################################
#  Model 5
###################################################################
# keep only location F with a more complex NN
subdata = data[  (data['location_name']=='F')]
subdata = subdata.drop(['location_name'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3 = 64*4, 32*4, 16
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 5

# intialize the model
n_feature = data_OH.shape[1]-1
model_5 = RNN(n_l1, n_l2, n_l3, n_feature= n_feature)

# train model and plot graph
train_loss_5, valid_loss_5 = train_model(model_5, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_5, valid_loss_5, num_epochs, num_model )










###################################################################
#  Model 6
###################################################################
# keep only location F with a less complex NN
subdata = data[  (data['location_name']=='F')]
subdata = subdata.drop(['location_name'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3 = 16, 8, 8
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 6

# intialize the model
n_feature = data_OH.shape[1]-1
model_6 = RNN(n_l1, n_l2, n_l3, n_feature= n_feature)

# train model and plot graph
train_loss_6, valid_loss_6 = train_model(model_6, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_6, valid_loss_6, num_epochs, num_model )














###################################################################
#  Model 7
###################################################################
# keep only location F and Direction SB with a complex NN and 
subdata = data[  (data['location_name']=='F') & (data['Direction']=='SB')]
subdata = subdata.drop(['location_name', 'Direction'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3 = 64, 64, 16
learning_rate = 0.01
batch_size = 32
num_epochs = 30
num_model = 7

# intialize the model
n_feature = data_OH.shape[1]-1
model_7 = RNN(n_l1, n_l2, n_l3, n_feature= n_feature)

# train model and plot graph
train_loss_7, valid_loss_7 = train_model(model_7, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_7, valid_loss_7, num_epochs, num_model )












###################################################################
#  Model 8
###################################################################
# trying a diferrent architecture, with 5 hidden layers

subdata = data[  (data['location_name']=='F')]
subdata = subdata.drop(['location_name', 'Direction'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3, n_l4, n_l5 = 64, 64, 32, 16, 8
learning_rate = 0.005
batch_size = 32
num_epochs = 30
num_model = 8

# intialize the model
n_feature = data_OH.shape[1]-1
model_8 = RNN2(n_l1, n_l2, n_l3, n_l4, n_l5 , n_feature= n_feature)

# train model and plot graph
train_loss_8, valid_loss_8 = train_model(model_8, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_8, valid_loss_8, num_epochs, num_model )








###################################################################
#  Model 9
###################################################################
# drop direction variable and train with more records an dmore hidden units:

subdata = data.drop([ 'Direction'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.2    # > 868K records
valid_frac = 0.2
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation


# fix hyperparameters:
n_l1, n_l2, n_l3, n_l4, n_l5 = 64*2, 64*2, 32, 16*2, 8
learning_rate = 0.01
batch_size = 32*2
num_epochs = 30
num_model = 9

# intialize the model
n_feature = data_OH.shape[1]-1
model_9 = RNN2(n_l1, n_l2, n_l3, n_l4, n_l5 , n_feature= n_feature)

# train model and plot graph
train_loss_9, valid_loss_9 = train_model(model_9, train, valid, learning_rate, batch_size, num_epochs)


plot_graph(train_loss_9, valid_loss_9, num_epochs, num_model )











###################################################################
#  Model 10
###################################################################
# drop direction variable and train with specific location, month and day of week:
subdata = data[  (data['location_name']=='A') & (data['Month']=='1') ]#& (data['Day of Week']=='1') ]
subdata = subdata.drop(['location_name', 'Direction', 'Month'], axis=1)
data_OH = pd.get_dummies(subdata)
train_frac = 0.8  
valid_frac = 0.1
train, valid, test = np.split(data_OH.sample(frac=1, random_state=777), [int(train_frac*len(data_OH)), int((valid_frac+train_frac)*len(data_OH))]) # 60% train, 20% test and validation

# fix hyperparameters:
n_l1, n_l2, n_l3, n_l4, n_l5 = 64*2, 64*2, 32, 16*2, 8
learning_rate = 0.01
batch_size = 32*2
num_epochs = 60
num_model = 10

# intialize the model
n_feature = data_OH.shape[1]-1
model_10 = RNN2(n_l1, n_l2, n_l3, n_l4, n_l5 , n_feature= n_feature)

# train model and plot graph
train_loss_10, valid_loss_10 = train_model(model_10, train, valid, learning_rate, batch_size, num_epochs)
plot_graph(train_loss_10, valid_loss_10, num_epochs, num_model )








###################################################################
###################################################################
###################################################################
##  LSTM for Time Series
###################################################################


data = pd.read_csv("Radar_Traffic_Counts.csv") 
#data = data.sample(n=10000)
# drop records with missing direction:
data.Direction.replace("None", float('nan'), inplace=True)
data.dropna(inplace=True)

# remove useless leading and trailing whitespaces from the location_name
data.location_name = data.location_name.apply(lambda x: x.strip()) 

# correct the Minute variable using the Time Bin variable, to have all values in {0,15,30,45}
data.Minute = data['Time Bin'].apply(lambda x: int(x[3:]))
data = data.astype(str)
data["date_full"] = data[['Year','Month','Day','Time Bin']].apply(lambda x: '-'.join(x), raw=True, axis=1)

# keep only columns of location name, direction, date and volume 
data = data[['location_name','Direction','date_full','Volume']]
data.Volume = data.Volume.astype(float)

# map location names to letters (for simplification) 
loc_letter, letter_loc = dict(), dict()
letters = 'ABCDEFGHIJKLMNOP'
names = data.location_name.unique()
for i in range(16):
    loc_letter[names[i]] = letters[i]
    letter_loc[letters[i]] = names[i]
data.location_name = data.location_name.apply(lambda x: loc_letter[x])



###################################################################
##  LSTM model: auxiliary functions
###################################################################

class LSTM_TS(nn.Module):
    def __init__(self, input_dim=1, hidden_layer_size=100, n_layers=1, output_fc_size=32, dropout=0.05):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_layer_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_fc_size)
        self.bn = nn.BatchNorm1d(num_features = output_fc_size)        
        self.act = nn.ReLU(inplace=True)
        self.est = nn.Linear(output_fc_size, 1)
        
                
    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size),
                            torch.zeros(1,batch_size,self.hidden_layer_size))
        output , self.hidden_cell = self.lstm( input_seq, hidden_cell)
        output = output.contiguous().view( -1, self.hidden_layer_size)
        output = self.est( self.act( self.bn( self.fc(output))))
        return output



def standarizeVolume(df):
    """this function enables standardization of the volume feature which in principle,
     converts all the input parameters to have a mean of 0 and standard deviation of 1.
     Internally in the different layers of the network, batch normalization is also
     used to avoid covariate shift problems.""" 
    SS = preprocessing.StandardScaler()   
    standarized = SS.fit_transform(np.reshape(df['Volume'].values, (-1,1)))
    standarized = np.reshape( standarized, (-1))
    df['Volume'] = pd.Series( standarized, index=df.index)
    del(standarized, SS)
    return df
    


def split_time_serie(data, location_name, Direction, window_lenght=20):
    """ This function takes as input the dataframe of a specific (location_name, Direction)
        tuple, creates the corresponding time_serie, and aplly a sliding window
        of length "window_lenght" to generate inputs and and their labels for training
        and evaluating our data """
    
    subdata = data[ (data['location_name']==location_name) & (data['Direction']==Direction) ]
    subdata = subdata.sort_values(['date_full']).iloc[:10000]  
    subdata = standarizeVolume(subdata)
    time_serie = list(subdata.Volume)
    X,Y = np.array([]), np.array([])
    for i in range(len(time_serie)-window_lenght):
        end_i = i + window_lenght
        X_i = np.reshape(np.array(time_serie[i: end_i]), (1,-1)) #, dtype=int), (1,1,5))
        Y_i = np.reshape(np.array(time_serie[end_i]), (1)) #, dtype=int), (1,1))
        if i==0:
            X = X_i.copy()
            Y = Y_i.copy()
        else:
            X = np.concatenate([X, X_i], axis=0)
            Y = np.concatenate([Y, Y_i], axis=0)
    return X, Y



def data_split(X, Y, train_frac, valid_frac ):
    train, valid, test = np.array_split(X, [int(train_frac*len(X)), int((train_frac+valid_frac)*len(X))])
    train_y, valid_y, test_y =np.array_split(Y, [int(train_frac*len(X)), int((train_frac+valid_frac)*len(X))])
    return train, valid, test, train_y, valid_y, test_y


# defin a training function, which returns training and validation loss lists
def train_model_LSTM(model, train, valid, train_y, valid_y, learning_rate, batch_size, num_epochs):
    
    # loss function (Mean Squared)  and optimizer (Adam):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_batches = np.array_split(train, 1+ train.shape[0]//batch_size)
    Y_train_batches = np.array_split(train_y, 1+ train_y.shape[0]//batch_size)

    X_valid_batches = np.array_split(valid, 1+ valid.shape[0]//batch_size)
    Y_valid_batches = np.array_split(valid_y, 1+ valid_y.shape[0]//batch_size)
    
    train_loss, valid_loss = [], []
    
    for epoch in range(num_epochs):
        
        error_train, error_valid = 0,0
        print("\n\n EPOCH ",epoch+1, "/", num_epochs)
        
        print("    Training ...")
        model.train()
        for X, Y in zip(X_train_batches, Y_train_batches):
            X, Y= torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
            X, Y= X.reshape(X.shape[0], -1, 1), Y.reshape(-1)
 
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            # Forward pass 
            outputs = model(X)
            error = loss(outputs, Y)
            error_train += error.item()
            
            #Propagating the error backward
            error.backward()
            
            # Optimizing the parameters
            optimizer.step()
        
        print("    Train loss = ", round(error_train, 2))
        
        print("    Validation ...")
        model.eval()
        for X, Y in zip(X_valid_batches, Y_valid_batches):
            X, Y= torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
            X, Y= X.reshape(X.shape[0], -1, 1), Y.reshape(-1)
            
            outputs = model(X)
            error = loss(outputs, Y)
            error_valid += error.item()
        
        print("    Validation loss = ", round(error_valid, 2))
        
        train_loss.append(error_train)
        valid_loss.append(error_valid)
        
    return train_loss, valid_loss


# Define a plot function using the training and validation list, essential to keep an eye on under/overfitting
def plot_graph_LSTM(train_loss, valid_loss, num_epochs, num_model ):
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
    plt.savefig( 'Loss_plots/model_LSTM_'+ str(num_model)+'_loss_plot.png') 
    plt.show()



###################################################################
##  LSTM model: train model 1
###################################################################


# prepare the data
window_lenght = 20
train_frac, valid_frac = 0.8, 0.1
X, Y= split_time_serie(data, 'F', 'SB', window_lenght)
train, valid, test, train_y, valid_y, test_y = data_split(X, Y, train_frac, valid_frac )

# define the model
input_dim = 1
hidden_layer_size = 100
n_layers = 10
output_fc_size = 32
dropout = 0.05
model_LSTM_1 = LSTM_TS(input_dim, hidden_layer_size=100, n_layers=1, output_fc_size=32, dropout=0.05)

# train the model
num_epochs = 30
batch_size = 32
learning_rate = 0.01
train_loss, valid_loss = train_model_LSTM(model_LSTM_1, train, valid, train_y, valid_y, learning_rate, batch_size, num_epochs)
plot_graph_LSTM(train_loss, valid_loss, num_epochs, 1 )

















b = torch.tensor([[0, 1], [2, 3], [6, 7]])
b
torch.reshape(b, (2,-1))
b.view( (2,-1))
