# %%
from glob import glob
import numpy as np
import librosa 
import os
from sklearn.preprocessing import StandardScaler
from pomegranate import *
import itertools
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score
from sklearn import metrics

# %%
import os
from glob import glob

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("/")[2].split(".")[0].split("_") for f in files]
    ids=[]
    speakers=[]
    y=[]
    for f in fnames:
        ids.append(f[2])
        speakers.append(f[1])
        y.append(int(f[0]) )

    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test

def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test


# %% [markdown]
# Βήμα 9

# %%
X_data, X_test, y_data, y_test, spk_train, spk_test = parser("./recordings")

print("Train set consist of %d samples" %len(X_data))    # this data set is used to adjust the weights on the neural network
print("Test set consist of %d samples" %len(X_test)) # this data set is used to minimize overfitting

# %%
X_tr, X_val, y_tr, y_val = train_test_split(X_data, y_data, test_size=0.20)

train_size = len(X_tr)
val_size = len(X_val)
test_size=len(X_test)

print("Train set consist of %d samples" %train_size)    # this data set is used to adjust the weights on the neural network
print("Validation set consist of %d samples" %val_size) # this data set is used to minimize overfitting
print("Test set consist of %d samples" %test_size)      # this data set is used to test the model

# %% [markdown]
# Βήμα 10

# %%
from pomegranate import *

def hmm_model(X, n_states, n_mixtures, gmm=True):

    #X = [] # data from a single digit (can be a numpy array)
    
    #n_states = 2 # the number of HMM states
    #n_mixtures = 2 # the number of Gaussians
    #gmm = True # whether to use GMM or plain Gaussian
    
    X_stacked = np.vstack(X)
    dists = [] # list of probability distributions for the HMM states
    
    for i in range(0,n_states):
        if gmm and n_mixtures>1:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, X_stacked.astype('float64')
) # Initialize General Mixture Models
        else:
            a = MultivariateGaussianDistribution.from_samples(X)
        dists.append(a)
    
    trans_mat = np.zeros((n_states, n_states)) # transition matrix
    for i in range(n_states-1):
        for j in range(n_states):
            if i == j or j == i+1:
                trans_mat[i, j] = 0.5 
    trans_mat[n_states-1, n_states-1] = 1 
    #print(trans_mat)

    starts =  np.zeros(n_states) # your starting probability matrix
    starts[0] = 1
    ends = np.zeros(n_states) # your ending probability matrix
    ends[-1] = 1
    
    #data = [] # your data: must be a Python list that contains: 2D lists with the sequences (so its dimension would be num_sequences x seq_length x feature_dimension)
            # But be careful, it is not a numpy array, it is a Python list (so each sequence can have different length)

    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

    # Fit the model
    # model.fit(data,algorithm='baum-welch')    #Etsi exo X==data poy den exei logikki!!


    return model

# Predict a sequence
#sample = [] # a sample sequence
#logp, _ = model.viterbi(sample) # Run viterbi algorithm and return log-probability

# %%
#necessary for model training/testing cause shapes don't fit otherwise
digits_tr = [[],[],[],[],[],[],[],[],[],[]]               # create a list of 10 positions, one for each digit

for i in range(len(X_tr)):
    digits_tr[y_tr[i]].append(np.asarray(X_tr[i],dtype="float64"))

# dimension would be num_of_digits(10) x num_sequences x seq_length x feature_dimension (6)

print(len(digits_tr))           # 10: one list for each digit
print(len(digits_tr[0]))        # num of samples of a specific digit
print(len(digits_tr[0][0]))     # num of samples that consist the digit
print(len(digits_tr[0][0][0]))  # 6 mfcc for each sample tha consist the digit

# %%
n_states = 3 # the number of HMM states
n_mixtures = 1 # the number of Gaussians

#X=[]

gmm_hmm=[]
for i in range(0,10):
    print("Creating model for digit {0}...".format(i))
    #X.append(np.asarray(digits_tr[i],dtype=object))        # data from a single digit (can be a numpy array)
    gmm_hmm.append(hmm_model(digits_tr[i][0], n_states, n_mixtures))

print("Step 10 Completed! 10 models have been created,  one for each digit")

# %% [markdown]
# Βήμα 11

# %% [markdown]
# Function fitting is the process of training a neural network on a set of inputs in order to produce an associated set of target outputs
# 
# One epoch is when an ENTIRE dataset is passed forward and backward through the neural network only once.

# %%
"""gmm_hmm=[]
for i in range(0,10):
    print("Creating model for digit {0}...".format(i))
    gmm_hmm.append(hmm_model(digits_tr[i][0], n_states, n_mixtures))

print("Step 10 Completed! 10 models have been created,  one for each digit\n")
"""

for i in range(0,10):
    print("Training model for digit {0}...".format(i))
    gmm_hmm[i].fit(digits_tr[i], algorithm='baum-welch')

print("Step 11 Completed! 10 models have been trained,  one for each digit")

# %% [markdown]
# Βήμα 12

# %%
def hmm_log_likelihood(i,model, X, y):
    log_likelihood=[]
    for i in range(len(y)):                            
        probability_percentage = (model.log_probability(X[i]))          # use model.log_probability() to calculate the log-likelihood 
        log_likelihood.append(probability_percentage)                   # prediction is the digit whose HMM maximizes the log-probability
        #print("The likelihood that the sample's sequence is digit {0} at {1}%.".format(i, probability_percentage))

    log_likelihood=np.asarray(log_likelihood)
    return log_likelihood

def hmm_predict(all_models,X,y, disp=False):
    log_likelihood=[]
    for digit in range(0,10):
        prob=[]
        prob=hmm_log_likelihood(digit, all_models[digit], X, y)
        if(disp==True):
            print("\nCalculating likelihood for model {0} ...".format(digit))
            print("maximum log_likelihood propability is:", max(prob))
            print("maximum propability is:                   {0}%".format(100*np.exp(max(prob))))

        log_likelihood.append(prob)
    if(disp==True):
        print("The log likelihood propability for every sample of validation set to be each of the 10 digits have been calculated \n")
    log_likelihood=np.asarray(log_likelihood)   # 10 digits x 600 samples  

    predictions=np.zeros(len(y))
    for i in range(0,len(y)):
        predictions[i]=np.where(log_likelihood[:,i]==max(log_likelihood[:,i]))[0][0]
    return(predictions)

def hmm_score(all_models,X,y,disp=False):
    predictions=hmm_predict(all_models,X,y,disp)
    score = np.sum(y==predictions)/len(y)
    if(disp==True):
        print("GMM_HMM success rate is {0} %".format(100*np.sum(y==predictions)/len(y)))
    return score

# %%
score=hmm_score(gmm_hmm,X_val,y_val,True)

# %% [markdown]
# #finding the best parameters
# #1-4 states and 1-5 mixtures
# best_param=(1,1)
# best_score=0
# all_scores=[]
# 
# for states in range(1,5):       # Number of hmm states
#     for mixtures in range(1,6): # Number of gaussian mixtures
#         print("Number of hmm states: {0}, Number of gaussian mixtures: {1}".format(states,mixtures))
#         models=[]
#         for i in range(0,10):
#             #print(type(digits_tr[0][0]))
#             # print("Creating and Training model for digit {0}...".format(i))
#             models.append(hmm_model(digits_tr[i][0], states, mixtures))
#             models[-1].fit(digits_tr[i])
# 
#         score=hmm_score(models,X_val,y_val)
#         all_scores.append(score)
#         print("GMM_HMM success rate is {0} %\n".format(100*score))
# 
#         if score>best_score:
#             best_score=score
#             best_param=(mixtures,states)
# 
# print("The best parameters based on the model's accurancy are:\n (n_mixtures,n_states)=({0},{1}) ".format(best_param[0],best_param[1]))
# 

# %%
def hmm_defin_train(X, y, digits_tr,n_states, n_mixtures):
    gmm_hmm=[]
    for i in range(0,10):
        #print("Creating model for digit {0}...".format(i))
        gmm_hmm.append(hmm_model(digits_tr[i][0], n_states, n_mixtures))

    for i in range(0,10):
        #print("Training model for digit {0}...".format(i))
        gmm_hmm[i].fit(digits_tr[i], algorithm='baum-welch')

    score=hmm_score(gmm_hmm,X,y)
    print("Number of hmm states: {0}, Number of gaussian mixtures: {1}".format(n_states,n_mixtures))
    print("GMM_HMM success rate is {0} %\n".format(100*score))

    return(gmm_hmm,score)
"""
# %%
best_param=(1,1)
best_score=0

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,1, 1)

if score>best_score:
    best_score=score
    best_param=(1,1)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,1, 2)

if score>best_score:
    best_score=score
    best_param=(1,2)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,1, 3)

if score>best_score:
    best_score=score
    best_param=(1,3)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,1, 4)

if score>best_score:
    best_score=score
    best_param=(1,4)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,1, 5)

if score>best_score:
    best_score=score
    best_param=(1,5)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,2, 1)

if score>best_score:
    best_score=score
    best_param=(2,1)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,2, 2)

if score>best_score:
    best_score=score
    best_param=(2,2)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,2, 3)

if score>best_score:
    best_score=score
    best_param=(2,3)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,2, 4)

if score>best_score:
    best_score=score
    best_param=(2,4)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,2, 5)

if score>best_score:
    best_score=score
    best_param=(2,5)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,3, 1)

if score>best_score:
    best_score=score
    best_param=(3,1)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,3, 2)

if score>best_score:
    best_score=score
    best_param=(3,2)

# %%
_,temp_score=hmm_defin_train(X_val, y_val, digits_tr,3,3)

if score>best_score:
    best_score=score
    best_param=(3,3)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,3,4)

if score>best_score:
    best_score=score
    best_param=(3,4)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,3,5)

if score>best_score:
    best_score=score
    best_param=(3,5)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,4,1)

if score>best_score:
    best_score=score
    best_param=(4,1)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,4, 2)

if score>best_score:
    best_score=score
    best_param=(4,2)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,4, 3)

if score>best_score:
    best_score=score
    best_param=(4,3)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,4, 4)

if score>best_score:
    best_score=score
    best_param=(4,4)

# %%
_,score=hmm_defin_train(X_val, y_val, digits_tr,4, 5)

if score>best_score:
    best_score=score
    best_param=(4,5)

# %%
print("Step 12: The best parameters are {0}, who give accurancy {1}%".format(best_param,100*best_score))
"""
# %% [markdown]
# n_states=best_param[0]
# n_mixtures=best_param[1]
# 
# gmm_hmm=[]
# for i in range(0,10):
#     print("Creating model for digit {0}...".format(i))
#     gmm_hmm.append(hmm_model(digits_tr[i][0], n_states, n_mixtures))
# 
# print("Step 10 Completed! 10 models have been created,  one for each digit\n")
# 
# for i in range(0,10):
#     print("Training model for digit {0}...".format(i))
#     gmm_hmm[i].fit(digits_tr[i], algorithm='baum-welch')
# 
# print("Step 11 Completed! 10 models have been trained,  one for each digit")

# %% [markdown]
# Βήμα 13

# %% [markdown]
# Confusion matrix
# In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm.

# %%
import matplotlib as plt
import matplotlib.pyplot

def hmm_confusion_matrix(actual, predicted, str=""):
    matplotlib.pyplot.figure()
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cm_display.plot()
    plt.pyplot.title(str)

    matplotlib.pyplot.show() 
    return 0

# %%
hmm_confusion_matrix(y_tr,hmm_predict(gmm_hmm,X_tr,y_tr), "Train Set")
score=hmm_score(gmm_hmm,X_tr,y_tr)
print("GMM_HMM success rate is {0} %\n".format(100*score))


# %%
hmm_confusion_matrix(y_val,hmm_predict(gmm_hmm,X_val,y_val), "Validation Set")
score=hmm_score(gmm_hmm,X_val,y_val)
print("GMM_HMM success rate is {0} %\n".format(100*score))


# %%
hmm_confusion_matrix(y_test,hmm_predict(gmm_hmm,X_test,y_test), "Test Set")
score=hmm_score(gmm_hmm,X_test,y_test)
print("GMM_HMM success rate is {0} %\n".format(100*score))

# %% [markdown]
# Βήμα 14

# %% [markdown]
# 
# 1: Create LSTM class based on helper code

# %%
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths = np.array([len(item) for item in feats])  # Find the lengths 

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)): #    lebels : The object that need to be checked as a part of class or not.
                                              #    (list, tuple) : class/type/tuple of class or type, against which object is needed to be checked.
                                              #    Returns : True, if object belongs to the given class/type if single class is passed or any of the class/type if tuple of class/type is passed, else returns False. Raises
            labels = np.array(labels).astype('int64')
        self.labels=labels

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        # --------------- Insert your code here ---------------- #
        
        max_sequence_length=max([len(item) for item in x])
        padded=[np.pad(item, ((0,max_sequence_length-item.shape[0]),(0,0)), mode='constant') for item in x]  # add zeros only
        padded=np.array(padded)

        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        self.num_layers = num_layers
        self.rnn_size=rnn_size
        #self.dropout=dropout
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=rnn_size, num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=0) 
        self.linear = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        
        if not self.bidirectional:
          factor=1
        else:
          factor=2
          
        h_0 = torch.zeros(self.num_layers*factor, x.size(0), self.rnn_size) # hidden state
        c_0 = torch.zeros(self.num_layers*factor, x.size(0), self.rnn_size) # internal state
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # define lstm with input, hidden, and internal state
        last_timestep_output=self.last_timestep(output, lengths, self.bidirectional)
        last_outputs = self.linear(last_timestep_output)
        
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

# %%
train_set = FrameLevelDataset(X_tr, y_tr)     # create trainig dataset
validation_set = FrameLevelDataset(X_val, y_val)    # create validation dataset
test_set = FrameLevelDataset(X_test, y_test)     # create test dataset

batch_size=16
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)            # create dataloader, shuffle data 
validation_loader = torch.utils.data.DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True)  # create dataloader, shuffle data 
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)             # create dataloader, shuffle data 

# %% [markdown]
# 
# 2: Initialize a simple LSTM from the above class
# 

# %%
input_dim     = 6      # number of MFCCs
num_layers    = 2      # number of hidden layers
rnn_dim       = 128    # dimension of hidden layers
output_dim    = 10     # dimension of output 
learning_rate = 0.001  # learning rate of optimizer 
num_epochs    = 40     # number of training epochs

lstm_model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers)       # initialize model
loss_fun = nn.CrossEntropyLoss()                                         # loss function
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)    

# %% [markdown]
# 
# 3: Train the initialized LSTM, and print the training loss for each epoch
# 

# %%
def Val_Loader(model,optimizer,epoch):
# _____evaluate on validation set:_____
    validation_losses=[]
    validation_loss_batch=[]
    predictions=np.array([])                                 # array to hold predicted labels for validation set
    labels=np.array([])

    for i, (X,y,length) in enumerate(validation_loader):
        with torch.no_grad():                                # disable gradient calculation as we will not call torch.backward()
          pred = model(X,length)                             # get predictions
          loss = loss_fun(pred,y)                            # calculate loss 
          validation_loss_batch.append(loss.item())          # add batch loss to array
          predict=torch.argmax(pred,dim=1).numpy()           # get predicted label
          predictions=np.concatenate((predictions,predict))  # add to predictions
          labels=np.concatenate((labels,y))

        mean_validation_loss=np.mean(validation_loss_batch)  # calculate mean loss in epoch
        validation_losses.append(mean_validation_loss)
                     
    print("            Mean validation loss per epoch: {}\n".format( mean_validation_loss))

    accuracy=np.sum(predictions == labels)/len(labels)
    return(accuracy,mean_validation_loss)

def Loader(model,optimizer,num_epochs,validation=False):
  train_losses=[]                                          # array to hold mean train losses per epoch
  validation_losses=[]

  for epoch in range(0,num_epochs):
    train_loss_batch=[]                                    # array to hold train losses per batch
    
    for i, (X_tr,y_tr,length) in enumerate(train_loader):
      optimizer.zero_grad()                            # set gradients to zero
      outputs = model(X_tr,length)                   # give input to model and get output                    
      loss = loss_fun(outputs, y_tr)                      # calculate loss 
      loss.backward()                                  # compute gradient   
      optimizer.step()                                 # update parameters
      train_loss_batch.append(loss.item())             # add batch loss to array 
    
    mean_training_loss=np.mean(train_loss_batch)     # calculate average training loss for epoch 
    train_losses.append(mean_training_loss)
    print("Epoch {}/{}: Mean training loss per epoch: {}".format(epoch,num_epochs, mean_training_loss))

    if validation:
        validation_losses.append(Val_Loader(model,optimizer,epoch))

  if validation: 
    return(train_losses,validation_losses)
  else:
    return train_losses

# %%
lstm_model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers)       # initialize model
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)    

train_losses=Loader(lstm_model,optimizer,num_epochs)

import matplotlib.pyplot as plt
x=np.arange(num_epochs)
plt.plot(x,train_losses)
plt.title('Training loss per epoch',fontsize=15)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# %% [markdown]
# 4: Train the initialized LSTM, and print the training and validation loss for each epoch
# 

# %%
#train_losses=Loader(num_epochs,train_loader,"training")
lstm_model2 = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers)       # initialize model
optimizer2 = torch.optim.Adam(lstm_model2.parameters(), lr=learning_rate)    

train_losses,validation_losses=Loader(lstm_model2,optimizer2,num_epochs, True)


# %%
validation_losses=np.asarray(validation_losses)

import matplotlib.pyplot as plt
x=np.arange(num_epochs)
plt.plot(x,train_losses)
plt.plot(x,validation_losses[:,1])
plt.title('Loss per epoch',fontsize=15)
plt.legend(["Train Set", "Validaion Set"])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# %% [markdown]
# 3: Train the initialized LSTM, and print the training and validation loss for each epoch
# 

# %%
train_losses=[]                                          # array to hold mean train losses per epoch
validation_losses=[]
    
for j in range(0,num_epochs):
  train_loss_batch=[]                                    # array to hold train losses per batch
  
  for i, (X_tr,y_tr,length) in enumerate(train_loader):
    optimizer.zero_grad()                            # set gradients to zero
    outputs = lstm_model(X_tr,length)                   # give input to model and get output                    
    loss = loss_fun(outputs, y_tr)                      # calculate loss 
    loss.backward()                                  # compute gradient   
    optimizer.step()                                 # update parameters
    train_loss_batch.append(loss.item())             # add batch loss to array 
    
    mean_training_loss=np.mean(train_loss_batch)     # calculate average training loss for epoch 
    train_losses.append(mean_training_loss)

# _____evaluate on validation set:_____
    
  validation_loss_batch=[]                              # array to hold validation losses per batch
  predictions=np.array([])                                 # array to hold predicted labels for validation set
  labels=np.array([])                                      # array to hold actual labels for validation set

  for i, (X,y,length) in enumerate(validation_loader):
      with torch.no_grad():                               # disable gradient calculation as we will not call torch.backward()
        pred = lstm_model(X,length)                            # get predictions
        loss = loss_fun(pred,y)                          # calculate loss 
        validation_loss_batch.append(loss.item())         # add batch loss to array
        predict=torch.argmax(pred,dim=1).numpy()          # get predicted label
        predictions=np.concatenate((predictions,predict)) # add to predictions
        labels=np.concatenate((labels,y))
      mean_validation_loss=np.mean(validation_loss_batch)   # calculate mean loss in epoch
      validation_losses.append(mean_validation_loss)
    
  print("Epoch {}: Mean training loss per epoch: {}".format(j,mean_training_loss))
  print("Epoch {}: Mean validation loss per epoch: {}".format(j,mean_validation_loss))
  print('--------------------------------------------------------------------------')
  accuracy=np.sum(predictions == labels)/len(labels)



