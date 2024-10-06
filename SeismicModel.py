# Class to defone a Model
'''
Class that defines the model architecture
Input: tensor signal
Output: binary signal
'''

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from torchinfo import summary

def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1)

# Gaussian loss function:
def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# KL Divergence loss function:
def vae_loss(z_mean, z_log_var):
    def kl_divergence(y_true, y_pred):
      #z_mean = model.get_layer(name='z_mean')
      #z_log_var = model.get_layer(name='z_log_var')
      # compute the average MSE error, then scale it up, ie. simply sum on all axes
      reconstruction_loss = K.sum(K.square(y_true - y_pred))
      # compute the KL loss
      kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
      # return the average loss over all
      total_loss = K.mean(reconstruction_loss + kl_loss)
      #total_loss = reconstruction_loss + kl_loss
      return total_loss
    return kl_divergence

from torch.utils.data import Dataset, DataLoader



class SeismicModel():
    class Wide(nn.Module):
      def __init__(self, input_dim, layer_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, layer_dim[0])
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(layer_dim[0], layer_dim[1])
        self.act2 = nn.ReLU()
        self.dense3 = nn.Linear(layer_dim[1], layer_dim[2])
        self.act3 = nn.ReLU()
        self.output = nn.Linear(layer_dim[2], input_dim)
        self.actFinal = nn.Sigmoid()

      def forward(self, x):
        x = self.act1(self.dense1(x))
        x = self.act2(self.dense2(x))
        x = self.act3(self.dense3(x))
        x = self.actFinal(self.output(x))
        return x

    class TorchDataset(Dataset):
      def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

      def __getitem__(self, index):
        return self.data[index], self.labels[index]

      def __len__(self):
        return len(self.data)
    
    def __init__(self, name=None):
      assert name is not None, "You must supply a model name"
      self.name = name
      self.rng = np.random.default_rng()
      self.isTrained = False

    def SetTrain(self, x:np.ndarray, y:np.ndarray):
      '''
      Method to receive the training examples
      x: tensor signal (input)
      y: binary signal (input)
      '''
      self.trainset = self.TorchDataset(x.astype(np.float32), y.astype(np.float32))
      self.xdim = x.shape
      self.ydim = y.shape

    def SetDataLoader(self, dataloader, xshape, yshape):
      '''
      Method to store a Pytorch DataLoader
      dataloader: Pytorch DataLoader (input)
      '''
      self.dataloader = dataloader
      self.xdim = xshape
      self.ydim = yshape

    def SetCollate(self, collate):
      '''
      Method to store a custom collating function
      dataloader: collate (input)
      '''
      self.collate = collate
        
    def SetTest(self, x:np.ndarray, y:np.ndarray=None):
      '''
      Method to receive the training examples
      x: tensor signal (input)
      y: binary signal (output)
      '''
      self.xdim = x.shape
      if y is None:
        self.testset = self.TorchDataset(x.astype(np.float32), np.zeros(x.shape, dtype=np.float32))
        self.ydim = x.shape
      else:
        self.testset = self.TorchDataset(x.astype(np.float32), y.astype(np.float32))
        self.ydim = y.shape

    def Predict(self):
      '''
      Method to generate predictions
      y: binary signal (output)
      '''
      #assert isinstance(self.x_test, np.ndarray), "x_test is not a NumPy ndarray"
      assert self.isTrained, "Model must be trained"

      self.y_hat = []
      test_dataloader = DataLoader(self.dataloader, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate)
      # fill yhat with predicted confidence of a seismic event starting at each element
      # just guess for now (with no real model)
      if self.model is None:
        N = self.ydim[1]
        guess = rng.integers(low=0, high=N-1, size=self.ydim[0])
        self.y_hat = np.zeros(self.ydim[0])
        self.y_hat[:,guess] = 1.0
      else:
        self.model.eval()
        with torch.no_grad():
          for i, data in enumerate(test_dataloader):
            inputs, labels_ohe, labels_bin = data
            self.y_hat.append(self.model(inputs))
      return np.vstack(self.y_hat)

    def PredictOne(self, inputs):
      '''
      Method to generate predictions
      y: binary signal (output)
      '''
      #assert isinstance(self.x_test, np.ndarray), "x_test is not a NumPy ndarray"
      assert self.isTrained, "Model must be trained"

      self.model.eval()
      with torch.no_grad():
        y_hat = self.model(inputs)
      return y_hat

    def Compile(self, lr=1e-3, batch_size=8):
      '''
      Compile the model
      '''

      self.learning_rate = lr     
      self.batch_size = batch_size
      # compile the model
      # loss function and optimizer
      self.loss_function = nn.BCELoss()  # binary cross entropy
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
 

      # summarize (print) the model architecture    
      #summary(self.model, input_size=(self.batch_size, self.xdim[1]))

    def Train(self, epochs=10):
      '''
      Method to train a model
      The trained weights are stored in self.model
      '''

      # assert self.model is a valid keras model
      # Finally, we train the model:
 
      # Hold the best model
      best_acc = - np.inf   # init to negative infinity
      best_weights = None
      self.history = {'loss': [], 'accuracy': [] }

      train_dataloader = DataLoader(self.dataloader, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate)

      # Training loop
      num_epochs = epochs
      for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
          inputs, labels_ohe, labels_bin = data

          # Zero the gradients
          self.optimizer.zero_grad()

          # Forward pass
          outputs = self.model(inputs)

          # Transform labels_ohe into tensor 
          y_true = torch.tensor(np.vstack(labels_ohe))
          #print('y_true: ', y_true.shape, y_true.dtype)
          #print('outputs: ', outputs.shape, outputs.dtype)
          # Calculate loss
          loss = self.loss_function(outputs, y_true)
          acc = (outputs == y_true).float().mean()

          # Backward pass and optimization
          loss.backward()
          self.optimizer.step()

          # Print progress (optional)
          if i % 10 == 0:
            print(f'Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item()}')    
          
          self.history['loss'].append(loss.detach().numpy())
          self.history['accuracy'].append(acc.detach().numpy())
          last_loss = loss
          last_acc = acc
            
      self.isTrained = True
      return last_loss, last_acc

    def BuildModel(self, layer_dim=[64,32,16]):
      '''
      Method to construct the model architecture
      Sets self.model to the constructed model
      '''
      self.model = self.Wide(self.xdim[1], layer_dim)


