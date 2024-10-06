# Class to defone a Model
'''
Class that defines the model architecture
Input: tensor signal
Output: binary signal
'''

# +
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy, KLDivergence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


# -

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

class SeismicModel():
    def __init__(self, name=None):
      assert name is not None, "You must supply a model name"
      self.name = name
      self.rng = np.random.default_rng()
      self.isTrained = False

    def SetTrain(self, x:np.ndarray, y:np.ndarray, batch_size=8):
      '''
      Method to receive the training examples
      x: tensor signal (input)
      y: binary signal (input)
      '''
      self.batch_size = batch_size
      self.x_train = x
      self.y_train = y
      self.dsTrain = tf.data.Dataset.from_tensor_slices((x, y))
      self.dsTrain = self.dsTrain.shuffle(self.dsTrain.cardinality()).batch(batch_size)

    def SetTest(self, x:np.ndarray, y:np.ndarray=None):
      '''
      Method to receive the training examples
      x: tensor signal (input)
      y: binary signal (output)
      '''
      #self.dsTest = tf.data.Dataset.from_tensor_slices((x, y))
      self.x_test = x
      self.y_test = y
      self.y_hat = np.zeros(self.x_test.shape)      # fill with predicted confidence of a seismic event starting at each element

    def Predict(self):
      '''
      Method to generate predictions
      y: binary signal (output)
      '''
      assert isinstance(self.x_test, np.ndarray), "x_test is not a NumPy ndarray"
      assert self.isTrained, "Model must be trained"

      # just guess for now (with no real model)
      if self.model is None:
        N = self.input_shape[0]
        guess = rng.integers(low=0, high=N-1, size=self.y_test.shape[0])        
        self.y_hat[:,guess] = 1.0
      else:
        self.y_hat = self.model.predict(self.x_test)
      return self.y_hat

    def Compile(self, lr=1e-3):
      '''
      Compile the model
      '''

      self.learning_rate = lr
      # select the optimizer
      opt = Adam(learning_rate=self.learning_rate)
      #opt = optimizers.Adam(learning_rate=self.learning_rate)
      #opt = optimizers.Adam(learning_rate=self.learning_rate, clipvalue=0.5)
      #opt = optimizers.RMSprop(learning_rate=0.0001)
      
      # compile the model
      self.model.compile(optimizer='Adam', loss=vae_loss(self.z_mean, self.z_log_var))

      # summarize (print) the model architecture
      self.model.summary()

    def Train(self, epochs=10):
      '''
      Method to train a model
      The trained weights are stored in self.model
      '''

      # assert self.model is a valid keras model
      # Finally, we train the model:
      results = self.model.fit(self.dsTrain, shuffle=True, epochs=epochs)
      self.isTrained = True
      return results

    def BuildModel(self, layer_dim=[64,32,16]):
      '''
      Method to construct the model architecture
      Sets self.model to the constructed model
      '''

      self.input_shape = (self.x_train.shape[1],)
      self.layer_dim = layer_dim
      self.latent_dim = layer_dim[-1]

      # encoder model
      inputs = Input(shape=self.input_shape, name='encoder_input')
      x = Dense(self.layer_dim[0], activation='relu')(inputs)
      x = Dense(self.layer_dim[1], activation='relu')(x)
      self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
      self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
      # use the reparameterization trick and get the output from the sample() function
      z = Lambda(sample, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
      self.encoder = Model(inputs, z, name='encoder')
      self.encoder.summary()

      # decoder model
      latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
      x = Dense(self.layer_dim[1], activation='relu')(latent_inputs)
      x = Dense(self.layer_dim[0], activation='sigmoid')(x)
      outputs = Dense(self.input_shape[0], activation='sigmoid')(x)
      # Instantiate the decoder model:
      self.decoder = Model(latent_inputs, outputs, name='decoder')
      self.decoder.summary()

      # full VAE model
      outputs = self.decoder(self.encoder(inputs))
      self.model = Model(inputs, outputs, name=self.name)


