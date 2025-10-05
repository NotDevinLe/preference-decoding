from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.layers import Dense
import numpy as np
import torch
import wandb

rewards = torch.load('data/rewards.pt')

train = rewards[:int(0.8*rewards.shape[0])].numpy()
test = rewards[int(0.8*rewards.shape[0]):].numpy()
x_train = np.reshape(train, (len(train), -1))
x_test = np.reshape(test, (len(test), -1))
x_train = (x_train - x_train.mean(axis=0)) / np.sqrt(x_train.var(axis=0) + 1e-6)
x_test = (x_test - x_test.mean(axis=0)) / np.sqrt(x_test.var(axis=0) + 1e-6)
x_train = x_train.reshape(-1, 256, 256)
x_test = x_test.reshape(-1, 256, 256)

def decoder(x):
    x = Dense(x_train.shape[1])(x)
    return x

selector = ConcreteAutoencoderFeatureSelector(K = 20, output_function = decoder, num_epochs = 800)

selector.fit(x_train, x_train, x_test, x_test)