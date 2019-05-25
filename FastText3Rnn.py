#!/usr/bin/env python
# coding: utf-8

# 
# ## echo TrumpVeryDown.csv| tr -d , |sed -e 's/ /,/g' >> thingPlace
# 

# In[8]:


import pickle
from gensim.models.keyedvectors import KeyedVectors


# In[1]:


from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, UpSampling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np


# In[ ]:


from keras.datasets import mnist


# In[13]:


from gensim.models import Word2Vec


# In[4]:


# import fastText
import numpy as np
from numpy import array
import pandas as pd 


# In[41]:


#secondCsv


#  For dynamic convert to:

# In[5]:



from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('cc.en.300.bin')


# In[ ]:





# For basic convert back:

# In[15]:


modelnow = KeyedVectors.load_word2vec_format('~/MachineLearningBase/cc.en.300.vec')


# In[19]:


testtestArray[1][1]


# In[ ]:


#fullArray = np.zeros((8850,55,300))


# In[13]:


#fullArray = pickle.load( open( "TrumpPostTextSave.p", "rb" ) )


# In[14]:


#secondCsv['1',1]


# In[40]:


#fullArray2 = fullArray[:][0:51]


# In[15]:


fullArray4 = np.zeros((8431,52,300))


# In[16]:


# for i in range(0,8431):
#     for j in range(0,52):
#         fullArray4[i][j] = fullMainArray[i][j]


# In[17]:


# pickle.dump( fullArray4, open( "TrumpPostFULLFinal.p", "wb" ) )


# In[60]:


#len(fullArray3[1][1])


# In[12]:


fullMainArray = pickle.load( open( "TrumpPostFULLFinal.p", "rb" ) )


# In[13]:


fullMainArray.shape


# In[2]:


#fullMainArray = pickle.load( open( "TrumpPostTextFinalSave.p", "rb" ) )


# In[32]:


testArray3 = np.array(["as","they","rounded","a","bend","in","the","path","that","ran","beside","the","river","they","recognized","the","silhouette","of","a","fig","tree","atop","a","nearby","hill","the","weather","was","hot","and","the","days","were","long","the","fig","tree","was","in","full","leaf","but","not","yet","bearing","fruit","soon","they","spotted","other","landmarks","too"])
sampleSentence = np.zeros((1,52,300))
for i in range(0,52):
    sampleSentence[0][i] = modelnow[testArray3[i]]


# In[20]:


for i in range(0,52):
    print(fullMainArray[5][i])
    print(modelnow.similar_by_vector(fullMainArray[5][i], topn=1))


# In[15]:


# Global Constants
#images_dir = "asdf"
#img_rows = 55 
#img_cols = 300
#noise_len = 100

#images_dir = "asdf"
#img_rows = 55 
#img_cols = 300
#noise_len = 100


# In[ ]:





# In[ ]:





# In[19]:


# Other Constants
images_dir = "out_place"
img_cols = 52
channels = 300
noise_len = 100


# [Link to example](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

# In[ ]:



def build_discriminator():
    '''
    Put together a CNN that will return a single confidence output.
    
    returns: the model object
    '''
    img_rows = 300
    img_cols = 52
    img_shape = (img_cols, 300)

    
    model = Sequential()

    model.add(LSTM(64, input_shape=img_shape))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def build_generator():
    '''
    Put together a model that takes in one-dimensional noise and outputs two-dimensional
    data representing a black and white image, with -1 for black and 1 for white.
    
    returns: the model object
    '''

    noise_shape = (noise_len,)
    
    model = Sequential()

    model.add(Dense(img_cols*16*2, activation="relu", input_shape=noise_shape))
    model.add(Reshape((13, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8)) 
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv1D(300, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model

def build_combined():
    '''
    Puts together a model that combines the discriminator and generator models.
    
    returns: the generator, discriminator, and combined model objects
    '''
    
    optimizer = Adam(0.0002, 0.5)

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', 
                          optimizer=optimizer,
                          metrics=['accuracy'])


    # Build and compile the generator
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # The generator takes noise as input and generates images
    noise = Input(shape=(noise_len,))
    img = generator(noise)
  #  print(len(img))
  #  print(len(img[0]))
  #  print(len(img[0][0]))
    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity 
    combined = Model(inputs=noise, outputs=valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator, discriminator, combined

def save_imgs(generator, epoch):
    '''
    Has the generator create images and saves the images in a single file that includes
    the epoch in the filename.
    
    inputs:
        generator: the generator model object returned by build_combined
        epoch: the epoch number (but can be anything that can be represented as a string)
    
    returns: None
    '''
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_len))
   # noise = np.random.normal(0, 1, noise_len).reshape(100,)
    
    gen_img = generator.predict(noise)
    #np.savetxt((str(epoch) + ".csv"), gen_img, delimiter=",")
    pickle.dump( gen_img, open("TTestNowFixedChecking/"+(str(epoch) + ".p"), "wb" ) )
    
#  # Rescale images 0 - 1
#  gen_imgs = 0.5 * gen_imgs + 0.5

#  fig, axs = plt.subplots(r, c)
#  #fig.suptitle("DCGAN: Generated digits", fontsize=12)
#  cnt = 0
#  for i in range(r):
#      for j in range(c):
#          axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
#          axs[i,j].axis('off')
#          cnt += 1
#  fig.savefig(os.path.join(images_dir, 'mnist_{}.png'.format(epoch)))
#  plt.close()

def train(generator, discriminator, combined, epochs, batch_size=128, save_interval=50):
    '''
    Trains all model objects
    
    generator: the generator model object returned by build_combined
    discriminator: the discriminator model object returned by build_combined
    combined: the combined model object returned by build_combined
    epochs: integer, the number of epochs to train for
    batch_size: integer, the number of training samples to use at a time
    save_interval: integer, will generate and save images when the current epoch % save_interval is 0
    
    returns: None
    '''

    # Load the dataset
#   (X_train, _), (_, _) = mnist.load_data()

#   # Rescale -1 to 1
#   X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#   X_train = np.expand_dims(X_train, axis=3)
#   
    X_train = fullMainArray
#  ################################################################################   X_train = sampleSentence
    half_batch = int(batch_size / 2)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

        # Sample noise and generate a half batch of new images
        noise = np.random.normal(0, 1, (half_batch, noise_len))
        gen_imgs = generator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
      #  print(imgs) 
      #  print(np.ones((half_batch, 1)))
      #  print(gen_imgs)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, noise_len))

        # Train the generator (wants discriminator to mistake images as real)
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
 
        # If at save interval => save generated image samples and plot progress
        if epoch % save_interval == 0:
            # Plot the progress
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print ("{} [D loss: {}, acc.: {:.2%}] [G loss: {}]".format(epoch, d_loss[0], d_loss[1], g_loss))
            save_imgs(generator, epoch)
            
def show_new_image(generator):
    '''
    Generates and displays a new image
    
    inputs: generator object model returned from build_combined
    
    returns: generated image
    '''
    
    noise = np.random.normal(0, 1, (1, noise_len))
    gen_img = generator.predict(noise)[0][:,:,0]
    
    return plt.imshow(gen_img, cmap='gray', vmin=-1, vmax=1)


# In[20]:


# #USE Conv1D

# def build_discriminator():
#     '''
#     Put together a CNN that will return a single confidence output.
    
#     returns: the model object
#     '''
#     img_rows = 300
#     img_cols = 52
#     img_shape = (img_cols, 300)

    
#     model = Sequential()

#     model.add(Conv1D(32, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.25))
#     model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
#     model.add(ZeroPadding1D(padding=((0,1))))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.25))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.25))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))

#     return model

# def build_generator():
#     '''
#     Put together a model that takes in one-dimensional noise and outputs two-dimensional
#     data representing a black and white image, with -1 for black and 1 for white.
    
#     returns: the model object
#     '''

#     noise_shape = (noise_len,)
    
#     model = Sequential()

#     model.add(Dense(img_cols*16*2, activation="relu", input_shape=noise_shape))
#     model.add(Reshape((13, 128)))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(128, kernel_size=3, padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(momentum=0.8)) 
#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(64, kernel_size=3, padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Conv1D(300, kernel_size=3, padding="same"))
#     model.add(Activation("tanh"))

#     return model

# def build_combined():
#     '''
#     Puts together a model that combines the discriminator and generator models.
    
#     returns: the generator, discriminator, and combined model objects
#     '''
    
#     optimizer = Adam(0.0002, 0.5)

#     # Build and compile the discriminator
#     discriminator = build_discriminator()
#     discriminator.compile(loss='binary_crossentropy', 
#                           optimizer=optimizer,
#                           metrics=['accuracy'])


#     # Build and compile the generator
#     generator = build_generator()
#     generator.compile(loss='binary_crossentropy', optimizer=optimizer)

#     # The generator takes noise as input and generates images
#     noise = Input(shape=(noise_len,))
#     img = generator(noise)
#   #  print(len(img))
#   #  print(len(img[0]))
#   #  print(len(img[0][0]))
#     # For the combined model we will only train the generator
#     discriminator.trainable = False

#     # The discriminator takes generated images as input and determines validity
#     valid = discriminator(img)

#     # The combined model  (stacked generator and discriminator) takes
#     # noise as input => generates images => determines validity 
#     combined = Model(inputs=noise, outputs=valid)
#     combined.compile(loss='binary_crossentropy', optimizer=optimizer)
#     return generator, discriminator, combined

# def save_imgs(generator, epoch):
#     '''
#     Has the generator create images and saves the images in a single file that includes
#     the epoch in the filename.
    
#     inputs:
#         generator: the generator model object returned by build_combined
#         epoch: the epoch number (but can be anything that can be represented as a string)
    
#     returns: None
#     '''
#     r, c = 5, 5
#     noise = np.random.normal(0, 1, (r * c, noise_len))
#    # noise = np.random.normal(0, 1, noise_len).reshape(100,)
    
#     gen_img = generator.predict(noise)
#     #np.savetxt((str(epoch) + ".csv"), gen_img, delimiter=",")
#     pickle.dump( gen_img, open("TTestNowFixedChecking/"+(str(epoch) + ".p"), "wb" ) )
    
# #  # Rescale images 0 - 1
# #  gen_imgs = 0.5 * gen_imgs + 0.5

# #  fig, axs = plt.subplots(r, c)
# #  #fig.suptitle("DCGAN: Generated digits", fontsize=12)
# #  cnt = 0
# #  for i in range(r):
# #      for j in range(c):
# #          axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
# #          axs[i,j].axis('off')
# #          cnt += 1
# #  fig.savefig(os.path.join(images_dir, 'mnist_{}.png'.format(epoch)))
# #  plt.close()

# def train(generator, discriminator, combined, epochs, batch_size=128, save_interval=50):
#     '''
#     Trains all model objects
    
#     generator: the generator model object returned by build_combined
#     discriminator: the discriminator model object returned by build_combined
#     combined: the combined model object returned by build_combined
#     epochs: integer, the number of epochs to train for
#     batch_size: integer, the number of training samples to use at a time
#     save_interval: integer, will generate and save images when the current epoch % save_interval is 0
    
#     returns: None
#     '''

#     # Load the dataset
# #   (X_train, _), (_, _) = mnist.load_data()

# #   # Rescale -1 to 1
# #   X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# #   X_train = np.expand_dims(X_train, axis=3)
# #   
#     X_train = fullMainArray
# #  ################################################################################   X_train = sampleSentence
#     half_batch = int(batch_size / 2)

#     for epoch in range(epochs):

#         # ---------------------
#         #  Train Discriminator
#         # ---------------------

#         # Select a random half batch of images
#         idx = np.random.randint(0, X_train.shape[0], half_batch)
#         imgs = X_train[idx]

#         # Sample noise and generate a half batch of new images
#         noise = np.random.normal(0, 1, (half_batch, noise_len))
#         gen_imgs = generator.predict(noise)

#         # Train the discriminator (real classified as ones and generated as zeros)
#         d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
#       #  print(imgs) 
#       #  print(np.ones((half_batch, 1)))
#       #  print(gen_imgs)
#         d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

#         # ---------------------
#         #  Train Generator
#         # ---------------------

#         noise = np.random.normal(0, 1, (batch_size, noise_len))

#         # Train the generator (wants discriminator to mistake images as real)
#         g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
 
#         # If at save interval => save generated image samples and plot progress
#         if epoch % save_interval == 0:
#             # Plot the progress
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#             print ("{} [D loss: {}, acc.: {:.2%}] [G loss: {}]".format(epoch, d_loss[0], d_loss[1], g_loss))
#             save_imgs(generator, epoch)
            
# def show_new_image(generator):
#     '''
#     Generates and displays a new image
    
#     inputs: generator object model returned from build_combined
    
#     returns: generated image
#     '''
    
#     noise = np.random.normal(0, 1, (1, noise_len))
#     gen_img = generator.predict(noise)[0][:,:,0]
    
#     return plt.imshow(gen_img, cmap='gray', vmin=-1, vmax=1)


# In[21]:


#np.random.randint(0, fullArray.shape[0], 64)


# In[22]:


# set up directories to hold the images that are saved during training checkpoints.
import os
images_dir = "testOut"
if (not os.path.isdir(images_dir)):
    os.mkdir(images_dir)


# In[23]:


generator, discriminator, combined = build_combined()


# In[24]:


train(generator, discriminator, combined, epochs=16001, batch_size=32, save_interval=50)


# In[25]:


# Uncomment to save your model files
generator.save('TTextFullergenerator.h5')
discriminator.save('TTextFullerdiscriminator.h5')
combined.save('TTextFullercombined.h5')


# ### Success Check:

# In[10]:


testing1600now = pickle.load( open( "/Users/cevbain/MachineLearningBase/TTestNowFixedChecking/16000.p", "rb" ) )


# In[11]:


testing1600now
for i in range(0,52):
    sampleSentence[0][i] = modelnow[testing1600now[i]]


# In[16]:


#Control
for i in range(0,52):
    print(modelnow.similar_by_vector(fullMainArray[1][i], topn=1))


# In[15]:


#Check output
for i in range(0,52):
    print(modelnow.similar_by_vector(testing1600now[1][i], topn=1))


# In[ ]:





# In[37]:





# ---
# ## Loading Data/Transformation post word speperation code || PRESERVE!

# In[3]:


from gensim.models import Word2Vec


# In[8]:


firstCsv = pd.read_csv('/Users/cevbain/Documents/TrumpVeryDown.csv')


# In[34]:


k = np.zeros((55,300))
for s in range(0,55):
    k[s] = model[secondCsv.iloc[[4],[s]]]


# In[72]:


testtestArray = pickle.load( open('TTestChecking/12000.p', "rb" ) )


# In[9]:


secondCsv = pd.read_csv('/Users/cevbain/Documents/outFancy3.csv')
secondCsv = pd.read_csv('/Users/cevbain/Documents/outFancy3.csv')


# In[10]:


#USED
fullArrayFinal = np.zeros((8431,55,300))
for r in range(0,8431):
    for s in range(0,52):
#         print(secondCsv[str(s)][r])
        try:
            fullArrayFinal[r][s] = model[secondCsv[str(s)][r]]
        except:
            fullArrayFinal[r][s] = np.zeros((300))


pickle.dump(fullArrayFinal, open("TrumpPostFULLSave3.p", "wb" ) ) #The 2 is for safety


# In[22]:


# Non Functional Bad!!!!
fullArray = np.zeros((8431,55,300))
for r in range(0,8431):
    for s in range(0,52):
        fullArrayFinal[r][s] = modelnow[secondCsv.iloc[[r],[s]]]


# ----
# ## Testing The Concept

# In[22]:


print(modelnow.similar_by_vector(modelnow["it"], topn=1))


# In[81]:


pickle.dump( fullArray, open( "TrumpPostTextSave.p", "wb" ) )


# In[12]:


#  model.predict(testtestArray) #will fail


# In[5]:


#model2 = Word2Vec.load('wiki-news-300d-1M.vec')


# In[14]:


for i in range(0,52):
    print(modelnow.similar_by_vector(testtestArray[1][i], topn=1))


# In[5]:


modelnow.similar_by_vector(np.zeros((300)), topn=1)


# In[1]:


testArray = "The fancy novel name of book by this guy is built entirely around the fancy phrase and its inventor. It depicts a fictional country off the south area coast that idealizes the pangram, chronicling the effects on literature and social structure as various letters are banned from daily use by government use"


# In[2]:


testArray2 = "the","fancy","novel","name","of","book","by","this","guy","is","built","entirely","around","the","fancy","phrase","and","its","inventor","it","depicts","a","fictional","country","off","the","south","area","coast","that","idealizes","the","thing","chronicling","the","effects","on","literature","and","social","structure","as","various","letters","are","banned","from","daily","use","by","government","use"


# In[5]:


testArray3 = np.array(["as","they","rounded","a","bend","in","the","path","that","ran","beside","the","river","they","recognized","the","silhouette","of","a","fig","tree","atop","a","nearby","hill","the","weather","was","hot","and","the","days","were","long","the","fig","tree","was","in","full","leaf","but","not","yet","bearing","fruit","soon","they","spotted","other","landmarks","too"])


# In[6]:


testArray3 = np.array(["as","they","rounded","a","bend","in","the","path","that","ran","beside","the","river","they","recognized","the","silhouette","of","a","fig","tree","atop","a","nearby","hill","the","weather","was","hot","and","the","days","were","long","the","fig","tree","was","in","full","leaf","but","not","yet","bearing","fruit","soon","they","spotted","other","landmarks","too"])
sampleSentence = np.zeros((1,52,300))
for i in range(0,52):
    sampleSentence[0][i] = modelnow[testArray3[i]]


# In[18]:


for i in sampleSentence[0]:
    print(modelnow.similar_by_vector(i, topn =1))


# In[17]:


for i in testtestArray[1]:
    print(modelnow.similar_by_vector(i, topn =1))


# In[16]:


testtestArray[1][1]


# In[11]:


for i in testArray3:
    print(modelnow.similar_by_vector(modelnow[i], topn =1))


# In[8]:


word_vectors = KeyedVectors.load_fastText_format('cc.en.300.bin', binary=True)
outModel = word2vec.load_fasttext_format()


# In[ ]:


#outModel = load_vectors("/Users/cevbain/Downloads/wiki-news-300d-1M.vec")
#modelMain = fasttext.load_model("/Users/cevbain/MachineLearning/cc.en.300.bin")


# In[9]:


# outModel.similar_by_vector(new, topn=1, restrict_vocab=None)


# In[15]:


model["m"]


# -------
# ## Looking At String Scrubbing

# In[56]:


import string

# Thanks to Martijn Pieters for this improved version

# This uses the 3-argument version of str.maketrans
# with arguments (x, y, z) where 'x' and 'y'
# must be equal-length strings and characters in 'x'
# are replaced by characters in 'y'. 'z'
# is a string (string.punctuation here)
# where each character in the string is mapped
# to None
translator = str.maketrans('', '', string.punctuation)

# This is an alternative that creates a dictionary mapping
# of every character from string.punctuation to None (this will
# also work)
#translator = str.maketrans(dict.fromkeys(string.punctuation))

s = 'string with "punctuation" inside of it! Does this work? I hope so.'

# pass the translator to the string's translate method.
print(s.translate(translator))


# In[66]:


string.punctuation


# In[75]:


import string

translator2 = str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')

def scrub (stringIn):
    h = stringIn
    try:
        h = h.translate(translator2)
    except:
        return float('nan')
#     h = h.lower()
    return h


# In[ ]:





# In[ ]:


# fullArray = np.zeros((8431,55,300))
for r in range(0,1):
    for s in range(0,55):
    secondCsv[str(s)][5]
        print(modelnow.similar_by_vector(model[secondCsv[str(s)][5]], topn=1))
# secondCsv["0"][5]


# In[76]:


# fullArray = np.zeros((8431,55,300))
#for r in range(0,8431):
for s in range(0,55):
    secondCsv[str(s)][5]
    print(modelnow.similar_by_vector(model[secondCsv[str(s)][5]], topn=1))
# secondCsv["0"][5]

