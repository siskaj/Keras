
# coding: utf-8

# In[2]:

from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[3]:

train_data[0]


# In[4]:

train_labels[0]


# In[5]:

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])


# In[6]:

decoded_review


# In[9]:

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[10]:

x_train[0]


# In[11]:

x_train[1]


# In[12]:

x_train.shape


# In[14]:

x_train[0,15]


# In[17]:

seqs = np.random.randint(100, size=(6,5))


# In[18]:

seqs


# In[ ]:



