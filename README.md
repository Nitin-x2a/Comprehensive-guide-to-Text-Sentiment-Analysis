In this notebook, I have attempted to accomodate all recipes that can be used for Text Sentiment Classification. I will start by downloading the raw data, then process it, and finally train various Machine Learning and Deep Learning models on it. 

This notebook can be used as reference for all kinds of binary sentiment classification tasks. It aims to cover as much as possible, but for every model covered in it, there is still good scope for performance improvement, by working further on hyperparameter tuning.

# Data Preprocessing

Refined data for IMDB reviews is avilable, but since I want this notebook to be as comprehensive as possible, I will download raw data and then process it step-by-step.

## Downloading IMDB data


```python
import requests

url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target_path = 'aclImdb_v1.tar.gz'

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(target_path, 'wb') as f:
        f.write(response.raw.read())
```


```python
import tarfile
tar = tarfile.open("aclImdb_v1.tar.gz")
tar.extractall()
tar.close()
```

The extracted .tar file generated two folders - 'Train', and 'Test'. Each folder contains 25,000 text files each, grouped into two folders 'pos' and 'neg', that contain 12,500 files each.

The following snippet reads from one of the folders, and creates a list of texts, and the corrsponding list of binary targets.


```python
import glob 
classes = {'pos':1, 'neg':0}

def read_txt(file_path):
  with open(file_path, 'r') as file:
    text = file.read()
  file.close()
  return text

def populate(main_folder):
  all_txts, all_sentiments = [], []
  for class_ in classes:
    directory = "aclImdb/{}/{}".format(main_folder, class_)
    file_paths = glob.glob(directory + '/*.txt')
    txts = [read_txt(path) for path in file_paths]
    sentiments = [classes[class_] for _ in range(len(txts))]
    all_txts.extend(txts)
    all_sentiments.extend(sentiments)
  return all_txts, all_sentiments
```


```python
X_train, y_train = populate('train')
X_test, y_test = populate('test')
```


```python
print(len(X_train))
print(len(X_test))
```

## Data Cleaning


```python
print(X_train[10])
```

We need to do three things first:

(1) Get rid of the HTML tags (characters enclosed by <>)

(2) Get rid of numbers and other special characters

(3) Lowercase all the remaining alphabet characters


```python
import re

extra_chars = re.compile("[0-9.;:!\'?,%\"()\[\]]")
html_tags = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def clean(texts):
    texts = [extra_chars.sub("", text.lower()) for text in texts]
    texts = [html_tags.sub(" ", text) for text in texts]
    return texts
```


```python
X_train = clean(X_train)
X_test = clean(X_test)
```

## Shuffling Data

Recall that in both our extracted train and test data, we have first 12,500 samples belonging to 'pos' sentiment, and the remaining 12,500 samples to the 'neg' sentiment. Our models will do better if we trained on data where the samples follow a mixed order.


```python
import random

def shuffle_set(X, y):
  all_data = list(zip(X, y))
  random.shuffle(all_data)
  X_shuff, y_shuff = [list(item) for item in zip(*all_data)]
  return X_shuff, y_shuff

X_train, y_train = shuffle_set(X_train, y_train)
X_test, y_test = shuffle_set(X_test, y_test)
```

## Removing stopwords

We will now get rid of common words such as 'the', 'of', 'at' etc (known as stopwords). They don't really contribute much useful information and also their removal will reduce our feature size.


```python
import nltk
nltk.download('stopwords')
```


```python
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 

def filter_text(text):
  words = text.split()
  return ' '.join([w for w in words if w not in stop_words])
```


```python
X_train = [filter_text(text) for text in X_train]
X_test = [filter_text(text) for text in X_test]
```

## Lemmatization

For training our models, all words will have to be encoded with some sort of numerical representation. The size of that word-to-number vocabulary would be huge. We can reduce it to some extent by lemmatizing the words. 

rocks ---> rock, 

better ---> good, 

walked ---> walk.

Note that Stemming is another popular method.

Read more - https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8


```python
nltk.download('wordnet')
```


```python
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def lemmatize(text):
  words = text.split()
  return ' '.join([lemmatizer.lemmatize(w) for w in words])
```


```python
X_train = [lemmatize(text) for text in X_train]
X_test = [lemmatize(text) for text in X_test]
```


```python
print(X_train[0])
```

# Machine Learning models

Before, we can fit or train any model on our data, we need to transform all the textual data into numerical form. The most common is one-hot encoding, where each text is represented by a sequence of binary data. The length of every sequence is equal to the size of the entire vocabulary. 

Each position in any sequence corresponds to a particular word. '1' tells you that the word is present in that text, and '0' denotes its absence. It is important to note that here the order of the words does not matter. We are simply treating any text sequence as a bag-of-words (BOW).

## One-Hot Encoding


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)

vectorizer.fit(X_train)

X_train_onehot = vectorizer.transform(X_train)

X_test_onehot = vectorizer.transform(X_test)
```


```python
print(X_train_onehot.shape)
print(X_test_onehot.shape)
```

Let's have a look at 20 words from the word vocabulary with their corresponding indices in the vocabulary.


```python
word_dict = vectorizer.vocabulary_

print({k: word_dict[k] for k in list(word_dict)[:20]})
```

We will start with the simple Naive Bayes models. They are simple probabilistic models that make use of the Bayes theorem.

Learn more - https://www.youtube.com/watch?v=EGKeC2S44Rs

## Naive Bayes 

We will fit two Naive Bayes models - Multinomial and Bernoulli. Bernoulli models the presence/absence of a feature. Multinomial models the number of counts of a feature. Recall that we have one-hot features currently. So, this data does not give a fair chance to Multinomial NB right now, and we will return to it again later in the notebook.

### Multinomial

We will use two evaluation metrics thoughout this notebook. Accuracy is the ratio of the true predictions (postive or negative) and the total number of predictions made.

F1 score is the harmonic mean of precision and recall.

Read more - https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9


```python
from sklearn.metrics import accuracy_score, f1_score
```

Following code-snippet will be used everytime we fit a machine learning model on the training set, and then evaluate the metrics over the test set.


```python
def fit_and_test(classifier, X_train, y_train, X_test, y_test, only_return_accuracy=False):

  classifier.fit(X_train, y_train)

  y_hat = classifier.predict(X_test)

  print('accuracy:', accuracy_score(y_test, y_hat))

  if not only_return_accuracy:
    print('f1_score:', f1_score(y_test, y_hat))
```


```python
from sklearn.naive_bayes import MultinomialNB
```


```python
mnb = MultinomialNB()
fit_and_test(mnb, X_train_onehot, y_train, X_test_onehot, y_test)
```

### Bernoulli


```python
from sklearn.naive_bayes import BernoulliNB
```


```python
bnb = BernoulliNB()
fit_and_test(bnb, X_train_onehot, y_train, X_test_onehot, y_test)
```

Now, we will move to linear models. Simply put, in any linear model, the target variable is just some form of linear combination of input features. In logistic regression, we have the logistic aka sigmoid function at the output side, that squishes the output to between 0 and 1. When we keep a threshold (usually equal to 0.5), we can obtain binary values, 0 and 1. This is how regression is used for binary classification tasks.

Read more - https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

## Linear Models

### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
```

We have a regularisation hyperparameter 'c', that we will grid search over.


```python
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:

  lr = LogisticRegression(C=c, max_iter=1000)

  print (f'At C = {c}:-', end=' ')

  fit_and_test(lr, X_train_onehot, y_train, X_test_onehot, y_test, True)
```


```python
lr_best = LogisticRegression(C=0.05, max_iter=1000)
fit_and_test(lr_best, X_train_onehot, y_train, X_test_onehot, y_test)
```

SGD-Regression is also a simple linear model that is basically regression as well. The only difference is in the training algorithm. Logistic Regression employs any of the ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers to help its parameter learn better values. SGDRegressor, as its name suggests, makes use of the more commonly known Stochastic gradient descent algorithm. The same is used by neural networks in their backpropagation process as well.

### SGDRegressor


```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

fit_and_test(sgd, X_train_onehot, y_train, X_test_onehot, y_test)
```

## K-Nearest Neighbours

KNN is a very simple model that simply stores its training data as points in a feature space. For making predictions for any new data point, it maps that data point onto that stored feature space, selects a certain number of closest neighbours, and returns the average value of them.

This number of neighbours is a hyperparameter, which we will grid search over.


```python
from sklearn.neighbors import KNeighborsClassifier

neighbours = [10, 20, 50, 100, 500]

for k in neighbours:

  knn = KNeighborsClassifier(n_neighbors=k)

  print (f'At K = {k}:-', end=' ')

  fit_and_test(knn, X_train_onehot, y_train, X_test_onehot, y_test, True)
```


```python
knn_best = KNeighborsClassifier(n_neighbors=50)

fit_and_test(knn_best, X_train_onehot, y_train, X_test_onehot, y_test)
```

## Support Vector Classifier

> In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well (look at the below snapshot).

Read more - https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/


```python
from sklearn.svm import LinearSVC
```


```python
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:

  svc = LinearSVC(C=c, max_iter=5000)

  print (f'At C = {c}:-', end=' ')

  fit_and_test(svc, X_train_onehot, y_train, X_test_onehot, y_test, True)
```

Scikit-Learn also provides a classification_report where we can obtain all the evaluation metrics for each class in our data.


```python
from sklearn.metrics import classification_report
```


```python
svc_best = LinearSVC(max_iter=5000, C=0.01)

svc_best.fit(X_train_onehot, y_train)
y_hat = svc_best.predict(X_test_onehot)

report = classification_report(y_test, y_hat, output_dict=True)

print('positive: ', report['1'])
print('negative: ', report['0'])
```

## WordCount Features

Now, we will see another way of encoding text data into numbers. Instead of simply indicating the presence or absence of any word, we will represent its exact count it the text. We simply have to set binary = False in the CountVectorizer method that we used earlier.


```python
vectorizer = CountVectorizer(binary=False)

vectorizer.fit(X_train)

X_train_wc = vectorizer.transform(X_train)

X_test_wc = vectorizer.transform(X_test)
```


```python
mnb = MultinomialNB()
fit_and_test(mnb, X_train_wc, y_train, X_test_wc, y_test)
```


```python
lr = LogisticRegression(C=0.05, max_iter=1000)
fit_and_test(lr, X_train_wc, y_train, X_test_wc, y_test)
```


```python
svc = LinearSVC(max_iter=5000, C=0.01)
fit_and_test(svc, X_train_wc, y_train, X_test_wc, y_test)
```

## n-gram features

We will go to the next level now. Inplace of using each word as a feature, we can use combinations of words as features. Such representation will help us to capture the information about which words appear together and how they affect the overall sentiment. 

For example, consider the text "That action scene was terribly enthrilling". 

Now, a model based on a plain one-word BOW models may view the word "terribly" as a negative indicator, but a bi-gram data (n=2) will correctly interpret the usage of that word in its context "terribly enthrilling". Note that  we are still not considering any review text in its entire sequence, but still n-grams representations have more contextual information comparitively.


```python
vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))

vectorizer.fit(X_train)

X_train_2gram = vectorizer.transform(X_train)

X_test_2gram = vectorizer.transform(X_test)
```

Setting ngram_range to (1, 2) creates both single word and two consecutive word features. If one only wants to have bi-gram fearures, they have to set the range to (2, 2).


```python
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:

  lr = LogisticRegression(C=c, max_iter=1000)

  print (f'At C = {c}:-', end=' ')

  fit_and_test(lr, X_train_2gram, y_train, X_test_2gram, y_test, True)
```


```python
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:

  svc = LinearSVC(C=c, max_iter=5000)

  print (f'At C = {c}:-', end=' ')

  fit_and_test(svc, X_train_2gram, y_train, X_test_2gram, y_test, True)
```

## tf-idf features

The last BOW represenation that we will consider is term frequency–inverse document frequency (tf-idf), that indicates how important a word is to a document in a collection or corpus. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

X_train_tf = vectorizer.transform(X_train)

X_test_tf = vectorizer.transform(X_test)
```


```python
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:

  lr = LogisticRegression(C=c, max_iter=1000)

  print (f'At C = {c}:-', end=' ')
  
  fit_and_test(lr, X_train_tf, y_train, X_test_tf, y_test, True)
```


```python
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:

  svc = LinearSVC(C=c, max_iter=5000)

  print (f'At C = {c}:-', end=' ')

  fit_and_test(svc, X_train_tf, y_train, X_test_tf, y_test, True)
```

# Deep Learning models

In all the deep learning models, we will have to make use of textual data sequentially. Each word can be represented by a vector of a certain length. The representation has to be meaningful in such a way that each vector captures a dimension of the word's meaning, and semantically similar words have similar vectors.

Read more - https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf

There are two ways to obtain these vectors (also called word-embeddings):-

(1) We can simply create a word-dictionary of all the existing words in our data, where each word is mapped with a number. Then we create an embedding matrix with the dimensions (num_words, length_of_each_embedding_vector). This matrix is essentially a look-up table, where for every word's index number in the dictionary, the matrix returns the embedding vector for that word. During the training process, the contents of this matrix are updated as well.

(2) In this method too, we have an embedding matrix. But this time, the matrix contains word-embeddings that are already trained on some other corpus. We may choose to either keep the values of this matrix forzen during the training process, or train it to tweak the values a little so as to suit our data better.

## Custom Word Embeddings

We will use the Tokenizer method of the Tensorflow package. It creates a word-dictionary and then maps the words to their index numbers. It has another functionality, using which we will restrict our vocabulary to the most frequent 5000 words in the data.


```python
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_NUM_WORDS = 5000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
X_train.extend(X_test)
tokenizer.fit_on_texts(X_train)
```

Let's have a look at the 20 most frequent words in our data.


```python
word_index = tokenizer.word_index

print([(w, i) for w, i in word_index.items()] [:20])
```


```python
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
```

For encoding words with index numbers, I have done so in batches, since the RAM size was not sufficient for operating on the entire data at once.


```python
batch_size = 128
l = len(X_train)

i = 0
while (i <= l-1):

  if (i + batch_size) >= (l-1):
    X_train[i:] = tokenizer.texts_to_sequences(X_train[i:])
  
  else:
    X_train[i:i+batch_size] = tokenizer.texts_to_sequences(X_train[i:i+batch_size])
  
  i += batch_size

X_train, X_test = X_train[:l//2], X_train[l//2:]
```


```python
print(X_train[10])
```

For faster and more convenient training of Neural Network models, we usually keep the training data uniform. For that, we will need to have the same sequence length for all our reviews. So, we will pad the sequences that are short and truncate the ones that are long. To choose a good sequence length, let's see the distribution of the length of reviews in our data.


```python
import matplotlib.pyplot as plt

seq_lengths = [len(seq) for seq in X_train]

plt.figure(figsize=(10, 4))
plt.hist(seq_lengths)
plt.show()
```

Most sequences have a length less than 200. LSTMs generally perform well up until sequences with 100 steps. I tried training LSTM models with various sequence lengths, and found that the number 120 gave good results.


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 120

X_train = pad_sequences(X_train, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre')

X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre')
```


```python
print(X_train.shape)
print(X_test.shape)
```


```python
print(X_train)
```


```python
import numpy as np
y_train = np.array(y_train)
y_test = np.array(y_test)
```

We will use two kinds of Neural Network Architectures - RNNs, and CNNs. Also, in RNNs, LSTMs will be used in particular. I am presuming you are familiar with the working of these models.

In case you aren't, here is a good place to start (follow the order):

(1) https://pathmind.com/wiki/neural-network

(2) https://pathmind.com/wiki/lstm

(3) https://pathmind.com/wiki/convolutional-network

## Recurrent Neural Networks

### Vanilla LSTM network


```python
#Embedding matrix first dimension
V = num_words

#Embedding matrix second dimension
D = 50

#Hidden state length
M = 100

#Number of steps
T = MAX_SEQ_LEN
```


```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.models import Model
```


```python
i = Input(shape=(T,))
x = Embedding(V, D)(i)
x = LSTM(M)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.summary()
```

I have used a callback method where the model checks the evalution metric over the validation set after every epoch, and if the metric is better than that achieved in any of the previous epochs, then the model (with its weights) right after that epoch are saved. I have found this to be a convenient way to prevent over-fitting.


```python
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
```

The code for compiling, training, and testing is the same for all models in this section.


```python
def train_and_test(model, label, batch_size, epochs):

  save_at = label + ".hdf5"

  save_best = ModelCheckpoint(save_at, monitor='val_loss', verbose=1, 
                              save_best_only=True, save_weights_only=False, mode='min')

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  s = len(X_test)//2

  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
            validation_data=(X_test[:s], y_test[:s]), callbacks=[save_best])

  trained_model = load_model(save_at)
  y_hat = model.predict(X_test[s:])
  y_hat  = (y_hat > 0.5)*1

  print('\n')
  print('-'*100)
  print(f"Test Results for '{save_at}' model")
  print('accuracy:', accuracy_score(y_test[s:], y_hat))
  print('f1_score:', f1_score(y_test[s:], y_hat))
```


```python
train_and_test(model, 'simple_lstm', batch_size=128, epochs=3)
```

### Using all Hidden States

Now we will create the LSTM model, where we will concatenate the hidden states from all the steps,and find the average of each unit in the hidden state vector, using the GlobalAvergePooling layer. One can also try using the GlobalMaxPooling layer instead.

In the previous LSTM model, only the last hidden state was used for making classification. Even though, the last hidden state is dependent on all the previous hidden states as well, the dependence is usually weak with hidden states from the initial steps. By concatenating them, our model has access to all the hidden states for making classifications.

This can be very easily done by setting 'return_sequences' to 'True' in the LSTM layer.


```python
from tensorflow.keras.layers import GlobalAveragePooling1D
```


```python
i = Input(shape=(T,))
x = Embedding(V, D)(i)
x = LSTM(M, return_sequences=True)(x)

x = GlobalAveragePooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.summary()
```


```python
train_and_test(model, 'lstm_all_hidden_states', batch_size=128, epochs=5)
```

### Bidirectional LSTM

The bidirectional model has two hidden states, one over which the sequence is run in its original sense, and another hidden state, over which the sequence is run backwards. The two states are then concatenated.


```python
from tensorflow.keras.layers import Bidirectional
```


```python
i = Input(shape=(T,))
x = Embedding(V, D)(i)

x = Bidirectional(LSTM(M, return_sequences=True))(x)

x = GlobalAveragePooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.summary()
```


```python
train_and_test(model, 'lstm_bidirectional', batch_size=128, epochs=5)
```

## Convolutional Neural Networks

CNNs are more commonly used for image data (Conv2D layers). For time sequence data which is one-dimensional in nature, we can use a Conv1D layer.


```python
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
```


```python
i = Input(shape=(T,))
x = Embedding(V, D)(i)

x = Conv1D(16, 3, activation='relu')(x)
x = MaxPooling1D()(x)
x = BatchNormalization()(x)

x = Conv1D(16, 3, activation='relu')(x)
x = MaxPooling1D()(x)
x = BatchNormalization()(x)

x = Conv1D(16, 3, activation='relu')(x)
x = MaxPooling1D()(x)
x = BatchNormalization()(x)

x = GlobalAveragePooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.summary()
```


```python
train_and_test(model, 'cnn', batch_size=128, epochs=8)
```

## Pre-trained Glove Embeddings


```python
! mkdir glove
```

We will use the popular Glove word embeddings. The following snippet can be used for downloading them. This server however is not very stable. So, I have shared my Google Drive link of the embeddings.


```python
# import zipfile, io

# data_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
# r = requests.get(data_url)
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall('glove/')
```


```python
! conda install -y gdown
```


```python
import gdown

url = "https://drive.google.com/uc?id=18WgSks6St7KVDgY4Y2e29dHhEcD-9SWK"

output = 'glove/glove.6B.100d.txt'

gdown.download(url, output, quiet=False)
```

Glove embeddings of lengths 100, 200, and 300 are avilable. I have used the first one.


```python
EMBEDDING_DIM = 100

embeddings_index = {}
with open('glove/glove.6B.100d.txt') as f:
  for line in f:
    word, coeff = line.split(maxsplit=1)
    coeff = np.fromstring(coeff, 'f', sep=' ')
    embeddings_index[word] = coeff
```


```python
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  #(num_words, length of each word embedding)

for word, i in word_index.items():
  if i >= num_words:
    continue
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:               # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector
```

### Freezing the Embedding layer


```python
from tensorflow.keras.initializers import Constant
```


```python
#Embedding matrix second dimension
D = EMBEDDING_DIM
```

The embedding matrix be directly loaded into the embedding layer. We can freeze it by setting 'trainable' to 'False.


```python
i = Input(shape=(T,))

x = Embedding(V, D, 
              embeddings_initializer=Constant(embedding_matrix),
              trainable=False)(i)

x = LSTM(M)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.summary()
```


```python
train_and_test(model, 'lstm_glove', batch_size=128, epochs=10)
```

### FIne-tuning the the Glove Embeddings


```python
i = Input(shape=(T,))

x = Embedding(V, D, 
              embeddings_initializer=Constant(embedding_matrix),
              trainable=True)(i)

x = LSTM(M)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.summary()
```


```python
train_and_test(model, 'lstm_glove_trainable', batch_size=128, epochs=10)
```
