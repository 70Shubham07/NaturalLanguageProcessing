# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.tsv')#delimiter = '\t', quoting = 3)


# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000): #38932):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Description'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:1000, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)





# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
# fix random seed for reproducibility
numpy.random.seed(7)


'''


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
top_words=1500
max_review_length = 600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32

def create_model(embedding_vecor_length = 32, filters = 32, activation_dense = 'tanh', pool_size = 2, dropout = 0.2, recurrent_dropout = 0.2, LSTMparaOne = 100,
                 batch_size = 40, epochs = 10):
    
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(LSTM(LSTMparaOne, dropout = dropout, recurrent_dropout = recurrent_dropout))

    model.add(Dense(1, activation=activation_dense))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return(model)


# Hyper parameters to tune
embedding_vecor_length = [28,30, 32, 34, 36]
filters = [28, 30, 32, 34, 36]
activation_dense = ['tanh','sigmoid', ' softmax', 'relu', 'linear','softplus']
pool_size = [2,4,6]
dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
recurrent_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
LSTMparaOne = [100,150,200,250]
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
# optimizer = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'Adam', 'Adamax', 'Nadam']



# Creating the param_grid dictionary.
param_grid = dict(embedding_vecor_length = embedding_vecor_length, filters=filters, activation_dense=activation_dense,
                  pool_size = pool_size, dropout = dropout, recurrent_dropout = recurrent_dropout, LSTMparaOne = LSTMparaOne,
                  batch_size = batch_size, epochs = epochs) # optimizer = optimizer)
 

from sklearn.model_selection import GridSearchCV


model = KerasClassifier(build_fn=create_model)
grid_search = GridSearchCV(estimator = model, param_grid = param_grid)
grid_result = grid_search.fit(X_train, y_train)


best_params = grid_result.best_params_
best_accuracy = grid_result.best_score_

# model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model

optim_model = create_model(embedding_vecor_length=best_params['embedding_vecor_lenght'],
                           filters = best_params['filters'], activation = best_params['activation_dense'], pool_size = best_params['pool_size'],
                           dropout = best_params['dropout'], recurrent_dropout = best_params['recurrent_dropout'], LSTMparaOne = best_params['LSTMparaOne'],
                           batch_size = best_params['batch_size'], epochs = best_params['epochs']) #optimizer = best_params['optimizer'])
optim_model.fit(X_train, y_train)
scores = optim_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


'''

dset=pd.read_csv('test.tsv')
corp = []
for j in range(0,29404):
#{
    rev = re.sub('[^a-zA-Z]', ' ', dset['Description'][j])
    rev = rev.lower()
    rev = rev.split()
    ps = PorterStemmer()
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))]
    rev = ' '.join(rev)
    corp.append(rev)
#}


# from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corp).toarray()

x = sequence.pad_sequences(x, maxlen=600)
y_test_predicted = model.predict(x)

y_test_predicted[y_test_predicted<0.5] = 0
y_test_predicted[y_test_predicted >= 0.5] = 1
'''
# y_test_predicted[y_test_predicted<=0.5] = 0
# y_test_predicted[y_test_predicted > 0.5] = 1
'''

y_test_pred = y_test_predicted.reshape(-1)
s = list(map(str, y_test_pred))
y_test_pr = np.array(s)

y_test_pr[y_test_pr == '0.0'] = 'happy'
y_test_pr[y_test_pr == '1.0']= 'not happy'
dset['Is_Response'] = y_test_pr
df = dset.copy()
for i in ['Description','Browser_ID','Device_Used']:
#{
    del df[i]
#}

dset.csv_write('TEST.csv', index = False)


'''






