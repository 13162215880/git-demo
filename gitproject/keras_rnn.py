from __future__ import print_function
import numpy as np
from six.moves import cPickle
np.random.seed(1337)
from six.moves import zip
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Activation,Embedding
from keras.layers import LSTM

max_features = 20000
maxlen = 80
batch_size = 32

def preprocess_data(path,nb_words,maxlen):
	f = open(path,'rb')
	(x_train,labels_train),(x_test,labels_test) = cPickle.load(f)
	f.close()
	
	seed = 133
	np.random.seed(seed)
	np.random.shuffle(x_train)
	np.random.seed(seed)
	np.random.shuffle(labels_train)
	
	np.random.seed(seed * 2)
	np.random.shuffle(x_test)
	np.random.seed(seed * 2)
	np.random.shuffle(labels_test)

	xs = x_train + x_test
	labels = labels_train + labels_test

	start_char = 1
	oov_char = 2
	index_from = 3
	skip_top = 0
	if start_char is not None:
		xs = [[start_char] + [w + index_from for w in x] for x in xs]
	elif index_from:
		xs = [[w + index_from for w in x] for x in xs]
	if maxlen:
		new_xs = []
		new_labels = []
		for x,y in zip(xs,labels):
			if len(x) < maxlen:
				new_xs.append(x)
				new_labels.append(y)
		xs = mew_xs
		labels = new_labels
	if not xs:
		raise ValueError('After filtering for sequences shorter than maxlen=' + str(maxlen) + ',no sequence was kept.' 'Increase maxlen.')

	if not nb_words:
		nb_words = max([max(x) for x in xs])
	if oov_char is not None:
		xs = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in xs]
	else:
		new_xs = []
		for x in xs:
			nx = []
			for w in x:
				if w>= nb_words or w< skip_top:
					nx.append(w)
			new_xs.append(nx)
		xs = new_xs
	
	x_train = np.array(xs[:len(x_train)])
	y_train = np.array(labels[:len(x_train)])

	x_test = np.array(xs[len(x_train):])
	y_test = np.array(labels[len(x_train):])
	return (x_train,y_train),(x_test,y_test)

(X_train,y_train),(X_test,y_test) = preprocess_data('imdb_full.pkl',nb_words=max_features,maxlen=None)
print(len(X_train),'train sequences')
print(len(X_test),'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train,maxlen=maxlen)
X_test = sequence.pad_sequences(X_test,maxlen=maxlen)
print('X_train shape:',X_train.shape)
print('X_test shape:',X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features,128,dropout=0.2))
model.add(LSTM(128,dropout_W=0.2,dropout_U=0.2))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer= 'adam',metrics = ['accuracy'])







print('Train...')
model.fit(X_train,y_train,batch_size = batch_size,nb_epoch=15,validation_data=(X_test,y_test))

score,acc = model.evaluate(X_test,y_test,batch_size=batch_size)
print('Test score:',score)
print('Test accuracy:',acc)








