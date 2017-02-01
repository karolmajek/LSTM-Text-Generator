#!/usr/bin/python3
from keras.models import Sequential,model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from tqdm import tqdm
import numpy
import json
import re


permitted_chars=[' ', ',', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'é', 'ó', 'Ą', 'ą', 'Ć', 'ć', 'Ę', 'ę', 'Ł', 'ł', 'ń', 'Ś', 'ś', 'ź', 'Ż', 'ż']

char_to_int = dict((c, i) for i, c in enumerate(permitted_chars))
int_to_char = dict((i, c) for i, c in enumerate(permitted_chars))

n_vocab = len(permitted_chars)
print( "Total Vocab: ", n_vocab)

def loadData(filelist):
    full_text=''

    for fname in filelist:
        raw_text = open(fname).read()
        raw_text=re.sub('[^'+''.join(permitted_chars)+']', ' ', raw_text)
        raw_text=re.sub("\s\s+", " ", raw_text)

        print(raw_text[:10000])

        full_text =full_text + raw_text.lower()
        print('File %s: %d'%(fname,len(raw_text)))
    print('Chars in dataset:',len(full_text))
    return full_text


def createModel(input_shape,output_shape):
    '''
    Function creates a model of a the network
    '''
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='softmax'))
    return model

def saveModel(model,fname='model.json'):
    '''
    Function saves the model to file
    '''
    json_string = model.to_json()
    with open(fname, 'w') as outfile:
        json.dump(json_string, outfile)
        print("Saved model to:",fname)



def getModel(model_file='model.json'):
    '''
    Function loads model from model.json file.
    '''
    with open(model_file, 'r') as f:
        model = model_from_json(json.load(f))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print("Model imported")
        return model
    return None

def main():
    raw_text=loadData(["mity.txt",'nesbit-poszukiwacze-skarbu.txt'])

    # summarize the loaded data
    n_chars = len(raw_text)
    print( "Total Characters: ", n_chars)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in tqdm(range(0, n_chars - seq_length, 1)):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print( "Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    print(y[0])

    #create the model
    model =createModel((X.shape[1], X.shape[2]),y.shape[1])

    #print summary
    model.summary()

    #save the model
    saveModel(model)

    #Load model from file
    model = getModel()
    model.summary()

    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',period=5)
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(X, y, nb_epoch=500, batch_size=128, callbacks=callbacks_list)
    # save weights
    model.save_weights('./model.h5')
    print("Saved model.h5")
if __name__ == '__main__':
    main()
