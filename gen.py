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
import sys
from train import permitted_chars, loadData,getModel,int_to_char,char_to_int

def main(weights_file):
    raw_text=loadData(["mity.txt",'nesbit-poszukiwacze-skarbu.txt'])

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(permitted_chars)
    print( "Total Characters: ", n_chars)
    print( "Total Vocab: ", n_vocab)

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



    #Load model from file
    model = getModel()
    model.summary()
    print('Loading',weights_file)
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in tqdm(range(1000)):
    	x = numpy.reshape(pattern, (1, len(pattern), 1))
    	x = x / float(n_vocab)
    	prediction = model.predict(x, verbose=0)
    	index = numpy.argmax(prediction)
    	result = int_to_char[index]
    	seq_in = [int_to_char[value] for value in pattern]
    	sys.stdout.write(result)
    	pattern.append(index)
    	pattern = pattern[1:len(pattern)]
    print ("\nDone.")
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('\nOne argument required: weights-improvement-file')
    else:
        main(sys.argv[1])
