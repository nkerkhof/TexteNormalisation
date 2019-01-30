# -*- coding: utf-8 -*-

import numpy as np
import time
np.set_printoptions(threshold=np.inf)

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras


####TODO: Vectoletter; essayer un reseau de neurones ####
######ATTENTION A LA MEMOIRE#####

alpha = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéèàùç€%$+*-!?.,:;&\"'()[]/=0123456789\n\t"

def delStartAndEnd(filename, result_filename):
    f = open(filename,"r")
    lines = f.readlines()
    f.close()
    f = open(result_filename,"w")
    
    i=0
    
    while lines[i].startswith("*** START OF") != True :
        i = i+1
    
    i = i+1
    
    while lines[i].startswith("*** END OF") != True :
        f.write(lines[i])
        i = i+1
        
    f.close()

def loadFiletxt(filename):
	with open(filename, 'r') as f:
		return f.read()


def sentencePerLine(txt):
	index = 1
	while index < len(txt)-1:
		if txt[index] == '\n':
			if (txt[index-1] == '.' or txt[index-1] == '!' or txt[index-1] == '?') and txt[index+1]=='\n':
				txt = txt[:index]+txt[index+1:]
				index = index + 2
			else:
				txt = txt[:index]+" "+txt[index+1:]
				index = index + 1
		else:
			index = index + 1
	if txt[0]=='\n':
		txt = txt[1:]
	if txt[-1]=='\n':
		txt = txt[:-1]
	return txt
			
		
def delEmptyLines(txt):
	index = 0
	while index < len(txt)-1:
		if txt[index]=='\n' and txt[index+1]=='\n':
			txt = txt[:index+1]+txt[index+2:]
		index = index+1
	if txt[-2]=='\n' and txt[-1]=='\n':
		txt = txt[:-1]
	return txt
			


def letterToVec(letter):
	alpha = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéèàùç€%$+*-!?.,:;&\"'()[]/=0123456789\n\t"
	sizeAB = len(alpha)
	final = np.zeros(sizeAB+1)
	place = -1
	for l in range(sizeAB):
		if letter == alpha[l]:
			final[l] = 1
			place = l
			return place
	if place==-1:
		final[-1] = 1
		return sizeAB
	#return final.tolist()

def vecToLetter(vect):
  alpha = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéèàùç€%$+*-!?.,:;&\"'()[]/=0123456789\n\t"
  index = vect.index(1)
  if index < len(alpha):
     return alpha[index]
  return ''

def txt_to_train_data(txt):
	#n_neighbors_letters=2
	result = [[letterToVec(-1),letterToVec(-1),letterToVec(txt[0]),letterToVec(txt[1]),letterToVec(txt[2])], [letterToVec(-1),letterToVec(txt[0]),letterToVec(txt[1]),letterToVec(txt[2]),letterToVec(txt[3])]]
	for index in range(2, len(txt)-2):
		result.append([[letterToVec(txt[index-2]), letterToVec(txt[index-1]), letterToVec(txt[index]), letterToVec(txt[index+1]), letterToVec(txt[index+2])]])
	result.append([[letterToVec(txt[-4]), letterToVec(txt[-3]), letterToVec(txt[-2]), letterToVec(txt[-1]), letterToVec(-1)], [letterToVec(txt[-3]), letterToVec(txt[-2]), letterToVec(txt[-1]), letterToVec(-1), letterToVec(-1)]])
	return np.array(result)



# python src/v1.py
#  bash irisa-text-normalizer-master/bin/fr/generic-normalisation.sh data/test/test_after.txt > data/test/test_after_norm1.txt
# perl irisa-text-normalizer-master/bin/fr/specific-normalisation.pl irisa-text-normalizer-master/cfg/nlp_modified.cfg data/test/test_after_norm1.txt > data/test/test_after_norm2.txt
#entrée : path_origine / path_dest_txt_test_after / path_dest_txt_test_after_norm2

#start = time.time()
#delStartAndEnd("8693-0-tokenized.txt","8692_v1.txt")
#txt = loadFiletxt("8692_v1.txt")
#txt = sentencePerLine(txt)
#txt = delEmptyLines(txt)


f = open("data/test/test_after_petit.txt","r")
lines_before = f.readlines()
f.close()

f = open("data/test/after_all_petit.txt","r")
lines_after = f.readlines()
f.close()


max_car_before = 0
for l in lines_before:
	if len(l)>max_car_before :
		max_car_before = len(l)

max_car_after = 0
for l in lines_after:
	if len(l)>max_car_after :
		max_car_after = len(l)

num_decoder_tokens = len(alpha)+1
num_encoder_tokens = len(alpha)+1
latent_dim = 128
batch_size = 16
epochs = 10

encoder_input_data = np.zeros((len(lines_before), max_car_before, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(lines_after), max_car_after, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(lines_after), max_car_after, num_decoder_tokens), dtype='float32')

for i_line in range(len(lines_before)):
	for i_car in range(len(lines_before[i_line])):
		encoder_input_data[i_line][i_car][letterToVec(lines_before[i_line][i_car])] = 1

for i_line in range(len(lines_after)):
	for i_car in range(len(lines_after[i_line])):
		decoder_input_data[i_line][i_car][letterToVec(lines_after[i_line][i_car])] = 1
	for i_car in range(1,len(lines_after[i_line])):
		decoder_target_data[i_line][i_car-1][letterToVec(lines_after[i_line][i_car])] = 1
	decoder_target_data[-1][-1][letterToVec(lines_after[-1][-1])] = 1

print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_target_data.shape)




encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)







