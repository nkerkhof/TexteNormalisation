# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(threshold=np.inf)

import time

from keras.models import Sequential
from keras.layers import Dense

####TODO: Vectoletter; essayer un reseau de neurones ####
######ATTENTION A LA MEMOIRE#####

def loadFiletxt(filename):
	with open(filename, 'r') as f:
		return f.read()


def letterToVec(letter):
	alpha = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéèàùç€%$+*-!?.,:;&\"'()[]/=0123456789\n"
	sizeAB = len(alpha)
	final = np.zeros(sizeAB+1)
	place = -1
	for l in range(sizeAB):
		if letter == alpha[l]:
			final[l] = 1
			place = l
	if place==-1:
		final[-1] = 1
	return final



def txt_to_train_data(txt):
	#n_neighbors_letters=2
	result = [[letterToVec(-1),letterToVec(-1),letterToVec(txt[0]),letterToVec(txt[1]),letterToVec(txt[2])], [letterToVec(-1),letterToVec(txt[0]),letterToVec(txt[1]),letterToVec(txt[2]),letterToVec(txt[3])]]
	for index in range(2, len(txt)-2):
		result.append([[letterToVec(txt[index-2]), letterToVec(txt[index-1]), letterToVec(txt[index]), letterToVec(txt[index+1]), letterToVec(txt[index+2])]])
	result.append([[letterToVec(txt[-4]), letterToVec(txt[-3]), letterToVec(txt[-2]), letterToVec(txt[-1]), letterToVec(-1)], [letterToVec(txt[-3]), letterToVec(txt[-2]), letterToVec(txt[-1]), letterToVec(-1), letterToVec(-1)]])
	return np.array(result)


x_train = loadFiletxt("data/tokenized/8692-0-tokenized.txt")+loadFiletxt("data/tokenized/8693-0-tokenized.txt")+loadFiletxt("data/tokenized/13737-0-tokenized.txt")
y_train = loadFiletxt("data/normalised/8692-0-normalised.txt")+loadFiletxt("data/normalised/8692-0-normalised.txt")+loadFiletxt("data/normalised/8692-0-normalised.txt")

x_test = loadFiletxt("data/tokenized/14688-0-tokenized.txt")
y_test = loadFiletxt("data/normalised/14688-0-normalised.txt")

start = time.time()
txt_to_train_data(x_train)
print(time.time()-start)










