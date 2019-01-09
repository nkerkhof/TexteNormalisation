# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(threshold=np.inf)

import time

from keras.models import Sequential
from keras.layers import Dense



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


txt = loadFiletxt("gutenberg/pg31117.txt")
txt_size = len(txt)

#print(txt[0:20])
#print(txt_to_train_data(txt[0:20]))
start = time.time()
txt_to_train_data(txt)
print(time.time()-start)
print(txt_size)

'''
x_train = ...avantNorm...
y_train = ...apresNorm...

x_test = .avantNorm.
y_test = .apresNorm.
'''












