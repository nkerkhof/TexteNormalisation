# -*- coding: utf-8 -*-
import numpy as np

def loadFiletxt(filename):
	with open(filename, 'r') as f:
		return f.read()


def alphabet(letter):
	alpha = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéèàùç€%$+*-!?.,:;&\"'()[]/=0123456789"
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


x= loadFiletxt("gutenberg/8692-0.txt")
for i in range(20):
	print alphabet(x[i])











