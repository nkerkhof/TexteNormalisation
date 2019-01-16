# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:12:17 2019

@author: Torblad
"""

f = open("data\\rawdata\\gutenberg\\8692-0.txt","r", encoding="utf8")
lines = f.readlines()
f.close()
f = open("test.txt","w", encoding="utf8")

i=0

while lines[i].startswith("*** START OF") != True :
    i = i+1

i = i+1

while lines[i].startswith("*** END OF") != True :
    f.write(lines[i])
    i = i+1
    
f.close()