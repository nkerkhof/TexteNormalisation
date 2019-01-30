
import sys
input_path = "data/test/test_after_norm2.txt"
output_path = sys.argv[1]


def loadFiletxt(filename):
	with open(filename, 'r') as f:
		return f.read()


def delEmptyLines(txt):
	index = 0
	while index < len(txt)-1:
		if txt[index]=='\n' and txt[index+1]=='\n':
			txt = txt[:index+1]+txt[index+2:]
		index = index+1
	if txt[-2]=='\n' and txt[-1]=='\n':
		txt = txt[:-1]
	return txt


txt = loadFiletxt(input_path)
f=open(output_path,'w+')
f.write(delEmptyLines(txt))
f.close()
