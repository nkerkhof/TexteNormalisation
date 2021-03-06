input_path = "data/rawdata/gutenberg/8692-0.txt"
output_path = "data/test/8692-0-proc.txt"


process_path = "data/test/test_proc.txt"



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


delStartAndEnd(input_path, process_path)
txt = loadFiletxt(process_path)
txt = sentencePerLine(txt)
txt = delEmptyLines(txt)

f=open(output_path,'w+')
f.write(txt)
f.close()

