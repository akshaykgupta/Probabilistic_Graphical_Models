import sys
import csv
import os

def square(x):
	return x * x

def identity(x):
	return x

def double(x):
	return x * 2

def triple(x):
	return x * 3

def quadruple(x):
	return x * 4

def pentuple(x):
	return x * 5

def runModel(ocrfile = "OCRdataset/potentials/ocr.dat", transfile = "OCRdataset/potentials/trans.dat", skip = 5):
	file = "crf.py "
	file += ocrfile
	file += " "
	file += transfile
	file += " "
	if skip != 5:
		file += str(skip)
		file += " "
	f = open("Results.txt", 'a')
	f.write(file)
	f.write('\n')
	f.close()
	os.system("python " + file + ">> Results.txt")

def changeParams(ocrdata, transdata, modification1, modification2):
	newOcrFile = "OCRdataset/potentials/ocr_"
	newOcrFile += modification1.__name__
	newOcrFile += ".dat"
	newTransFile = "OCRdataset/potentials/trans_"
	newTransFile += modification2.__name__
	newTransFile += ".dat"
	nof = open(newOcrFile, 'wb')
	ntf = open(newTransFile, 'wb')
	tsv1 = csv.writer(nof, delimiter = '\t')
	tsv2 = csv.writer(ntf, delimiter = '\t')
	for line in ocrdata:
		newline = line[:]
		newline[2] = str(modification1(float(newline[2])))

		tsv1.writerow(newline)
	for line in transdata:
		newline = line[:]
		newline[2] = str(modification2(float(newline[2])))
		tsv2.writerow(newline)
	nof.close()
	ntf.close()
	return newOcrFile, newTransFile

of = open("OCRdataset/potentials/ocr.dat")
tf = open("OCRdataset/potentials/trans.dat")
csv1 = csv.reader(of, delimiter = '\t', quoting = csv.QUOTE_NONE)
csv2 = csv.reader(tf, delimiter = '\t', quoting = csv.QUOTE_NONE)
ocrdata = [line for line in csv1]
transdata = [line for line in csv2]
arg1, arg2 = changeParams(ocrdata, transdata, identity, square)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, square, identity)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, identity, double)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, double, identity)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, identity, triple)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, triple, identity)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, identity, quadruple)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, quadruple, identity)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, identity, pentuple)
runModel(arg1, arg2)
arg1, arg2 = changeParams(ocrdata, transdata, pentuple, identity)
runModel(arg1, arg2)

for i in range(1, 5):
	runModel(skip = i)