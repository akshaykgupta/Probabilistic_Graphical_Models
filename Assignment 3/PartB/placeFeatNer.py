import sys, os
import csv
import argparse

if __name__ == '__main__':
	file_name = sys.argv[1]
	new_file = sys.argv[2]
	f = open(file_name)
	p = open('places.txt')
	m = p.readlines()
	dict = {}
	for line in m:
		dict[line.strip()] = True
	p.close()
	csvi = f.readlines()
	prev = False
	written_rows = []
	for i, line in enumerate(csvi):
		written_rows.append([])
		line = line.strip().split()
		if len(line) == 0:
			continue
		if len(sys.argv) > 3 and len(line) > 1:	
			for word in line[:len(line)-1]:
				written_rows[i].append(word)
		else:
			for word in line:
				written_rows[i].append(word)
		w = line[0].lower()
		if w in dict:
			if line[0][0].isupper() or len(line[0][0]) > 4:
				written_rows[i].append('LOC')
		elif i < len(csvi) - 1 :
			nline = csvi[i+1].strip().split()
			if len(nline) > 0:
				nw = nline[0].lower()
				w += ' '
				w += nw
				if w in dict:
					written_rows[i].append('LOC')
				prev = True
		elif prev:
			prev = False
			written_rows[i].append('LOC')
		if len(sys.argv) > 3 and len(line) > 1:
			written_rows[i].append(line[-1])
	fo = open(new_file, 'wb')
	for line in written_rows:
		for i, word in enumerate(line):
			fo.write(word)
			if i != len(line) - 1:
				fo.write(' ')
		fo.write('\n')