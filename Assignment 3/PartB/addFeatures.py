import sys, os
import csv
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Add features for POS tagger')
	parser.add_argument('--use-cap', dest = 'capital', default = False, nargs = 1)
	parser.add_argument('--test', dest = 'testOption', default = False, nargs = 1)
	parser.add_argument('--file', dest = 'file', default = '', nargs = 1)
	parser.add_argument('--add-ing', dest = 'end_ing', default = False, nargs = 1)
	parser.add_argument('--add-at', dest = 'start_at', default = False, nargs = 1)
	parser.add_argument('--add-hash', dest = 'start_hash', default = False, nargs = 1)
	parser.add_argument('--add-ly', dest = 'end_ly', default = False, nargs = 1)
	parser.add_argument('--add-nom-ver', dest = 'end_nom_ver', default = False, nargs = 1)
	parser.add_argument('--use-url', dest = 'url', default = False, nargs = 1)
	parser.add_argument('--add-poss', dest = 'poss', default = False, nargs = 1)
	parser.add_argument('--new-file', dest = 'new_file', default = '', nargs = 1)
	args = parser.parse_args()
	if args.file != '':
		args.file = args.file[0]
		fi = open(args.file)
		csvi = fi.readlines()
	else:
		print "File name required"
		exit()
	written_rows = []
	for i, line in enumerate(csvi):
		written_rows.append([])
		line = line.strip().split()
		if len(line) == 0:
			continue
		for word in line[:len(line)-1]:
			written_rows[i].append(word)
		if args.capital and word[0].isupper():
			written_rows[i].append('CAP')
		if args.end_ing and word.endswith('ing'):
			written_rows[i].append('ING')
		if args.start_at and word.startswith('@') and len(word) > 1:
			written_rows[i].append('AT')
		if args.start_hash and word.startswith('#') and len(word) > 1:
			written_rows[i].append('HASH')
		if args.end_ly and word.endswith('ly'):
			written_rows[i].append('LY')
		if args.end_nom_ver and (word.endswith('\'re') or word.endswith('\'m')):
			written_rows[i].append('NOMVER')
		if args.url and word.startswith('http'):
			written_rows[i].append('URL')
		if args.poss and word.endswith('\'s'):
			written_rows[i].append('POSS')
		if args.testOption and len(line) > 1:
			written_rows[i].append(line[-1])
	if args.new_file == '':
		new_file = 'prepared_'
		new_file += args.file
	else:
		new_file = args.new_file[0]
	fo = open(new_file, 'wb')
	for line in written_rows:
		for i, word in enumerate(line):
			fo.write(word)
			if i != len(line) - 1:
				fo.write(' ')
		fo.write('\n')