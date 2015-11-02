import csv
import math, operator
import itertools
import sys, copy
import argparse

def learn_probs(training_seq, training_islands):
	trans_probs = []
	for i in range(8):
		trans_probs.append([])
		for j in range(8):
			trans_probs[i].append(0.0)
	lines = [line.rstrip('\n') for line in list(training_seq.readlines())]
	all_chars = ''.join(lines)
	ranges = [line.split() for line in list(training_islands.readlines())]
	counts = [0.0 for i in range(8)]
	last_in_island = 1
	for arange in ranges:
		if int(arange[0]) <= 1 and int(arange[1]) >= 1:
			last_in_island = 0
			break
	for i in range(1, len(all_chars)):
		in_island = 1
		for arange in ranges:
			if int(arange[0]) <= i+1 and int(arange[1]) >= i+1:
				in_island = 0
				break
		trans_probs[mapping[all_chars[i-1]]*2+last_in_island][mapping[all_chars[i]]*2+in_island] += 1.0
		counts[mapping[all_chars[i-1]]*2+last_in_island] += 1.0
		last_in_island = in_island
	for i in range(8):
		for j in range(8):
			trans_probs[i][j] /= counts[i]
	for i in range(8):
		for j in range(8):
			if trans_probs[i][j] == 0.0:
				trans_probs[i][j] = 1.0e-20
			trans_probs[i][j] = math.log(trans_probs[i][j])
	for arange in ranges:
		start = int(arange[0])-1
		end = int(arange[1])
		all_chars = all_chars[:start] + all_chars[end:]
	initial_prob = []
	for i in range(8):
		initial_prob.append(0.0)
	for char in all_chars:
		initial_prob[mapping[char]*2 + 1] += 1.0
	for i in range(8):
		initial_prob[i] /= len(all_chars)
		if initial_prob[i] == 0.0:
			initial_prob[i] = 1.0e-20
		initial_prob[i] = math.log(initial_prob[i])
	return initial_prob, trans_probs

def mpe_inference(test_seq, trans_probs, emission_probs, initial_prob):
	value = []
	path = []
	lines = [line.rstrip('\n') for line in list(test_seq.readlines())]
	all_chars = ''.join(lines) 
	for i in range(len(all_chars)):
		path.append([])
		value.append([])
		for j in range(8):
			path[i].append(j)
			value[i].append(0)
	for j in range(8):
		value[0][j] = initial_prob[j] + emission_probs[j][mapping[all_chars[0]]]
	for i in range(1, len(all_chars)):
		for j in range(8):
			maxval = -10000000
			maxind = 0
			for k in range(8):
				val = value[i-1][k] + trans_probs[k][j]
				if(val > maxval):
					maxval = val
					maxind = k
			value[i][j] = maxval + emission_probs[j][mapping[all_chars[i]]]
			path[i][j] = maxind
	maxval = -10000000
	maxind = 0
	for k in range(8):
		if(value[len(all_chars)-1][k] > maxval):
			maxval = value[len(all_chars)-1][k]
			maxind = k
	prediction = []
	last = maxind
	for i in range(len(all_chars)):
		prediction.append(last)
		last = path[len(all_chars)-i-1][last]
	return list(reversed(prediction)), maxval

if __name__ == '__main__':
	if len(sys.argv) > 2:
		training_seq = open(sys.argv[1])
		training_islands = open(sys.argv[2])
	else:
		training_seq = open('gene_data/training.txt')
		training_islands = open('gene_data/cpg_island_training.txt')
	if len(sys.argv) > 4:
		test_seq = open(sys.argv[3])		
	else:
		test_seq = open('gene_data/testing.txt')
	mapping = {'A':0, 'G':1, 'T':2, 'C':3}
	emission_probs = []
	for i in range(8):
		emission_probs.append([])
		for j in range(4):
			emission_probs[i].append(1.0e-20)
			if i/2 == j:
				emission_probs[i][j] = 1.0
			emission_probs[i][j] = math.log(emission_probs[i][j])
	initial_prob, trans_probs = learn_probs(training_seq, training_islands)
	# print initial_prob
	# print trans_probs
	# print emission_probs	
	prediction, predicted_proba = mpe_inference(test_seq, trans_probs, emission_probs, initial_prob)
	print "CpG islands:"
	in_island = False
	for ind, i in enumerate(prediction):
		if i % 2 == 0:
			if in_island is False:
				sys.stdout.write(str(ind+1))
				sys.stdout.write("\t")
				in_island = True
		else:
			if in_island is True:
				sys.stdout.write(str(ind))
				sys.stdout.write("\n")
				in_island = False
	print "Log probability: " + str(predicted_proba)
