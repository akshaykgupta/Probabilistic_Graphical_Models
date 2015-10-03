import csv
import math
import itertools
import sys
import argparse

class CliqueTreeNode:

	def __init__(self):
		self.assoc_potens = []
		self.scope = set()

class CRF:
	
	def __init__(self):
		self.n_chars = 10
		self.n_images = 1000
		self.char_map = {'e':0, 't':1, 'a':2, 'o':3, 'i':4, 'n':5, 's':6, 'h':7, 'r':8, 'd':9}
		self.transmission_probs = []
		self.emission_probs = []
		self.skip_potential = 0.0
		if len(sys.argv) > 3:
			self.skip_potential = int(sys.argv[3])
		else:
			self.skip_potential = 5.0
		if len(sys.argv) > 4:
			self.pair_skip_potential = int(sys.argv[3])
		else:
			self.pair_skip_potential = 5.0
		if len(sys.argv) > 1:
			file = open(sys.argv[1])
		else:
			file = open("OCRdataset-2/potentials/ocr.dat")
		c = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for i in range(self.n_images):
			self.emission_probs.append([])
			for j in range(self.n_chars):
				self.emission_probs[i].append(0)
		for line in c:
			image_number = int(line[0])
			char = line[1]
			prob = float(line[2])
			self.emission_probs[image_number][self.char_map[char]] = math.log(prob)
		if len(sys.argv) > 2:
			file = open(sys.argv[2])
		else:
			file = open("OCRdataset-2/potentials/trans.dat")
		c = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for i in range(self.n_chars):
			self.transmission_probs.append([])
			for j in range(self.n_chars):
				self.transmission_probs[i].append(0)
		for line in c:
			prev_char = line[0]
			curr_char = line[1]
			prob = float(line[2])
			self.transmission_probs[self.char_map[prev_char]][self.char_map[curr_char]] = math.log(prob)
		self.skip_factor = []
		for i in range(self.n_chars):
			self.skip_factor.append([])
			for j in range(self.n_chars):
				if i == j:
					self.skip_factor[i].append(5.0)
				else:
					self.skip_factor[i].append(1.0)

	def model_score(self, image_vectors, words, model):
		assert(len(image_vector) == len(word))
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		model_score = 0.0
		for img, char in zip(image_vectors[0], words[0]):
			model_score += self.emission_probs[img][self.char_map[char]]
		for img, char in zip(image_vector[1], words[1]):
			model_score += self.emission_probs[img][self.char_map[char]]
		if (model == 'ocr'):
			return math.exp(model_score)
		for i in range(1, len(image_vectors[0])):
			model_score += self.transmission_probs[self.char_map[words[0][i-1]]][self.char_map[words[0][i]]]
		for i in range(1, len(image_vectors[1])):
			model_score += self.transmission_probs[self.char_map[words[1][i-1]]][self.char_map[words[1][i]]]
		if (model == 'trans'):
			return math.exp(model_score)
		for image_vector, word in zip(image_vectors, words):
			for i in range(len(image_vector)):
				for j in range(i+1, len(image_vector)):
					if (image_vector[i] == image_vector[j] and word[i] == word[j]):
						model_score += math.log(self.skip_potential)
		if (model == 'skip'):
			return math.exp(model_score)
		for i in range(len(image_vectors[0])):
			for j in range(len(image_vectors[1])):
				if (image_vectors[0][i] == image_vectors[1][j] and words[0][i] == words[1][j]):
					model_score += math.log(self.pair_skip_potential)
		if (model == 'pair_skip'):
			return math.exp(model_score)

	def construct_graph(self, image_vectors, model):
		self.length1 = len(image_vectors[0])
		self.length2 = len(image_vectors[1])
		self.full_length = self.length1 + self.length2
		self.full_vector = [item for sublist in image_vectors for item in sublist]
		markov_network = {i:set() for i in range(full_length)} 
		assoc_potens = [[] for i in range(self.full_length)]
		for i in range(self.full_length):
			assoc_potens[i].append(self.emission_probs[self.full_vector[i]])
			if model != 'ocr':
				if (i != self.length1 - 1) and (i != self.full_length - 1):
					markov_network[i].add(i+1)
					markov_network[i+1].add(i)
					assoc_potens[i].append(self.transmission_probs)
					assoc_potens[i+1].append(self.transmission_probs)
				if model != 'ocr' and model != 'trans':
					if i < self.length1:
						for j in range(i+1, self.length1):
							if self.full_vector[i] == self.full_vector[j]:
								markov_network[i].add(j)
								markov_network[j].add(i)
								assoc_potens[i].append(self.skip_factor)
								assoc_potens[j].append(self.skip_factor)
						if model != 'ocr' and model != 'trans' and model != 'skip':					
							for j in range(self.length1, self.full_length):
								if self.full_vector[i] == self.full_vector[j]:
									markov_network[i].add(j)
									markov_network[j].add(i)
									assoc_potens[i].append(self.skip_factor)
									assoc_potens[j].append(self.skip_factor)
					else:
						for j in range(i+1, self.full_length):
							if self.full_vector[i] == self.full_vector[j]:
								markov_network[i].add(j)
								markov_network[j].add(i)
								assoc_potens[i].append(self.skip_factor)
								assoc_potens[j].append(self.skip_factor)

		return markov_network, assoc_potens

	def min_fill(self, markov_network):
		var_elim_end = True
		min_fill_edges = len(markov_network) + 1
		next_var = 0
		for i in markov_network:
			fill_edges = 0
			if len(markov_network[i]) != len(markov_network) - 1:
				var_elim_end = False
			adj_list = list(markov_network[i])
			for j in range(len(adj_list)):
				for k in range(j+1, len(adj_list)):
					if adj_list[k] not in markov_network[adj_list[j]]:
						fill_edges += 1
			if fill_edges < min_fill_edges:
				min_fill_edges = fill_edges
				next_var = i
		if var_elim_end is True:
			scope = set(markov_network.keys())
			return next_var, scope, var_elim_end
		adj_list = list(markov_network[next_var])
		scope = set()
		scope.add(next_var)
		for j in range(len(adj_list)):
			scope.add(adj_list[j])
			markov_network[adj_list[j]].pop(next_var)
			for k in range(j+1, len(adj_list)):
				if adj_list[k] not in markov_network[adj_list[j]]:
					markov_network[adj_list[k]].add(adj_list[j])
					markov_network[adj_list[j]].add(adj_list[k])
		markov_network.pop(next_var)
		return next_var, scope, var_elim_end

	def construct_clique_tree(self, image_vectors, model):
		markov_network, assoc_potens = self.construct_graph(self, image_vectors, model)
		var_elim_end = False
		clique_tree = []
		clique_tree_edges = []
		out_done = set()
		while var_elim_end is not True:
			next_var, scope, var_elim_end = self.min_fill(markov_network)
			new_node = CliqueTreeNode()
			new_node.scope = scope
			if var_elim_end is True:
				for i in markov_network:
					new_node.assoc_potens.extend(assoc_potens[i])
			else:
				new_node.assoc_potens = assoc_potens[next_var]
			clique_tree_edges.append({})
			j = 
			for i, node in enumerate(clique_tree):
				if i not in out_done and next_var in node.scope():
					clique_tree_edges[i].
			clique_tree.append(new_node)

	def two_way_message_passing(self, clique_tree):


	# def predict(self, image_vector, model, *args, **kwargs):
 #        #TODO: Message passing and Loopy Belief Propagation
	# 	assert(model == 'ocr' or model == 'trans' or model == 'skip')
	# 	# print image_vector
	# 	all_words = itertools.product(self.char_map.keys(), repeat = len(image_vector))
	# 	all_words = [''.join(word) for word in all_words]
	# 	predicted_word = ''
	# 	highest_score = 0.0
	# 	Z = 0.0
	# 	true_word_score = 0.0
	# 	true_word = kwargs.get('true_word', None)
	# 	for word in all_words:
	# 		curr_score = self.model_score(image_vector, word, model)
	# 		if true_word == word:
	# 			true_word_score = curr_score
	# 		if (curr_score > highest_score):
	# 			highest_score = curr_score
	# 			predicted_word = word
	# 		Z += curr_score
	# 	predicted_prob = highest_score / Z
	# 	true_word_prob = None
	# 	if true_word is not None:
	# 		true_word_prob = true_word_score / Z
	# 	print predicted_word
	# 	print true_word
	# 	return {'predicted_word' : predicted_word, 'predicted_prob' : predicted_prob, 'true_word_prob' : true_word_prob}

	# def accuracy(self, dataset, metric, model):
	# 	assert(model == 'ocr' or model == 'trans' or model == 'skip')
	# 	assert(dataset == 'small' or dataset == 'large')
	# 	assert(metric == 'character' or metric == 'word' or metric == 'likelihood' or metric == 'all')
	# 	if (dataset == 'small'):
	# 		image_file = open('OCRdataset/data/small/images.dat')
	# 		word_file = open('OCRdataset/data/small/words.dat')
	# 	elif (dataset == 'large'):
	# 		image_file = open('OCRdataset/data/large/allimages1.dat')
	# 		word_file = open('OCRdataset/data/large/allwords.dat')
	# 	all_images = [[int(a) for a in line.split()] for line in image_file.readlines()]
	# 	all_words = [word.rstrip('\n') for word in word_file.readlines()]
	# 	predicted_words = []
	# 	predicted_probs = []
	# 	true_word_probs = []
	# 	for image_vector, true_word in zip(all_images, all_words):
	# 		prediction = self.predict(image_vector, model, true_word = true_word)
	# 		predicted_words.append(prediction['predicted_word'])
	# 		predicted_probs.append(prediction['predicted_prob'])
	# 		true_word_probs.append(prediction['true_word_prob'])
	# 	if (metric == 'character') or (metric == 'all'):
	# 		all_chars = ''.join(all_words)
	# 		predicted_chars = ''.join(predicted_words)
	# 		correct = 0
	# 		for char, correct_char in zip(predicted_chars, all_chars):
	# 			if(char == correct_char):
	# 				correct = correct + 1
	# 		sys.stdout.write('Character accuracy: ')
	# 		print float(correct) / float(len(all_chars))
	# 	if (metric == 'word') or (metric == 'all'):
	# 		correct = 0
	# 		for word, correct_word in zip(predicted_words, all_words):
	# 			if(word == correct_word):
	# 				correct = correct + 1
	# 		sys.stdout.write('Word accuracy: ')
	# 		print float(correct) / float(len(all_words))
	# 	if (metric == 'likelihood') or (metric == 'all'):
	# 		log_predicted_probs = [math.log(prob) for prob in true_word_probs]
	# 		sys.stdout.write('Log likelihood: ')
	# 		print sum(log_predicted_probs) / len(log_predicted_probs)