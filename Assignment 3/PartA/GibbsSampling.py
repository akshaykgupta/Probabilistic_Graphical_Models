import csv
import math, operator
import itertools
import sys, copy
import random
import time

fr = open("Results1.txt", 'wb')
c = csv.writer(fr)

class Factor:

	'''Class for defining factors and associated operations like factor product, marginalisation, maximisation etc.
	'''

	def __init__(self):
		self.scope = set()
		self.card = {}
		self.stride = {}
		self.value = []

	def __str__(self):
		return "Factor( Scope: " + str(self.scope) + "\nCardinalities: " + str(self.card) + "\nStrides: " + str(self.stride) + "\nValues: " + str(self.value) + " )"

	def __repr__(self):
		return self.__str__()

	def __mul__(self, other):
		if len(other.value) == 0:
			return copy.deepcopy(self)
		if len(self.value) == 0:
			return copy.deepcopy(other)
		psi = Factor()
		psi.scope = self.scope | other.scope
		ordered_scope = sorted(psi.scope)
		for i,l in enumerate(ordered_scope):
			if l in self.card:
				psi.card[l] = self.card[l]
			else:
				psi.card[l] = other.card[l]
			if i == 0:
				psi.stride[l] = 1
			else:
				psi.stride[l] = psi.stride[last] * psi.card[last]
			last = l
		j = k = 0
		assignment = {}
		for l in ordered_scope:
			assignment[l] = 0
		for i in range(reduce(operator.mul, psi.card.values(), 1)):
			psi.value.append(0)
			psi.value[i] = self.value[j] * other.value[k]
			for l in ordered_scope:
				assignment[l] += 1
				if assignment[l] == psi.card[l]:
					assignment[l] = 0
					if l in self.stride:
						j = j - (psi.card[l]-1)*self.stride[l]
					if l in other.stride:
						k = k - (psi.card[l]-1)*other.stride[l]
				else:
					if l in self.stride:
						j = j + self.stride[l]
					if l in other.stride:
						k = k + other.stride[l]
					break
		return psi

	def normalise(self):
		unnorm_sum = sum(self.value)
		self.value = [value / unnorm_sum for value in self.value]

	def reduce(self, values):
		V = set(values.keys())
		if len(V) == 0:
			return copy.deepcopy(self)
		assert(V <= self.scope)
		psi = Factor()
		psi.scope = self.scope - V
		ordered_scope = sorted(psi.scope)
		for i,l in enumerate(ordered_scope):
			psi.card[l] = self.card[l]
			if i == 0:
				psi.stride[l] = 1
			else:
				psi.stride[l] = psi.card[last] * psi.stride[last]
			last = l
		jump = 0
		for var, k in values.iteritems():
			jump += self.stride[var] * k
		new_range_list = [range(self.card[i]) for i in reversed(ordered_scope)]
		new_assignments = list(itertools.product(*new_range_list))
		for i,j in enumerate(new_assignments):
			psi.value.append(0)
			base_index = 0
			for var,k in zip(reversed(ordered_scope), j):
				base_index += self.stride[var] * k
			index = base_index + jump
			psi.value[i] = self.value[index]
		return psi


class CRF:
	
	def __init__(self):
		self.n_chars = 10
		self.n_images = 1000
		self.char_map = {'e':0, 't':1, 'a':2, 'o':3, 'i':4, 'n':5, 's':6, 'h':7, 'r':8, 'd':9}
		self.char_rmap = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r', 'd']
		self.transmission_probs = []
		self.trans_keymap = lambda i,j: j * self.n_chars + i
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
			file = open("OCRdataset/potentials/ocr.dat")
		c = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for i in range(self.n_images):
			self.emission_probs.append([])
			for j in range(self.n_chars):
				self.emission_probs[i].append(0)
		for line in c:
			image_number = int(line[0])
			char = line[1]
			prob = float(line[2])
			self.emission_probs[image_number][self.char_map[char]] = prob
		if len(sys.argv) > 2:
			file = open(sys.argv[2])
		else:
			file = open("OCRdataset/potentials/trans.dat")
		c = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
		for i in range(self.n_chars):
			for j in range(self.n_chars):
				self.transmission_probs.append(0)
		for line in c:
			prev_char = line[0]
			curr_char = line[1]
			prob = float(line[2])
			self.transmission_probs[self.trans_keymap(self.char_map[prev_char], self.char_map[curr_char])] = prob
		self.skip_factor = []
		self.pair_skip_factor = []
		for i in range(self.n_chars):
			for j in range(self.n_chars):
				if i == j:
					self.skip_factor.append(self.skip_potential)
					self.pair_skip_factor.append(self.pair_skip_potential)
				else:
					self.skip_factor.append(1.0)
					self.pair_skip_factor.append(1.0)

	def model_score(self, image_vectors, words, model):
		assert(len(image_vectors[0]) == len(words[0]))
		assert(len(image_vectors[1]) == len(words[1]))
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		model_score = 0.0
		for img, char in zip(image_vectors[0], words[0]):
			model_score += math.log(self.emission_probs[img][self.char_map[char]])
		for img, char in zip(image_vectors[1], words[1]):
			model_score += math.log(self.emission_probs[img][self.char_map[char]])
		if (model == 'ocr'):
			return math.exp(model_score)
		for i in range(1, len(image_vectors[0])):
			model_score += math.log(self.transmission_probs[self.trans_keymap(self.char_map[words[0][i-1]], self.char_map[words[0][i]])])
		for i in range(1, len(image_vectors[1])):
			model_score += math.log(self.transmission_probs[self.trans_keymap(self.char_map[words[1][i-1]], self.char_map[words[1][i]])])
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

	def construct_markov_network(self, image_vectors, model):
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		length1 = len(image_vectors[0])
		length2 = len(image_vectors[1])
		full_length = length1 + length2
		full_vector = [item for sublist in image_vectors for item in sublist]
		markov_network = {i:set() for i in range(full_length)}
		assoc_potens = [Factor() for i in range(full_length)]
		for i in range(full_length):
			new_factor = Factor()
			new_factor.scope.add(i)
			new_factor.stride[i] = 1
			new_factor.card[i] = self.n_chars
			new_factor.value = copy.deepcopy(self.emission_probs[full_vector[i]])
			assoc_potens[i] = assoc_potens[i] * new_factor
			if model != 'ocr':
				if (i != length1 - 1) and (i != full_length - 1):
					markov_network[i].add(i+1)
					markov_network[i+1].add(i)
					new_factor = Factor()
					new_factor.scope.add(i)
					new_factor.scope.add(i+1)
					new_factor.stride[i] = 1
					new_factor.stride[i+1] = self.n_chars
					new_factor.card[i] = self.n_chars
					new_factor.card[i+1] = self.n_chars
					new_factor.value = copy.deepcopy(self.transmission_probs)
					assoc_potens[i] = assoc_potens[i] * new_factor
					assoc_potens[i+1] = assoc_potens[i+1] * new_factor
				if model != 'ocr' and model != 'trans':
					if i < length1:
						for j in range(i+1, length1):
							if full_vector[i] == full_vector[j]:
								markov_network[i].add(j)
								markov_network[j].add(i)
								new_factor = Factor()
								new_factor.scope.add(i)
								new_factor.scope.add(j)
								new_factor.stride[i] = 1
								new_factor.stride[j] = self.n_chars
								new_factor.card[i] = self.n_chars
								new_factor.card[j] = self.n_chars
								new_factor.value = copy.deepcopy(self.skip_factor)
								assoc_potens[i] = assoc_potens[i] * new_factor
								assoc_potens[j] = assoc_potens[j] * new_factor
						if model != 'ocr' and model != 'trans' and model != 'skip':					
							for j in range(length1, full_length):
								if full_vector[i] == full_vector[j]:
									markov_network[i].add(j)
									markov_network[j].add(i)
									new_factor = Factor()
									new_factor.scope.add(i)
									new_factor.scope.add(j)
									new_factor.stride[i] = 1
									new_factor.stride[j] = self.n_chars
									new_factor.card[i] = self.n_chars
									new_factor.card[j] = self.n_chars
									new_factor.value = copy.deepcopy(self.pair_skip_factor)
									assoc_potens[i] = assoc_potens[i] * new_factor
									assoc_potens[j] = assoc_potens[j] * new_factor
					else:
						for j in range(i+1, full_length):
							if full_vector[i] == full_vector[j]:
								markov_network[i].add(j)
								markov_network[j].add(i)
								new_factor = Factor()
								new_factor.scope.add(i)
								new_factor.scope.add(j)
								new_factor.stride[i] = 1
								new_factor.stride[j] = self.n_chars
								new_factor.card[i] = self.n_chars
								new_factor.card[j] = self.n_chars
								new_factor.value = copy.deepcopy(self.skip_factor)
								assoc_potens[i] = assoc_potens[i] * new_factor
								assoc_potens[j] = assoc_potens[j] * new_factor

		return markov_network, assoc_potens

	def gibbs_predictor(self, image_vectors, words, model, technique, burn_in = 1000, n_iterations = 20000):
		assert(technique == 'ordered' or technique == 'uniform')		
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		length1 = len(image_vectors[0])
		length2 = len(image_vectors[1])
		full_length = length1 + length2
		markov_network, assoc_potens = self.construct_markov_network(image_vectors, model)
		full_assignment = [random.randint(0, self.n_chars-1) for i in range(full_length)]
		iteration = 0
		samples = [[0 for j in range(self.n_chars)] for i in range(full_length)]
		# print image_vectors
		# print markov_network
		# print assoc_potens
		# print samples
		# print full_assignment
		avg_ll = 0.0
		converged = False
		while (iteration < burn_in + n_iterations) and not converged:
			for i in range(full_length):
				if technique == 'ordered':
					next_var = i
				elif technique == 'uniform':
					next_var = random.randint(0, full_length-1)
				values = {}
				for nbr in markov_network[next_var]:
					values[nbr] = full_assignment[nbr]
				reduced_factor = assoc_potens[next_var].reduce(values)
				# print reduced_factor
				reduced_factor.normalise()
				# print reduced_factor
				toss = random.random()
				prev_val = 0.0
				for j, val in enumerate(reduced_factor.value):
					if toss < prev_val + val:
						sample = j
						break
					prev_val = prev_val + val
				full_assignment[next_var] = sample
				if iteration > burn_in:
					for j, val in enumerate(full_assignment):
						samples[j][val] += 1
				# print toss
				# print full_assignment
			# print samples
			# if iteration > burn_in:
			# 	for i, val in enumerate(full_assignment):
			# 		samples[i][val] += 1
			if iteration > burn_in and iteration % 100 == 0:
				current_ll = 0.0
				for i in range(full_length):
					max_count = max(samples[i])
					likelihood = float(max_count) / float(sum(samples[i]))
					current_ll += math.log(likelihood)
				# print current_ll
				if abs(current_ll - avg_ll) < 0.001:
					converged = True
				avg_ll = current_ll
			iteration += 1
		prediction = ''
		if converged == True:
			print "Converged"
		else:
			print "Not Converged"
		log_likelihood = 0.0
		for i in range(full_length):
			idx_max = max(enumerate(samples[i]), key=lambda x:x[1])[0]
			prediction += self.char_rmap[idx_max]
			true_idx = self.char_map[words[i]]
			likelihood = float(samples[i][true_idx]) / float(sum(samples[i]))
			log_likelihood += math.log(likelihood)
		# print prediction
		# print log_likelihood
		# exit()
		return prediction, log_likelihood

	def accuracy(self, dataset, metric, model, technique, burn_in = 1000, n_iterations = 20000):
		assert(technique == 'ordered' or technique == 'uniform')
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		assert(dataset == 'loopsWS' or dataset == 'loops' or dataset == 'treeWS' or dataset == 'tree')
		assert(metric == 'character' or metric == 'word' or metric == 'likelihood' or metric == 'all')
		row = []
		if (dataset == 'tree'):
			image_file = open('OCRdataset/data/data-tree.dat')
			word_file = open('OCRdataset/data/truth-tree.dat')
		elif (dataset == 'treeWS'):
			image_file = open('OCRdataset/data/data-treeWS.dat')
			word_file = open('OCRdataset/data/truth-treeWS.dat')
		elif (dataset == 'loops'):
			image_file = open('OCRdataset/data/data-loops.dat')
			word_file = open('OCRdataset/data/truth-loops.dat')
		elif (dataset == 'loopsWS'):
			image_file = open('OCRdataset/data/data-loopsWS.dat')
			word_file = open('OCRdataset/data/truth-loopsWS.dat')
		f = list(image_file.readlines())
		all_images = []
		for i in range(0, len(f), 3):
			image1 = [int(a) for a in f[i].split()]
			image2 = [int(a) for a in f[i+1].split()]
			all_images.append((image1, image2))
		f = list(word_file.readlines())
		all_words = []
		for i in range(0, len(f), 3):
			word1 = f[i].rstrip('\n')
			word2 = f[i+1].rstrip('\n')
			all_words.append((word1, word2))
		predicted_words = []
		begin = time.time()
		log_likelihoods = []
		for image_vectors, words in zip(all_images, all_words):
			#print image_vectors
			word = words[0] + words[1]
			prediction, log_likelihood = self.gibbs_predictor(image_vectors, word, model, technique, burn_in, n_iterations)
			predicted_words.append(prediction)
			log_likelihoods.append(log_likelihood)
			#print "\n\n\n\n\n\n\n\n\n\n"
			# break
		time_taken = time.time() - begin
		sys.stdout.write('Time taken: ')
		print time_taken
		if (metric == 'character') or (metric == 'all'):
			all_words_list = [words[0] + words[1] for words in all_words]
			all_chars = ''.join(all_words_list)
			predicted_chars = ''.join(predicted_words)
			correct = 0
			for char, correct_char in zip(predicted_chars, all_chars):
				if(char == correct_char):
					correct = correct + 1
			sys.stdout.write('Character accuracy: ')
			print float(correct) / float(len(all_chars))
			row.append(float(correct) / float(len(all_chars)))
		if (metric == 'word') or (metric == 'all'):
			correct = 0
			for word, correct_words in zip(predicted_words, all_words):
				length1 = len(correct_words[0])
				length2 = len(correct_words[1])
				full_length = length1 + length2
				if(word[:length1] == correct_words[0]):
					correct = correct + 1
				if(word[length1:full_length] == correct_words[1]):
					correct = correct + 1
			sys.stdout.write('Word accuracy: ')
			print float(correct) / float(len(all_words) * 2)
			row.append(float(correct) / float(len(all_words) * 2))
		if (metric == 'likelihood') or (metric == 'all'):
			log_predicted_probs = []
			print (sum(log_likelihoods) / len(log_likelihoods))
			row.append(sum(log_likelihoods) / len(log_likelihoods))
		row.append(time_taken)
		c.writerow(row)

crf = CRF()
for iterations in range(10000, 15000, 500):
	# crf.accuracy(dataset = 'tree', metric = 'all', model = 'pair_skip', technique = 'ordered')
	# crf.accuracy(dataset = 'treeWS', metric = 'all', model = 'pair_skip', technique = 'ordered')
	# crf.accuracy(dataset = 'loops', metric = 'all', model = 'pair_skip', technique = 'ordered')
	crf.accuracy(dataset = 'loopsWS', metric = 'all', model = 'pair_skip', technique = 'ordered', n_iterations = iterations)

# crf.accuracy(dataset = 'tree', metric = 'all', model = 'pair_skip', technique = 'ordered')
# crf.accuracy(dataset = 'treeWS', metric = 'all', model = 'pair_skip', technique = 'ordered')
# crf.accuracy(dataset = 'loops', metric = 'all', model = 'pair_skip', technique = 'ordered')
# crf.accuracy(dataset = 'loopsWS', metric = 'all', model = 'pair_skip', technique = 'ordered')

