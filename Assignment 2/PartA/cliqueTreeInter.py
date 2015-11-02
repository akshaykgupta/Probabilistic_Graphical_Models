import csv
import math, operator
import itertools, time
import sys, copy
import argparse

f = open("Results.txt", 'wb')
c = csv.writer(f)

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
			psi.value[i] = self.value[j] + other.value[k]
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

	def __div__(self, other):
		assert(other.scope <= self.scope)
		if len(other.value) == 0:
			return copy.deepcopy(self)
		# psi = Factor()
		# psi.scope = copy.deepcopy(self.scope)
		# psi.card = copy.deepcopy(self.card)
		# psi.stride = copy.deepcopy(self.stride)
		# for i in range(len(self.value)):
		# 	psi.value.append(0)
		# other_scope = sorted(other.scope)
		# this_scope = sorted(self.scope - other.scope)
		# other_range_list = [range(self.card[i]) for i in reversed(other_scope)]
		# other_assignments = list(itertools.product(*other_range_list))
		# this_range_list = [range(self.card[i]) for i in reversed(this_scope)]
		# this_assignments = list(itertools.product(*this_range_list))
		# jumps = []
		# for assn in other_assignments:
		# 	jumps.append(0)
		# 	for var, k in zip(reversed(other_scope), assn):
		# 		jumps[-1] += self.stride[var] * k
		# for i, j in enumerate(this_assignments):
		# 	base_index = 0
		# 	for var, k in zip(reversed(this_scope), j):
		# 		base_index += self.stride[var] * k
		# 	for l, assn in enumerate(other_assignments):
		# 		index = base_index + jumps[l]
		# 		if self.value[index] == 0.0 and other.value[l] == 0.0:
		# 			psi.value[index] = 0.0 
		# 		else:
		# 			psi.value[index] = float(self.value[index]) - float(other.value[l])
		intermediate_factor = copy.deepcopy(other)
		intermediate_factor.value = [-1*value for value in intermediate_factor.value]
		psi = self * intermediate_factor
		return psi

	def index_to_assignment(self, index):
		assignment = {}
		for i in self.scope:
			assignment[i] = (index // self.stride[i]) % self.card[i]
		return assignment

	def assignment_to_index(self, assignment):
		if len(assignment) != len(self.scope):
			raise ValueError
		if isinstance(assignment, dict):
			return sum(self.stride[i]*assignment[i] for i in assignment)
		else:
			return sum(self.stride[j]*assignment[i] for i,j in enumerate(sorted(self.scope)))

	def marginalise(self, V):
		if len(V) == 0:
			return copy.deepcopy(self)
		assert(V <= self.scope)
		exp_values = [math.exp(val) for val in self.value]
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
		ordered_vars = sorted(V)
		range_list = [range(self.card[i]) for i in reversed(ordered_vars)]
		assignments = list(itertools.product(*range_list))
		jumps = []
		for assn in assignments:
			jumps.append(0)
			for var,k in zip(reversed(ordered_vars), assn):
				jumps[-1] += self.stride[var] * k
		new_range_list = [range(self.card[i]) for i in reversed(ordered_scope)]
		new_assignments = list(itertools.product(*new_range_list))
		for i,j in enumerate(new_assignments):
			psi.value.append(0)
			factor_sum = 0
			base_index = 0
			for var,k in zip(reversed(ordered_scope), j):
				base_index += self.stride[var] * k
			for jump in jumps:
				index = base_index + jump
				factor_sum += exp_values[index]
			psi.value[i] = math.log(factor_sum)
		return psi

	def maximise(self, V):
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
		ordered_vars = sorted(V)
		range_list = [range(self.card[i]) for i in reversed(ordered_vars)]
		assignments = list(itertools.product(*range_list))
		jumps = []
		for assn in assignments:
			jumps.append(0)
			for var,k in zip(reversed(ordered_vars), assn):
				jumps[-1] += self.stride[var] * k
		new_range_list = [range(self.card[i]) for i in reversed(ordered_scope)]
		new_assignments = list(itertools.product(*new_range_list))
		for i,j in enumerate(new_assignments):
			psi.value.append(0)
			factor_max = -1000000
			base_index = 0
			for var,k in zip(reversed(ordered_scope), j):
				base_index += self.stride[var] * k
			for jump in jumps:
				index = base_index + jump
				if self.value[index] > factor_max:
					factor_max = self.value[index]
			psi.value[i] = factor_max
		return psi

	def normalise(self):
		unnorm_sum = self.value[0]
		for value in self.value[1:]:
			unnorm_sum = math.log(math.exp(unnorm_sum) + math.exp(value))
		self.value = [value-unnorm_sum for value in self.value]

	def scale_down(self):
		self.value = [value / self.value[0] for value in self.value]
		
class CliqueTreeNode:

	def __init__(self):
		self.assoc_potens = []
		self.scope = set()
		self.nbrs = {}
		self.belief = Factor()

	def __str__(self):
		return "Node( Scope: " + str(self.scope) + ", Potentials: " + str(self.assoc_potens) + ", Neighbours: " + str(self.nbrs) + ", Belief: " + str(self.belief) + " )"

	def __repr__(self):
		return self.__str__()

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
			self.skip_potential = math.log(int(sys.argv[3]))
		else:
			self.skip_potential = math.log(5.0)
		if len(sys.argv) > 4:
			self.pair_skip_potential = math.log(int(sys.argv[3]))
		else:
			self.pair_skip_potential = math.log(5.0)
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
			self.emission_probs[image_number][self.char_map[char]] = math.log(prob)
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
			self.transmission_probs[self.trans_keymap(self.char_map[prev_char], self.char_map[curr_char])] = math.log(prob)
		self.skip_factor = []
		self.pair_skip_factor = []
		for i in range(self.n_chars):
			for j in range(self.n_chars):
				if i == j:
					self.skip_factor.append(self.skip_potential)
					self.pair_skip_factor.append(self.pair_skip_potential)
				else:
					self.skip_factor.append(0.0)
					self.pair_skip_factor.append(0.0)

	def model_score(self, image_vectors, words, model):
		assert(len(image_vectors[0]) == len(words[0]))
		assert(len(image_vectors[1]) == len(words[1]))
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		model_score = 0.0
		for img, char in zip(image_vectors[0], words[0]):
			model_score += self.emission_probs[img][self.char_map[char]]
		for img, char in zip(image_vectors[1], words[1]):
			model_score += self.emission_probs[img][self.char_map[char]]
		if (model == 'ocr'):
			return math.exp(model_score)
		for i in range(1, len(image_vectors[0])):
			model_score += self.transmission_probs[self.trans_keymap(self.char_map[words[0][i-1]], self.char_map[words[0][i]])]
		for i in range(1, len(image_vectors[1])):
			model_score += self.transmission_probs[self.trans_keymap(self.char_map[words[1][i-1]], self.char_map[words[1][i]])]
		if (model == 'trans'):
			return math.exp(model_score)
		for image_vector, word in zip(image_vectors, words):
			for i in range(len(image_vector)):
				for j in range(i+1, len(image_vector)):
					if (image_vector[i] == image_vector[j] and word[i] == word[j]):
						model_score += self.skip_potential
		if (model == 'skip'):
			return math.exp(model_score)
		for i in range(len(image_vectors[0])):
			for j in range(len(image_vectors[1])):
				if (image_vectors[0][i] == image_vectors[1][j] and words[0][i] == words[1][j]):
					model_score += self.pair_skip_potential
		if (model == 'pair_skip'):
			return math.exp(model_score)

	def construct_markov_network(self, image_vectors, model):
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		length1 = len(image_vectors[0])
		length2 = len(image_vectors[1])
		full_length = length1 + length2
		full_vector = [item for sublist in image_vectors for item in sublist]
		markov_network = {i:set() for i in range(full_length)} 
		assoc_potens = [[] for i in range(full_length)]
		for i in range(full_length):
			assoc_potens[i].append((self.emission_probs[full_vector[i]], i))
			if model != 'ocr':
				if (i != length1 - 1) and (i != full_length - 1):
					markov_network[i].add(i+1)
					markov_network[i+1].add(i)
					assoc_potens[i].append((self.transmission_probs, i+1))
					assoc_potens[i+1].append((self.transmission_probs, i))
				if model != 'ocr' and model != 'trans':
					if i < length1:
						for j in range(i+1, length1):
							if full_vector[i] == full_vector[j]:
								markov_network[i].add(j)
								markov_network[j].add(i)
								assoc_potens[i].append((self.skip_factor, j))
								assoc_potens[j].append((self.skip_factor, i))
						if model != 'ocr' and model != 'trans' and model != 'skip':					
							for j in range(length1, full_length):
								if full_vector[i] == full_vector[j]:
									markov_network[i].add(j)
									markov_network[j].add(i)
									assoc_potens[i].append((self.pair_skip_factor, j))
									assoc_potens[j].append((self.pair_skip_factor, i))
					else:
						for j in range(i+1, full_length):
							if full_vector[i] == full_vector[j]:
								markov_network[i].add(j)
								markov_network[j].add(i)
								assoc_potens[i].append((self.skip_factor, i))
								assoc_potens[j].append((self.skip_factor, i))

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
			markov_network[adj_list[j]].remove(next_var)
			for k in range(j+1, len(adj_list)):
				if adj_list[k] not in markov_network[adj_list[j]]:
					markov_network[adj_list[k]].add(adj_list[j])
					markov_network[adj_list[j]].add(adj_list[k])
		del markov_network[next_var]
		return next_var, scope, var_elim_end

	def construct_clique_tree(self, image_vectors, model):
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		markov_network, assoc_potens = self.construct_markov_network(image_vectors, model)
		# print markov_network
		# print assoc_potens
		var_elim_end = False
		clique_tree = []
		out_done = set()
		levels = []
		levels.append(set())
		level = {}
		parent = {}
		children = {}
		def make_factor(i, pot):
			new_factor = Factor()
			if pot[1] != i:
				new_factor.scope.add(i)
				new_factor.scope.add(pot[1])
				new_factor.card[i] = self.n_chars
				new_factor.card[pot[1]] = self.n_chars
				# new_factor.stride[i] = 1
				# new_factor.stride[pot[1]] = self.n_chars				
				low = min([i, pot[1]])
				high = max([i, pot[1]])
				new_factor.stride[low] = 1
				new_factor.stride[high ] = self.n_chars
				new_factor.value = pot[0]
			else:
				new_factor.scope.add(i)
				new_factor.card[i] = self.n_chars
				new_factor.stride[i] = 1
				new_factor.value = pot[0]				
			return new_factor

		while var_elim_end is not True:
			next_var, scope, var_elim_end = self.min_fill(markov_network)
			new_node = CliqueTreeNode()
			# print next_var
			# print scope
			# print markov_network
			new_node.scope = scope
			if var_elim_end is True:
				for i in markov_network:
					for j in assoc_potens[i]:
						if j[1] in new_node.scope and j[1] >= i:
							new_factor = make_factor(i, j)
							new_node.assoc_potens.append(new_factor)
			else:
				for j in assoc_potens[next_var]:
					if j[1] in new_node.scope:
						new_factor = make_factor(next_var, j)
						new_node.assoc_potens.append(new_factor)

			j = len(clique_tree)
			levels[0].add(j)
			level[j] = 0
			children[j] = []
			parent[j] = j
			for i, node in enumerate(clique_tree):
				if i not in out_done and ((next_var in node.scope) or (var_elim_end is True and len(new_node.scope & node.scope) != 0)):
					sep_set = node.scope & new_node.scope
					clique_tree[i].nbrs[j] = sep_set
					new_node.nbrs[i] = sep_set
					out_done.add(i)
					if level[i] + 1 > level[j]:
						levels[level[j]].remove(j)
						level[j] = level[i] + 1
						if level[j] >= len(levels):
							levels.append(set())
						levels[level[j]].add(j)
					parent[i] = j
					children[j].append(i)
			clique_tree.append(new_node)
		# print levels
		# print level
		# print parent
		# print children
		return clique_tree, levels, parent, children

	def two_way_message_passing(self, clique_tree, **kwargs):
		if 'root' in kwargs:
			return
		elif 'levels' not in kwargs or 'parent' not in kwargs or 'children' not in kwargs:
			return
		else:
			levels = kwargs.get('levels')
			children = kwargs.get('children')
			parent = kwargs.get('parent')
		use_max = False
		if 'use_max' in kwargs:
			use_max = kwargs.get('use_max')
		#print clique_tree
		for clique in clique_tree:
			psi = clique.assoc_potens[0]
			for poten in clique.assoc_potens[1:]:
				psi = psi * poten
			clique.assoc_potens = [psi]
		# print clique_tree
		delta = {}
		# print levels
		for level in levels:
			for node in level:
				# print node
				# print clique_tree[node]
				clique = clique_tree[node]
				message = Factor()
				for child in children[node]:
					message = message * delta[(child, node)]
				clique.belief = message * clique.assoc_potens[0]
				if parent[node] != node:
					if use_max:
						delta[(node, parent[node])] = clique.belief.maximise(clique.scope - clique.nbrs[parent[node]])
					else:
						delta[(node, parent[node])] = clique.belief.marginalise(clique.scope - clique.nbrs[parent[node]])
		for level in list(reversed(levels)):
			for node in level:
				clique = clique_tree[node]
				if parent[node] != node:
					clique.belief = clique.belief * delta[(parent[node], node)]
				for child in children[node]:
					# message = clique.belief.marginalise(clique.scope - clique.nbrs[child])
					# delta[(node, child)] = message / delta[(child, node)]
					message = clique.belief / delta[(child, node)]
					if use_max:
						delta[(node, child)] = message.maximise(clique.scope - clique.nbrs[child])
					else:
						delta[(node, child)] = message.marginalise(clique.scope - clique.nbrs[child])					
		#print delta
		return clique_tree

	def predict_mp(self, image_vectors, model, use_max = False):
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		clique_tree, levels, parent, children = self.construct_clique_tree(image_vectors, model)
		clique_tree = crf.two_way_message_passing(clique_tree, levels = levels, parent = parent, children = children, use_max = use_max)
		prediction = {}
		marginals = {}
		for clique in clique_tree:
			for var in clique.scope:
				if var not in prediction:
					if use_max:
						marginal = clique.belief.maximise(clique.scope - set([var]))
					else:
						marginal = clique.belief.marginalise(clique.scope - set([var]))
					idx_max = max(enumerate(marginal.value), key=lambda x:x[1])[0]
					exp_values = [math.exp(val) for val in marginal.value]
					unnorm_sum = sum(exp_values)
					normalized_probs = [val/unnorm_sum for val in exp_values]
					prediction[var] = self.char_rmap[idx_max]
					marginals[var] = normalized_probs
					# print var
					# print normalized_probs
			if len(prediction) == len(image_vectors[0]) + len(image_vectors[1]):
				break
		word1 = ""
		word2 = ""
		for i in range(len(image_vectors[0])):
			word1 += prediction[i]
		for i in range(len(image_vectors[1])):
			word2 += prediction[i + len(image_vectors[0])]
		words = (word1, word2)
		return words, marginals

	def construct_factor_graph(self, image_vectors, model):
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		length1 = len(image_vectors[0])
		length2 = len(image_vectors[1])
		full_length = length1 + length2
		full_vector = [item for sublist in image_vectors for item in sublist]
		factor_graph = []
		separators = {}
		for i in range(full_length):
			factor_graph.append(CliqueTreeNode())
			factor_graph[i].scope.add(i)
			factor_graph[i].assoc_potens = self.emission_probs[full_vector[i]]
			factor_graph[i].belief = Factor()
			factor_graph[i].belief.scope = set([i])
			factor_graph[i].belief.stride = {i:1}
			factor_graph[i].belief.card = {i:self.n_chars}
			factor_graph[i].belief.value = self.emission_probs[full_vector[i]]
		for i in range(full_length):
			if model != 'ocr':
				if (i != length1 - 1) and (i != full_length - 1):
					index = len(factor_graph)
					factor_graph.append(CliqueTreeNode())
					factor_graph[-1].assoc_potens = self.transmission_probs
					factor_graph[-1].belief = Factor()
					factor_graph[-1].belief.scope = set([i,i+1])
					factor_graph[-1].belief.stride = {i:1, i+1:self.n_chars}
					factor_graph[-1].belief.card = {i:self.n_chars, i+1:self.n_chars}
					factor_graph[-1].belief.value = self.transmission_probs
					factor_graph[-1].scope.add(i)
					factor_graph[-1].scope.add(i+1)
					factor_graph[-1].nbrs[i] = i
					factor_graph[-1].nbrs[i+1] = i+1
					factor_graph[i].nbrs[index] = i
					factor_graph[i+1].nbrs[index] = i+1
					separators[(index, i)] = Factor()
					separators[(i, index)] = Factor()
					separators[(index, i+1)] = Factor()
					separators[(i+1, index)] = Factor()
				if model != 'ocr' and model != 'trans':
					if i < length1:
						for j in range(i+1, length1):
							if full_vector[i] == full_vector[j]:
								index = len(factor_graph)
								factor_graph.append(CliqueTreeNode())
								factor_graph[-1].assoc_potens = self.skip_factor
								factor_graph[-1].belief = Factor()
								factor_graph[-1].belief.scope = set([i,j])
								factor_graph[-1].belief.stride = {i:1, j:self.n_chars}
								factor_graph[-1].belief.card = {i:self.n_chars, j:self.n_chars}
								factor_graph[-1].belief.value = self.skip_factor
								factor_graph[-1].scope.add(i)
								factor_graph[-1].scope.add(j)
								factor_graph[-1].nbrs[i] = i
								factor_graph[-1].nbrs[j] = j
								factor_graph[i].nbrs[index] = i
								factor_graph[j].nbrs[index] = j
								separators[(index, i)] = Factor()
								separators[(i, index)] = Factor()
								separators[(index, j)] = Factor()
								separators[(j, index)] = Factor()
						if model != 'ocr' and model != 'trans' and model != 'skip':					
							for j in range(length1, full_length):
								if full_vector[i] == full_vector[j]:
									index = len(factor_graph)
									factor_graph.append(CliqueTreeNode())
									factor_graph[-1].assoc_potens = self.pair_skip_factor
									factor_graph[-1].belief = Factor()
									factor_graph[-1].belief.scope = set([i,j])
									factor_graph[-1].belief.stride = {i:1, j:self.n_chars}
									factor_graph[-1].belief.card = {i:self.n_chars, j:self.n_chars}
									factor_graph[-1].belief.value = self.pair_skip_factor
									factor_graph[-1].scope.add(i)
									factor_graph[-1].scope.add(j)
									factor_graph[-1].nbrs[i] = i
									factor_graph[-1].nbrs[j] = j
									factor_graph[i].nbrs[index] = i
									factor_graph[j].nbrs[index] = j
									separators[(index, i)] = Factor()
									separators[(i, index)] = Factor()
									separators[(index, j)] = Factor()
									separators[(j, index)] = Factor()
					else:
						for j in range(i+1, full_length):
							if full_vector[i] == full_vector[j]:
								index = len(factor_graph)
								factor_graph.append(CliqueTreeNode())
								factor_graph[-1].assoc_potens = self.skip_factor
								factor_graph[-1].belief = Factor()
								factor_graph[-1].belief.scope = set([i,j])
								factor_graph[-1].belief.stride = {i:1, j:self.n_chars}
								factor_graph[-1].belief.card = {i:self.n_chars, j:self.n_chars}
								factor_graph[-1].belief.value = self.skip_factor
								factor_graph[-1].scope.add(i)
								factor_graph[-1].scope.add(j)
								factor_graph[-1].nbrs[i] = i
								factor_graph[-1].nbrs[j] = j
								factor_graph[i].nbrs[index] = i
								factor_graph[j].nbrs[index] = j
								separators[(index, i)] = Factor()
								separators[(i, index)] = Factor()
								separators[(index, j)] = Factor()
								separators[(j, index)] = Factor()
		return factor_graph, separators

	def belief_propagation(self, factor_graph, separators, epsilon = 1.0e-4, use_max = False):
		previous_separators = {}
		previous_beliefs = []
		initial_beliefs = []
		# for i,node in enumerate(factor_graph):
		# 	for j in node.nbrs:
		# 		print (i,j)
		for i, node in enumerate(factor_graph):
			previous_beliefs.append(copy.deepcopy(node.belief))
			initial_beliefs.append(copy.deepcopy(node.belief))
			initial_beliefs[i].normalise()
			previous_beliefs[i].normalise()
			for j in node.nbrs:
				if( (j,i) in separators):
					scope = node.scope & factor_graph[j].scope
					previous_separators[(i,j)] = Factor()
					previous_separators[(i,j)].scope = node.scope & factor_graph[j].scope
					for var in scope:
						previous_separators[(i,j)].card[var] = self.n_chars
						previous_separators[(i,j)].stride[var] = 1
					for n in range(self.n_chars):
						previous_separators[(i,j)].value.append(1.0)
					previous_separators[(i,j)].normalise()
					# print previous_separators[(i,j)]
		converged = False
		begin = time.time()
		time_taken = 0.0
		while(not converged and time_taken < 60.0):
			#print "here"
			converged = True
			for i, node in enumerate(factor_graph):
				# print previous_beliefs[i]
				start = True
				for j in node.nbrs:
					if( (j,i) in separators):
						# print "OLD:\n"
						# print previous_separators[(i,j)]
						message = previous_beliefs[i] / previous_separators[(j,i)]
						if use_max:
							separators[(i,j)] = message.maximise(previous_beliefs[i].scope - previous_separators[(j,i)].scope)
						else:
							separators[(i,j)] = message.marginalise(previous_beliefs[i].scope - previous_separators[(j,i)].scope)
						separators[(i,j)].normalise()
						# print "\n\n\nNEW:\n"
						# print separators[(i,j)]
						for val, prev_val in zip(separators[(i,j)].value, previous_separators[(i,j)].value):
							if abs(val - prev_val) >= epsilon:
								converged = False
			
			for i, node in enumerate(factor_graph):
				# print previous_beliefs[i]
				node.belief = initial_beliefs[i]
				for j in node.nbrs:
							node.belief = node.belief * separators[(j,i)]
							node.belief.normalise()
				for val, prev_val in zip(node.belief.value, previous_beliefs[i].value):
					if abs(val - prev_val) >= epsilon:
						converged = False


			for i, node in enumerate(factor_graph):
				# print "HERE\n"
				# print node.belief
				# print "\n\n\n\n\n\n\n"
				previous_beliefs[i] = copy.deepcopy(node.belief)
				for j in node.nbrs:
					if( (j,i) in separators):
						previous_separators[(i,j)] = copy.deepcopy(separators[(i,j)])
			time_taken = time.time() - begin
		print time_taken
		return factor_graph

	def predict_bp(self, image_vectors, model, use_max = False):
		assert(model == 'ocr' or model == 'trans' or model == 'skip' or model == 'pair_skip')
		factor_graph, separators = self.construct_factor_graph(image_vectors, model)
		factor_graph = self.belief_propagation(factor_graph, separators, use_max = use_max)
		prediction = {}
		marginals = {}		
		for i, node in enumerate(factor_graph):
			for var in node.scope:
				if var not in prediction:
					if use_max:
						marginal = node.belief.maximise(node.scope - set([var]))
					else:
						marginal = node.belief.marginalise(node.scope - set([var]))
					idx_max = max(enumerate(marginal.value), key=lambda x:x[1])[0]
					exp_values = [math.exp(val) for val in marginal.value]
					unnorm_sum = sum(exp_values)
					normalized_probs = [val/unnorm_sum for val in exp_values]
					prediction[var] = self.char_rmap[idx_max]
					marginals[var] = normalized_probs
					# print var
					# print normalized_probs
			if len(prediction) == len(image_vectors[0]) + len(image_vectors[1]):
				break
		word1 = ""
		word2 = ""
		for i in range(len(image_vectors[0])):
			word1 += prediction[i]
		for i in range(len(image_vectors[1])):
			word2 += prediction[i + len(image_vectors[0])]
		words = (word1, word2)

		return words, marginals

	def accuracy(self, technique, dataset, metric, model, use_max = False):
		assert(technique == 'MP' or technique == 'BP')
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
			all_words.extend([word1, word2])
		predicted_words = []
		predicted_marginals = []
		begin = time.time()
		for image_vectors in all_images:
			#print image_vectors
			if(technique == 'MP'):
				prediction, marginals = self.predict_mp(image_vectors, model, use_max)
			elif(technique == 'BP'):
				prediction, marginals = self.predict_bp(image_vectors, model, use_max)
			predicted_words.extend([prediction[0], prediction[1]])
			predicted_marginals.extend([marginals, marginals])
			#print "\n\n\n\n\n\n\n\n\n\n"
			# break
		time_taken = time.time() - begin
		sys.stdout.write('Time taken: ')
		print time_taken
		if (metric == 'character') or (metric == 'all'):
			all_chars = ''.join(all_words)
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
			for word, correct_word in zip(predicted_words, all_words):
				if(word == correct_word):
					correct = correct + 1
			sys.stdout.write('Word accuracy: ')
			print float(correct) / float(len(all_words))
			row.append(float(correct) / float(len(all_words)))
		if (metric == 'likelihood') or (metric == 'all'):
			log_predicted_probs = []
			for i, word in enumerate(all_words):
				log_predicted_probs.append(0.0)
				for j, char in enumerate(word):
					log_predicted_probs[i] += math.log(predicted_marginals[i][j][self.char_map[char]])
			sys.stdout.write('Log likelihood: ')
			print sum(log_predicted_probs) / len(log_predicted_probs)
			row.append(sum(log_predicted_probs) / len(log_predicted_probs))
		row.append(time_taken)
		c.writerow(row)


# a = Factor()
# a.scope = set([1,2,3])
# a.card[1] = 2
# a.card[2] = 3
# a.card[3] = 2
# a.stride[1] = 1
# a.stride[2] = 2
# a.stride[3] = 6
# a.value = [2, 4, 3, 5, 3, 2, 1, 4, 2, 3, 4, 2]
# b = Factor()
# b.scope = set([2,3,4])
# b.card[2] = 3
# b.card[3] = 2
# b.card[4] = 2
# b.stride[2] = 1
# b.stride[3] = 3
# b.stride[4] = 6
# b.value = [1, 2, 3, 3, 2, 4, 2, 3, 5, 3, 2, 1]
# c = Factor()
# c.scope = set([1, 3])
# c.card[1] = 2
# c.card[3] = 2
# c.stride[1] = 1
# c.stride[3] = 2
# c.value = [2, 6, 3, 5]
#print a / c
#print a*b
#print a.marginalise(set({1,2}))
#print a

crf = CRF()
crf.accuracy(technique = 'BP', dataset = 'tree', metric = 'all', model = 'pair_skip', use_max = True)
crf.accuracy(technique = 'BP', dataset = 'treeWS', metric = 'all', model = 'pair_skip', use_max = True)
crf.accuracy(technique = 'BP', dataset = 'loops', metric = 'all', model = 'pair_skip', use_max = True)
crf.accuracy(technique = 'BP', dataset = 'loopsWS', metric = 'all', model = 'pair_skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'tree', metric = 'all', model = 'ocr', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'tree', metric = 'all', model = 'trans', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'tree', metric = 'all', model = 'skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'tree', metric = 'all', model = 'pair_skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'treeWS', metric = 'all', model = 'ocr', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'treeWS', metric = 'all', model = 'trans', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'treeWS', metric = 'all', model = 'skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'treeWS', metric = 'all', model = 'pair_skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loops', metric = 'all', model = 'ocr', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loops', metric = 'all', model = 'trans', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loops', metric = 'all', model = 'skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loops', metric = 'all', model = 'pair_skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loopsWS', metric = 'all', model = 'ocr', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loopsWS', metric = 'all', model = 'trans', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loopsWS', metric = 'all', model = 'skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'loopsWS', metric = 'all', model = 'pair_skip', use_max = True)
# crf.accuracy(technique = 'MP', dataset = 'tree', metric = 'all', model = 'pair_skip')


