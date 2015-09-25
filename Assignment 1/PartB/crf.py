import csv
import math
import itertools
import sys
import argparse

class CRF:
	
	def __init__(self):
		self.n_chars = 10
		self.n_images = 1000
		self.char_map = {'e':0, 't':1, 'a':2, 'o':3, 'i':4, 'n':5, 's':6, 'h':7, 'r':8, 'd':9}
		self.transmission_probs = []
		self.emission_probs = []
		self.skip_potential = 0.0
		if len(sys.argv) > 3:
			self.skip_potential = math.log(int(sys.argv[3]))
		else:
			self.skip_potential = math.log(5.0)
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
			self.transmission_probs.append([])
			for j in range(self.n_chars):
				self.transmission_probs[i].append(0)
		for line in c:
			prev_char = line[0]
			curr_char = line[1]
			prob = float(line[2])
			self.transmission_probs[self.char_map[prev_char]][self.char_map[curr_char]] = math.log(prob)

	def model_score(self, image_vector, word, model):
		assert(len(image_vector) == len(word))
		assert(model == 'ocr' or model == 'trans' or model == 'skip')
		model_score = 0.0
		for img, char in zip(image_vector, word):
			model_score += self.emission_probs[img][self.char_map[char]]
		if (model == 'ocr'):
			return math.exp(model_score)
		for i in range(1, len(image_vector)):
			model_score += self.transmission_probs[self.char_map[word[i-1]]][self.char_map[word[i]]]
		if (model == 'trans'):
			return math.exp(model_score)
		for i in range(len(image_vector)):
			for j in range(i+1, len(image_vector)):
				if (image_vector[i] == image_vector[j] and word[i] == word[j]):
					model_score += self.skip_potential
		if (model == 'skip'):
			return math.exp(model_score)
	
	def calc_prob(self, image_vector, word, model):
		assert(len(image_vector) == len(word))
		assert(model == 'ocr' or model == 'trans' or model == 'skip')
		word_score = self.model_score(image_vector, word, model)
		Z = self.partition(image_vector, model)
		cond_prob = word_score / Z
		return cond_prob

	def partition(self, image_vector, model):
		assert(model == 'ocr' or model == 'trans' or model == 'skip')
		all_words = itertools.product(self.char_map.keys(), repeat = len(image_vector))
		all_words = [''.join(word) for word in all_words]
		Z = 0.0
		for word in all_words:
			Z += self.model_score(image_vector, word, model)
		return Z

	def predict(self, image_vector, model, *args, **kwargs):
		assert(model == 'ocr' or model == 'trans' or model == 'skip')
		# print image_vector
		all_words = itertools.product(self.char_map.keys(), repeat = len(image_vector))
		all_words = [''.join(word) for word in all_words]
		predicted_word = ''
		highest_score = 0.0
		Z = 0.0
		true_word_score = 0.0
		true_word = kwargs.get('true_word', None)
		for word in all_words:
			curr_score = self.model_score(image_vector, word, model)
			if true_word == word:
				true_word_score = curr_score
			if (curr_score > highest_score):
				highest_score = curr_score
				predicted_word = word
			Z += curr_score
		predicted_prob = highest_score / Z
		true_word_prob = None
		if true_word is not None:
			true_word_prob = true_word_score / Z
		print predicted_word
		print true_word
		return {'predicted_word' : predicted_word, 'predicted_prob' : predicted_prob, 'true_word_prob' : true_word_prob}

	def accuracy(self, dataset, metric, model):
		assert(model == 'ocr' or model == 'trans' or model == 'skip')
		assert(dataset == 'small' or dataset == 'large')
		assert(metric == 'character' or metric == 'word' or metric == 'likelihood' or metric == 'all')
		if (dataset == 'small'):
			image_file = open('OCRdataset/data/small/images.dat')
			word_file = open('OCRdataset/data/small/words.dat')
		elif (dataset == 'large'):
			image_file = open('OCRdataset/data/large/allimages1.dat')
			word_file = open('OCRdataset/data/large/allwords.dat')
		all_images = [[int(a) for a in line.split()] for line in image_file.readlines()]
		all_words = [word.rstrip('\n') for word in word_file.readlines()]
		predicted_words = []
		predicted_probs = []
		true_word_probs = []
		for image_vector, true_word in zip(all_images, all_words):
			prediction = self.predict(image_vector, model, true_word = true_word)
			predicted_words.append(prediction['predicted_word'])
			predicted_probs.append(prediction['predicted_prob'])
			true_word_probs.append(prediction['true_word_prob'])
		if (metric == 'character') or (metric == 'all'):
			all_chars = ''.join(all_words)
			predicted_chars = ''.join(predicted_words)
			correct = 0
			for char, correct_char in zip(predicted_chars, all_chars):
				if(char == correct_char):
					correct = correct + 1
			sys.stdout.write('Character accuracy: ')
			print float(correct) / float(len(all_chars))
		if (metric == 'word') or (metric == 'all'):
			correct = 0
			for word, correct_word in zip(predicted_words, all_words):
				if(word == correct_word):
					correct = correct + 1
			sys.stdout.write('Word accuracy: ')
			print float(correct) / float(len(all_words))
		if (metric == 'likelihood') or (metric == 'all'):
			log_predicted_probs = [math.log(prob) for prob in true_word_probs]
			sys.stdout.write('Log likelihood: ')
			print sum(log_predicted_probs) / len(log_predicted_probs)


model = CRF()
model.accuracy(dataset = 'small', metric = 'all', model = 'skip')
model.accuracy(dataset = 'small', metric = 'all', model = 'ocr')
model.accuracy(dataset = 'small', metric = 'all', model = 'trans')