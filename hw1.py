import time
import numpy as np
import random

TRAINING_DATA_PATH = 'letter-recognition-training.txt'
TEST_DATA_PATH = 'letter-recognition-test.txt'

def test_knn(train_x, train_y, test_x, num_nn):
	results = np.empty(shape=[len(test_x),], dtype=str)
	t_start = time.time()
	closest = np.array([np.array([np.sum((train[:] - test[:])**2) for train in train_x]).argsort()[:num_nn] for test in test_x])
	classes = train_y[closest]
	i=0
	for point in classes:
		letters = {}
		for letter in point:
			if letter not in letters:
				letters[letter] = 1
			else:
				letters[letter] += 1
		results[i] = max(letters, key=letters.get)
		i += 1
	t_end = time.time()
	print ('\ntrain knn (' + str(num_nn) + ', ' + str(len(train_x)) + '): ' + str(t_end - t_start))
	return results

def condense_data(train_x, train_y):
	t_start = time.time()
	t_end = time.time()
	print ('\ncondense: ' + str(t_end - t_start))

	return None

def train_pocket(train_x, train_y, num_iters):
	t_start = time.time()
	train_y_let = train_y.view(np.uint8) - 65
	train_x_let = np.concatenate((np.ones(shape=[len(train_x), 1], dtype=int).T, train_x.T)).T
	weights = np.random.rand(len(train_x_let[0]))
	y_labels = np.empty(shape=[len(train_y_let),], dtype=int)
	models = np.zeros(shape=[26,len(train_x_let[1])])
	errors = np.zeros(shape=[26,])
	for i in range(0, 26):
		for j in range(0, len(train_y_let)):
			if train_y_let[j] - i == 0:
				y_labels[j] = 1
			else:
				y_labels[j] = -1
		for j in range (0, num_iters):
			hypothesis = np.sign(train_x_let.dot(weights))
			good = True
			while good:
				k = random.randint(0, len(train_x_let) - 1)
				if (y_labels[k] != hypothesis[k]):
					weights = weights + y_labels[k] * train_x_let[k]
					good = False
			error = sum((train_x_let.dot(weights) - y_labels) ** 2) / len(train_x_let)
			if error < errors[i] or errors[i] == 0:
				models[i] = weights
				errors[i] = error
	t_end = time.time()
	print ('\ntrain pocket:  ' + str(t_end - t_start))
	print (models)
	return models

def test_pocket(w, test_x):
	t_start = time.time()
	test_x_let = np.concatenate((np.ones(shape=[len(test_x), 1], dtype=int).T, test_x.T)).T
	predictions = test_x_let.dot(w[0].T)
	pred_y = np.empty(shape=[len(test_x),])
	for i in range(1, len(predictions)):
		pred_y[i] = chr(max(predictions[i]) + 65)
	t_end = time.time()
	print ('\ntest pocket:  ' + str(t_end - t_start))
	return pred_y

def compute_accuracy(test_y, pred_y):
	t_start = time.time()
	diff = test_y.view(np.uint8) - pred_y.view(np.uint8)
	diff_num = np.count_nonzero(diff)
	accuracy = (len(test_y) - diff_num)/float(len(test_y))
	print('accuracy: ' + str(accuracy))
	t_end = time.time()
	print ('\naccuracy computing:  ' + str(t_end - t_start))
	return accuracy

def get_id():
	return 'tug73611'

def confusion_matrix(test_y, pred_y):
	conf_mat = np.zeros(shape=[26, 26], dtype=int)
	test_y_let = test_y.view(np.uint8) - 65
	pred_y_let = pred_y.view(np.uint8) - 65
	for i in range(0, len(test_y_let)):
		for j in range(0, len(pred_y_let)):
			conf_mat[test_y_let[i], pred_y_let[j]] += 1
	print('\nconfusion matrix:')
	print conf_mat

def main():
	num_train = 15000
	num_iters = 2
	num_train_size = [100, 1000, 2000, 5000, 10000, 15000]
	knn_size = [1, 3, 5, 7, 9]
	train_x_full = np.genfromtxt(TRAINING_DATA_PATH, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), dtype=int)
	train_y_full = np.genfromtxt(TRAINING_DATA_PATH, delimiter=',', usecols=(0,), dtype=str)
	test_x = np.genfromtxt(TEST_DATA_PATH, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), dtype=int)
	test_y = np.genfromtxt(TEST_DATA_PATH, delimiter=',', usecols=(0,), dtype=str)
	for sample_size in num_train_size:
		if sample_size < 15000:
			sample = np.array(random.sample(xrange(1,num_train), sample_size))
			train_x = np.array([train_x_full[j] for j in sample])
			train_y = np.array([train_y_full[j] for j in sample])
		else:
			train_x = train_x_full
			train_y = train_y_full
		for num_nn in knn_size:
			pred_y = test_knn(train_x, train_y, test_x, num_nn)
			acc = compute_accuracy(test_y, pred_y)
		 	conf_mat = confusion_matrix(test_y, pred_y)
		# condensed function is not done
		# condensed_idx = condense_data(train_x, train_y)
		# acc = compute_accuracy(test_y, pred_y)
		# conf_mat = confusion_matrix(test_y, pred_y)
		w = train_pocket(train_x, train_y, num_iters),
		pred_y = test_pocket(w, test_x)
		acc = compute_accuracy(test_y, pred_y)
		conf_mat = confusion_matrix(test_y, pred_y)

	return None

if __name__ == "__main__":
	main()
