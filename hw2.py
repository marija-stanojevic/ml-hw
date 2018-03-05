import csv
import time
import numpy as np
import tensorflow as tf

def train_classifier(train_x, train_y, num_iters, learn_rate, loss, regularizer):
	lmb = 5
	session = tf.Session()
	#w = np.zeros(shape=(np.shape(train_x)[1],1))
	w = np.random.rand(np.shape(train_x)[1], 1) - 0.5
	t_start = time.time()
	for i in range(1, num_iters):
		if loss == 'squared':
			# squared loss: l(y, y') = (y - y')^2; gradient: alpha * (- (y - xw)* x^T)
			w = tf.add(w, tf.scalar_mul(learn_rate, tf.matmul(tf.transpose(train_x), tf.subtract(train_y, tf.matmul(train_x, w)))))
			#had to keep w as tensor, because there is an error when I try to use w that I transfered from tensor to numpy array
		elif loss == 'hinge':
			#hinge loss: l(y, y') = max(0, 1 - yy'); gradient: alpha * (-yx, if yxw < 1; 0 otherwise)
			condition = session.run(tf.multiply(train_y, tf.matmul(train_x, w))).reshape(len(train_x),)
			coef = np.zeros(shape=len(train_x))
			coef[condition < 1] = 1
			w = tf.add(w, tf.scalar_mul(learn_rate, tf.matmul(tf.transpose(train_x), tf.multiply(coef.reshape(len(train_y),1), train_y))))
		elif loss == 'logistic':
			#logistic loss: l(y,y') = 1/ln 2 * ln(1 + exp(-yy')); gradient: alpha * (-1/N * sum(yx / (1 + exp(y * w^T * x))))
			multiplier = session.run(tf.matmul(tf.transpose(train_y), tf.matmul(train_x, w)))[0, 0]
			multiplier = 1 / (1 + multiplier) * learn_rate/len(train_x)
			w = tf.add(w, tf.scalar_mul(multiplier, tf.matmul(tf.transpose(train_x), train_y)))
		if regularizer == 'l1':
			w = tf.subtract(w, tf.scalar_mul(learn_rate * lmb, tf.sign(w)))
		elif regularizer == 'l2':
			w = tf.subtract(w, tf.scalar_mul(learn_rate * lmb, w))
	t_end = time.time()
	#print ('Training lasted ' + str(t_end - t_start) + ' for ' + str(num_iters) + ' iterations')
	return session.run(w)

def test_classifier(w, test_x):
	t_start = time.time()
	pred_y = np.sign(test_x.dot(w))
	t_end = time.time()
	#print ('Test lasted:  ' + str(t_end - t_start))
	return pred_y

def compute_accuracy(test_y, pred_y):
	t_start = time.time()
	diff = sum(abs(test_y - pred_y))/2
	accuracy = ((len(test_y) - diff) / float(len(test_y)))[0]
	t_end = time.time()
	#print('Accuracy: ' + str(accuracy))
	#print ('Accuracy computing lasted:  ' + str(t_end - t_start))
	return accuracy

def cross_validation(train_x, train_y, num_cv, epochs, learning_rate, cv_acc, loss, regularization):
	for i in range(0, 5):
		cv_x = train_x[i * num_cv: (i + 1) * num_cv, :]
		cv_y = train_y[i * num_cv: (i + 1) * num_cv, :]
		if i > 0 and i < 4:
			train_x_red = np.concatenate((train_x[0: i * num_cv, :], train_x[(i + 1) * num_cv : , :]))
			train_y_red = np.concatenate((train_y[0: i * num_cv, :], train_y[(i + 1) * num_cv : , :]))
		elif i < 4:
			train_x_red = train_x[(i + 1) * num_cv : , :]
			train_y_red = train_y[(i + 1) * num_cv : , :]
		else:
			train_x_red = train_x[0: i * num_cv, :]
			train_y_red = train_y[0: i * num_cv, :]
		w = train_classifier(train_x_red, train_y_red, epochs, learning_rate, loss, regularization)
		pred_y = test_classifier(w, cv_x)
		cv_acc[i] = compute_accuracy(cv_y, pred_y)
	return np.average(cv_acc)

def read_data(train_x, train_y, test_x, test_y, num_train, num_test, num_dims):
	random_test_exmp = np.random.choice(num_train + num_test, num_test, replace=False)
	i = 0
	j = 0
	with open('winequality-white.csv') as csvfile:
		reader = csv.reader(csvfile)
		first = True
		for row in reader:
			if first:
				first = False
			else:
				vals = np.array((row[0]).split(';'))
				vals = vals.astype(np.float)
				vals = np.insert(vals, 0, [1])
				if vals[num_dims] == 6:
					continue
				elif vals[num_dims] > 6:
					vals[num_dims] = 1
				else:
					vals[num_dims] = -1
				if i in random_test_exmp:
					test_x[i - j, :] = vals[ : num_dims]
					test_y[i - j, :] = vals[num_dims]
				else:
					train_x[j, :] = vals[ : num_dims]
					train_y[j, :] = vals[num_dims]
					j += 1
				i += 1
	for i in range(1, num_dims):
		max_val = max(train_x[:, i])
		max_val_test = max(test_x[:, i])
		max_val = max(max_val, max_val_test)
		train_x[:, i] /= max_val
		test_x[:, i] /= max_val
	return None

def main():
	num_train = 2200
	num_test = 500
	num_dims = 12
	n_fold = 5
	epochs = [5, 10, 15, 20, 25, 30, 45, 60, 80, 100]
	learning_rate = 0.01
	train_x = np.zeros(shape=(num_train, num_dims))
	train_y = np.zeros(shape=(num_train, 1))
	test_x = np.zeros(shape=(num_test, num_dims))
	test_y = np.zeros(shape=(num_test, 1))
	read_data(train_x, train_y, test_x, test_y, num_train, num_test, num_dims)
	num_cv = num_train / n_fold
	svm_cv_acc = np.zeros(shape=(n_fold,))
	logistic_cv_acc = np.zeros(shape=(n_fold,))
	svm_accuracy = np.zeros(shape=(len(epochs), ))
	logistic_accuracy = np.zeros(shape=(len(epochs), ))
	k = 1
	for i in range (0, len(epochs)):
		svm_cv_avg = 0
		logistic_cv_avg = 0
		for j in range(0,k):
			# svm
			svm_cv_avg += cross_validation(train_x, train_y, num_cv, epochs[i], learning_rate, svm_cv_acc, 'hinge', 'l2')
			w = train_classifier(train_x, train_y, epochs[i], learning_rate, 'hinge', 'l2')
			pred_y = test_classifier(w, test_x)
			svm_accuracy[i] = compute_accuracy(test_y, pred_y)
			#logistic
			logistic_cv_avg += cross_validation(train_x, train_y, num_cv, epochs[i], learning_rate, logistic_cv_acc, 'logistic', 'none')
			w = train_classifier(train_x, train_y, epochs[i], learning_rate, 'logistic', 'none')
			pred_y = test_classifier(w, test_x)
			logistic_accuracy[i] = compute_accuracy(test_y, pred_y)
		print('SVM 5-fold cross validation accuracy average for ' + str(epochs[i]) + ' epochs is ' + str(svm_cv_avg / k) + ' and test accuracy is ' + str(svm_accuracy[i] / k))
		print('Logistic regression 5-fold cross validation accuracy average for ' + str(epochs[i]) + ' epochs is ' + str(logistic_cv_avg / k) + ' and test accuracy is ' + str(logistic_accuracy[i] / k))
	return None

if __name__ == "__main__":
	main()
