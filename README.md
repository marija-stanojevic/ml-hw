<h3> Machine Learning Class - Homework solutions (October 2017)</h3>

These are solutions to homework 1 and homework 2 in machine learning class. Both of them required implementation of certain machine learning methods without using existing libraries except for basic functionalities and data structures.

<h4> Homework 1: </h4> Uses Letter Recognition Data Set from the UCI Machine Learning (https://archive.ics.uci.edu/ml/datasets/letter+recognition). First 15,000 examples are for training and the remaining 5,000 for testing. Homework is implementing basic k-NN classification (test_knn method) and "pocket" algorithm (train_pocket and test_pocket methods). Additionally, method compute_accuracy computes the accuracy of methods results.

<h4> Homework 2: </h4> Implements gradient descent (train_classifier method) for different cost functions (squared, hinge, logistic) without regularizer and with L1 and L2 regularizers. Classifiers can be crossvalidated for different parameters values (cross_validation method), tested using test_classifier method and evaluated using compute_accuracy method. System is tested on Wine Quality data set from the  UCI Machine Learning (https://archive.ics.uci.edu/ml/datasets/wine+quality).
