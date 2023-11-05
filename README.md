# gender-classifier
Basic gender classify used to learn data science.
The program is simple. It uses the measurements of height, weight, and shoesize to attempt to predict the gender of the user.
The first function is tree, organizing the data into a decision tree until reaching its answer.
Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It creates a collection of decision trees and combines their predictions to make more accurate and robust predictions.
SVM is a supervised machine learning model used for classification and regression tasks. In this case, you're using an SVM classifier for binary classification (predicting "male" or "female"). SVM finds a hyperplane that best separates the data points of different classes, maximizing the margin between them.
K-Nearest Neighbors is a simple yet effective classification algorithm. Given a new data point, it identifies the k-nearest data points in the training dataset (based on distance metrics like Euclidean distance) and assigns the majority class among those neighbors as the predicted class for the new data point.
This code will create, train, and evaluate four classifiers and then print the predictions and accuracy of each classifier, finally identifying the classifier with the best accuracy.
