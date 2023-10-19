from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
clf2 = RandomForestClassifier()
clf3 = SVC()
clf4 = KNeighborsClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])
prediction4 = clf4.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print(prediction)

from sklearn.metrics import accuracy_score

accuracy1 = accuracy_score(Y, clf.predict(X))
accuracy2 = accuracy_score(Y, clf2.predict(X))
accuracy3 = accuracy_score(Y, clf3.predict(X))
accuracy4 = accuracy_score(Y, clf4.predict(X))

# Create a dictionary to store classifiers and their accuracies
classifiers = {
    "Decision Tree": accuracy1,
    "Random Forest": accuracy2,
    "SVM": accuracy3,
    "KNN": accuracy4
}

# Find the classifier with the highest accuracy
best_classifier = max(classifiers, key=classifiers.get)
best_accuracy = classifiers[best_classifier]

print("Predictions:")
print("Decision Tree:", prediction)
print("Random Forest:", prediction2)
print("SVM:", prediction3)
print("KNN:", prediction4)

print("Classifier with the best accuracy:", best_classifier)
print("Best accuracy:", best_accuracy)