# loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# 2. Classification - K Nearest Neighbors (No/less training computation)
iris = datasets.load_iris() # loading datasets
# print(iris.DESCR) # print description of features and classes(labels)
features = iris.data
labels = iris.target
# print(features[0], labels[0])

# Training classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[9.1, 9.5, 6.4, 0.2]])
print(preds)