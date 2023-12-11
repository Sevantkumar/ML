#Program - 9:

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target
print(iris_data)
x_train,X_test,Y_train,Y_test = train_test_split(iris_data,iris_label,test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,Y_train)
y_predict = classifier.predict(X_test)
print(confusion_matrix(Y_test,y_predict))
print(classification_report(Y_test,y_predict))
