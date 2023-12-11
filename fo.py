from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets

iris= datasets.load_iris()
iris_data=iris.data
iris_lables=iris.target
print(f"Sl No.\tY\tX")
for i in range(len(iris_lables)):
    print(f"\t{i+1}\t{iris_lables[i]}\t{iris_data[i]}")
X_train, X_test, Y_train, Y_test = train_test_split( iris_data,iris_lables, test_size=0.20)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,Y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
