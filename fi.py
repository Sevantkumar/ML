import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
df=pd.read_csv("pima_indian.csv")
feature_col_names=['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_names=['diabetes']
x=df[feature_col_names].values
y=df[predicted_class_names].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33)
print('\n the total number of training data:',ytrain.shape)
print('\n the total number of test data:',ytest.shape)
clf=GaussianNB().fit(xtrain,ytrain.ravel())
predicted=clf.predict(xtest)
predictTestData=clf.predict([[6,148,72,35,0,33.6,0.267,50]])
print("Confusion matrix:")
print(metrics.confusion_matrix(ytest,predicted))
print('\n Accuracy of classifier is:',metrics.accuracy_score(ytest,predicted))
print('\n the value of precision',metrics.precision_score(ytest,predicted))
print('\n the value of recall',metrics.recall_score(ytest,predicted))
print("Predicted value for individual TestData:",predictTestData)
