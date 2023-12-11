import pandas as pd

msg = pd.read_csv("textdocx.csv",names=['message','label'])
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.labelnum
print(X.shape)
print(Y.shape)
print("\nThe message and its label of first 5 instances are listed below")

X5,y5=X[0:],msg.label[0:]
for x,y in zip(X5,y5):
    print(x,' ',y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)

print("\nDatasetis split into training and testing samples")
print("Total training instances: ",xtrain.shape[0])
print("Total testing instances: ",xtest.shape[0])


from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
xtrain_dtm=count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)

print("\nTotal features extracted using count Vectroizer: ",xtrain_dtm.shape[1])

print("\nFeature for first 5 training instances are listed below")

df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names_out())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)

print("Classification result of tesing samples are given below")

for doc,p in zip(xtest,predicted):
    if p==1:
        pred='pos'
    else:
        pred='neg'
    print("%s -> %s" %(doc,pred))

from sklearn.metrics import confusion_matrix,classification_report

print('Confusion Matrix')
print(confusion_matrix(ytest,predicted))
print('classification_report')
print(classification_report(ytest,predicted))
