#Program -8:
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
iris=datasets.load_iris()
X=pd.DataFrame(iris.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
Y=pd.DataFrame(iris.target)
Y.columns=['Targets']
model= KMeans(n_clusters=3,n_init=10)
model.fit(X)
plt.figure(figsize=(14,14))
colormap=np.array(['red','lime','black'])

plt.subplot(2,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[Y.Targets],s=40)
plt.title('Real Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(2,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model.labels_],s=40)
plt.title('K Means Cluster')
plt.xlabel('Petal Length')
plt.ylabel('Petal width')

from sklearn import preprocessing

scalar = preprocessing.StandardScaler()
scalar.fit(X)
xsa = scalar.transform(X)
xs= pd.DataFrame(xsa,columns=X.columns)

from sklearn.mixture import GaussianMixture
plt.figure(figsize=(14,14))
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
colormap= np.array(['red','lime','black'])
gmmy=gmm.predict(xs)

plt.subplot(2,2,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[gmmy],s=40)
plt.title('GMM Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()




