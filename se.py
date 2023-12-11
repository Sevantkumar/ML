import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
iris=datasets.load_iris()

X=pd.DataFrame(iris.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
Y=pd.DataFrame(iris.target)
Y.columns=['Targets']


model=KMeans(n_clusters=3,n_init=10)
model.fit(X)

plt.figure(figsize=(14,14))
colormap=np.array(['red','lime','black'])
plt.subplot(2,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[Y.Targets],s=40)
plt.title('Real clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')


plt.subplot(2,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model.labels_],s=40)
plt.title("K-means clustering")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')




#general EM for GMM
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
scaler.fit(X)
xsa=scaler.transform(X)
xs=pd.DataFrame(xsa,columns=X.columns)


from sklearn.mixture import  GaussianMixture
plt.figure(figsize=(14,14))
colormap=np.array(['red','lime','black'])
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y=gmm.predict(xs)
plt.subplot(2,2,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[gmm_y],s=40)

plt.title('GMM clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
print("Observation: The GMM using EM algorithm based clusters matched the tree labels more closely than the k-means")
