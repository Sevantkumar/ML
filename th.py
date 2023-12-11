import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
PlayTennis = pd.read_csv("PlayTennis.csv")
print(PlayTennis)

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()

PlayTennis['outlook']=Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp']=Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity']=Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy']=Le.fit_transform(PlayTennis['windy'])
PlayTennis['play']=Le.fit_transform(PlayTennis['play'])
# PlayTennis

y=PlayTennis['play']
x=PlayTennis.drop(['play'],axis=1)

from sklearn import tree

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(x,y)

tree.plot_tree(clf)
plt.show()
x_pred=clf.predict(x)
x_pred=y
