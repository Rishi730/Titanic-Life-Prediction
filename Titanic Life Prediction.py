#Code done on Google Colab

import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv("/content/Titanic-Dataset.csv")
print(data)

data.head()

data.head()

data.shape

data.describe()

"""# New Section"""

data.tail()

data.info()

data.isnull().sum()

a=data["Age"].mean()

print(a)

data['Age'].fillna(a,inplace=True)

data.info()



data["Sex"]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Sex"]=le.fit_transform(data["Sex"])

data["Sex"]

data.info()

x=data[["Pclass","Sex","Age","Fare"]]

x.head()

y=data["Survived"]

y.head()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

x_test.shape

data.corr()

sns.heatmap(data.corr(),annot=True)



sns.scatterplot(y=data["Age"],x=data["Fare"],hue=data["Survived"])

sns.swarmplot(y=data["Age"],data=data,hue=data["Survived"])

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
mo=DecisionTreeClassifier()

model.fit(x_train,y_train)
mo.fit(x_train,y_train)

test=np.array([3,1,29,7.00])
test=test.reshape(1,-1)

print(model.predict(test))
print(mo.predict(test))

y_pred=model.predict(x_test)
y_pred

y_prede=mo.predict(x_test)
y_prede

from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

accuracy_score(y_test,y_pred)

accuracy_score(y_test,y_prede)

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
confusion_matrix(y_test,y_pred)