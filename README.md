# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Dharini PV
RegisterNumber:  212222240024
```
```python
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

import chardet 
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

## Result output

![image](https://github.com/DHARINIPV/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119400845/0ec58261-8df8-4f45-9c6e-440464f071d0)

## data.head()

![image](https://github.com/DHARINIPV/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119400845/74d967ba-5f1f-4996-b11d-c61d06c92c3d)

## data.info()

![image](https://github.com/DHARINIPV/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119400845/8267b319-1e42-4892-969d-2afeae06d4ce)

## data.isnull().sum()

![image](https://github.com/DHARINIPV/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119400845/34e8d223-7ad9-47b1-857b-1642f35c40b7)

## Y_prediction value

![image](https://github.com/DHARINIPV/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119400845/72d1f9a9-31aa-449c-b46c-26219024ddca)

## Accuracy value

![image](https://github.com/DHARINIPV/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119400845/8cde9bc5-164e-4de5-a126-a2aedda4ddc0)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
