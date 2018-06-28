import numpy as np
from sklearn.svm import SVC
clf=SVC()
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
import csv
with open('/home/abhishek/Downloads/fraud_test/train.csv', "r") as f:
    reader = csv.reader(f)
    x=[]
    for row in reader:
        x.append(row)

f.close()

for i in range(1,len(x)):
    x_new=[]
    for j in range(0,len(x[i])-1):
        if j>0 and j<8:
            x_new.append(float(x[i][j]))
        if j>25 and j<(len(x[i])-1):
            x_new.append(float(x[i][j]))
    X_train.append(x_new)
    Y_train.append(int(x[i][50]))
    
    
import csv
with open('/home/abhishek/Downloads/fraud_test/test.csv', "r") as f:
    reader = csv.reader(f)
    x=[]
    for row in reader:
        x.append(row)

f.close()

for i in range(1,len(x)):
    x_new=[]
    for j in range(0,len(x[i])):
        if j>0 and j<8:
            x_new.append(float(x[i][j]))
        if j>25 and j<(len(x[i])):
            x_new.append(float(x[i][j]))
    X_test.append(x_new)

clf.fit(X_train,Y_train)

