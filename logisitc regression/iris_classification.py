import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
def iris_type(s):
    s = str(s,encoding='utf-8')
    it = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    return it[s]
if __name__ == '__main__':
    path = 'E:\\python program\\ML code\\logisitc regression\\iris.data'
    #pandas读数据
    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes
    print(data.head())

    #numpy读数据
    # data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    # print(data)
    x = data[[0, 1, 2, 3]]
    y = data[4]
    print(x)
    logreg = LogisticRegression()
    model = logreg.fit(x,y)
    #预测值所在区域的显示
    x1_min, x1_max = data[0].min(), data[0].max()
    x2_min, x2_max = data[1].min(), data[1].max()
    t1 = np.linspace(x1_min,x1_max,500)
    t2 = np.linspace(x2_min,x2_max,500)
    x1,x2 = np.meshgrid(t1,t2)
    x_test = np.stack((x1.flat, x2.flat),axis=1)
    print(len(x1),len(x2))
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(x1.shape)
    plt.pcolormesh(x1,x2,y_pred,cmap=plt.cm.prism)
    #plt.show()
    print(y_pred)
    plt.scatter(data[0],data[1],c=y,edgecolors='k',cmap=plt.cm.prism)
    plt.xlabel('length')
    plt.ylabel('width')
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.grid()
    plt.show()


    y_pred = model.predict(x)
    y = y.values.reshape(-1)
    result = y_pred == y
    acc = np.count_nonzero(result)
    print('accuracy:%.2f%%'%(100 * float(acc)/float(len(result))))
