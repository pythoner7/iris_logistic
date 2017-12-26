import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
if __name__ == '__main__':
    path = 'E:\\python program\\ML code\\linear regression\\Advertising.csv'
    data = pd.read_csv(path)
    print(data.head())
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    print(type(x), type(y))
    # plt.figure()
    # plt.plot(x, y, 'ro')
    # plt.grid()
    #plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print(x_train)
    # print(x_test)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    # print(linreg.intercept_)
    # print(linreg.coef_)
    y_pred = model.predict(x_test)
    sum_mean = 0
    for i in range(len(y_test)):

        sum_mean+=(y_pred[i] - y_test.values[i])**2
    rmse = np.sqrt(sum_mean/len(y_pred))
    print(rmse)
    mse = np.sqrt(mean_squared_error(y_test,y_pred))
    print(mse)
    #print(y_pred)
    plt.plot(range(len(y_test)), y_test, 'r')
    plt.plot(range(len(y_test)), y_pred, 'b')
    plt.grid()
    #plt.show()