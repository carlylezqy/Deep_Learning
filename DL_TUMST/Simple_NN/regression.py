import numpy as np
import pandas as pd
import layers
from sklearn.model_selection import train_test_split

np.random.seed(0)

df = pd.read_csv('../dataset/housing.data', header=None, sep='\s+')
df.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

y = np.array(df['MEDV']).reshape(-1, 1)
X = np.array(df.drop(['MEDV'], axis=1))
 
x_mean = X.mean(axis=0) ##(13,)
x_variant = (X ** 2).mean(axis=0) - x_mean ** 2 ##(13,)
x_std = np.array((X - x_mean) / np.sqrt(x_variant)) ## (506, 13)

X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.25)
model = layers.regr_layer(input_size=13, hidden_size=10, output_size=1, learning_rate=0.01)

for i in range(5000):
    batch = np.random.choice(len(X_train), 100, replace=False)
    X_train_batch = X_train[batch]
    y_train_batch = y_train[batch]

    model.forward(X_train_batch)
    model.backward(y_train_batch)

    if(i%500 == 0):
        train_predict = model.forward(X_train).flatten()
        train_correct = y_train.flatten()
        mse_train = np.mean((train_predict-train_correct)**2)

        #テストデータに対する精度の計算
        test_predict = model.forward(X_test).flatten()
        test_correct = y_test.flatten()
        mse_test = np.mean((test_predict-test_correct)**2)

        print('iteration:', i, 'train mse =', mse_train, 'test mse =', mse_test)

predict = model.forward(X_test).flatten()
correct = y_test.flatten()
RMS = np.mean((predict - correct) ** 2)
print('テストデータに対する2乗誤差の平均', RMS)

compare = pd.DataFrame(np.array([correct, predict]).T)
compare.columns = ['正解', '予測値']
print(compare[:20])