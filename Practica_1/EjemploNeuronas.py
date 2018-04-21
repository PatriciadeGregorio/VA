import numpy as np

x1_a = np.random.normal(7,1,100)
x2_a = x1_a * 2 + np.random.normal(5,1,100)
x1_b = np.random.normal(9,1,100)
x2_b = x1_b * 2 + np.random.normal(11,1,100)
x1_c = np.random.normal(10,1,100)
x2_c = np.random.normal(20,1,100)
from matplotlib import pyplot as plt
plt.plot(x1_a,x2_a,'+')
plt.plot(x1_b,x2_b,'o')
plt.plot(x1_c,x2_c,'.')
plt.show()
x_a = np.array([x1_a,x2_a])
x_b = np.array([x1_b,x2_b])
x_c = np.array([x1_c,x2_c])
X = np.concatenate((x_a,x_b,x_c),axis=1)
X = X.transpose()
y = np.concatenate((np.zeros(100),np.ones(100),2*np.ones(100)))
from sklearn.neural_network import MLPClassifier
net = MLPClassifier(solver='sgd',
activation='logistic',
learning_rate_init=0.01,
hidden_layer_sizes=(20, 20),
random_state=1,
shuffle=True,
verbose=True,
tol=0.000001,
max_iter=100000)
net.fit(X, y)

net.predict(np.array([[6.5,25.0]]))
net.predict(np.array([[9.2,23.0]]))
net.predict(np.array([[10.0,20.0]]))