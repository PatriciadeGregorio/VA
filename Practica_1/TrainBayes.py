from sklearn.naive_bayes import GaussianNB
import numpy as np
train_data = np.genfromtxt('TrainBayes.csv', delimiter=',')
X = train_data[:,0:2]
Y = train_data[:,2]
gnb = GaussianNB()
gnb.fit(X,Y)
predicted = gnb.predict(X)
acierto_train = np.sum(1*(Y == predicted))/Y.shape[0]
print("El acierto en train es del ", acierto_train*100, "%")
test_data = np.genfromtxt('TestBayes.csv', delimiter=',')
X_test = test_data[:,0:2]
Y_test = test_data[:,2]
test_predicted = gnb.predict(X_test)
acierto_test = np.sum(1*(Y_test == test_predicted))/Y_test.shape[0]
print("El acierto en test es del ", acierto_test*100, "%")
prueba = np.array([[4.0, 2.0]])
print(gnb.predict(prueba))

matriz = np.zeros((80*80, 2))
cont = 0
for i in range(1, 80):
    for j in range (1, 80):
        matriz[cont, 0] = i/10
        matriz[cont, 1] = j/10
        cont = cont + 1
print(matriz[cont])

