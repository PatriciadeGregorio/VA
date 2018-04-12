from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
inputs = np.array([[1,1],[2,2],[3,3],[4,4], [1,3],[2,4],[3,5],[4,6]],dtype=np.int)
outputs = np.array([0,0,0,0,1,1,1,1])
lda = LinearDiscriminantAnalysis()
lda.fit(inputs,outputs)
prediction = lda.predict(inputs)
print(lda.transform(inputs))