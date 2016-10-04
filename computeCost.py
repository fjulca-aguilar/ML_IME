import numpy as np
def computeCost(X, y, theta):
	m = y.size
	J = 0
	for i in range(m):
		J += np.square(X[i, :].dot(theta) - y[i])
	J /= (2 * m)
	return (J)