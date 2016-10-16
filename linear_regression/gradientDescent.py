import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
	m = y.size
	J_history = np.zeros(num_iters)
	temp = np.zeros(theta.size)
	numParameters = theta.size

	for iter in range(num_iters):
		for j in range(numParameters):
			delta_j = 0
			for i in range(m):
				delta_j += (X[i, :].dot(theta) - y[i]) * X[i, j]
			temp[j] = theta[j] - alpha * (delta_j / m)
		theta = temp
		J_history[iter] = computeCost(X, y, theta)

	return (theta, J_history)