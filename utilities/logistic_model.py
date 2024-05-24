import copy, math
import numpy as np


def sigmoid(x):
    z = np.clip( x, -500, 500 )           # protect against overflow
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """
    Computes cost using logistic loss, non-matrix version

    Args:
      X (ndarray): Shape (m,n)  matrix of examples with n features
      y (ndarray): Shape (m,)   target values
      w (ndarray): Shape (n,)   parameters for prediction
      b (scalar):               parameter  for prediction
      lambda_ : (scalar, float) Controls amount of regularization, 0 = no regularization
      safe : (boolean)          True-selects under/overflow safe algorithm
    Returns:
      cost (scalar): cost
    """

    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i    = np.dot(X[i],w) + b                                             #(n,)(n,) or (n,) ()
        if safe:  #avoids overflows
            cost += -(y[i] * z_i ) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)                                                   #(n,)
            # Added small value to avoid log(0)
            cost += (-y[i] * np.log(f_wb_i + 1e-10) - (1 - y[i]) * np.log(1 - f_wb_i + 1e-10)) + (lambda_/(2*m))*np.dot(w,w)
    cost = cost/m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j]**2)                                               # scalar
        reg_cost = (lambda_/(2*m))*reg_cost

    return cost + reg_cost

def calculate_gradient_for_loop(X, y, w, b, l):
    m, n = X.shape
    dw = np.zeros((n,))  # initialize the gradient vector
    db = 0.              # initialize the intercept gradient
    
    for i in range(m):
        z = np.dot(X[i], w) + b
        a = sigmoid(z)
        dz = a - y[i]
        for j in range(n):
            dw[j] += X[i][j] * dz + l/m * w[j]
        db += dz
    return dw / m, db / m

def logistic_model(X, y, w_initial, b_initial, learning_rate=0.01, num_iterations=1000, l=0.0):
    J_history = []
    w = copy.deepcopy(w_initial)
    b = b_initial
    n = X.shape[1]
    
    for i in range(num_iterations):
        # Calculate the predicted values
        dw, db = calculate_gradient_for_loop(X, y, w, b, l)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
        for j in range(n):
            pass
        
            # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b, l) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")    
    
    return w, b, J_history