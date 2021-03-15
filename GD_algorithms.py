"""
Error functions :
"""

# This is the Mean Squared Error (MSE) function that is used to compute the loss
def MSE(a,b,data):
    # E(a,b) = (y1 - (a*x1 +b))^2 + (y2 - (a*x2+b))^2 + ..... + (yN - (a*xN+b))^2
    sumError = 0
    for x,y in data: #x,y coordinates of the points
        sumError+=(y-(a*x+b))**2
    return sumError


# This is the gradient of the MSE function that is used to update the equation's coefficients (a and b)
def gradientMSE(a,b,data):
    # gradA = dE(a,b)/da = -2*x*(y-a*x+b)
    # gradB = dE(a,b)/db = -2*(y-a*x+b)
    gradientA = 0;
    gradientB = 0;
    for x,y in data:
        gradientA += -2*x*(y-(a*x+b))
        gradientB += -2*(y-(a*x+b))

    return np.array([gradientA,gradientB])



# This is the Error function used to compute the loss for SGD
def E(a,b,data):
    # E(a,b) = (y - (a*x +b))^2
    x,y = data # x and y coordinates of the point
    return (y-(a*x+b))**2



# This is the gradient of the Error function used to update the equation's coefficients (a and b) for SGD
def gradientE(a,b,data):
    # gradA = dE(a,b)/da = -2*x*(y-a*x+b)
    # gradB = dE(a,b)/db = -2*(y-a*x+b)
    x,y = data
    gradientA = -2*x*(y-(a*x+b))
    gradientB = -2*(y-(a*x+b))

    return np.array([gradientA,gradientB])




"""
Algorithms : 
"""



# This is the classic gradient descent algorithm
def gradientDescent(MSE, gradientMSE,data, startingCoefficients, learningRate,nbIteration):

    coefHistory = [] # Used to save the value of a and b at each iteration
    lossHistory = [] # Used to save the value of the loss at each iteration
    gradientHistory = [] # Used to save the value of the gradient at each iteration
    X = startingCoefficients # This is a tuple (a,b) with a and b the starting coefficients

    for _ in range(nbIteration): # Here, one iteration = one epoch

        loss = MSE(*X,data) # Compute the loss thanks to the data and the coefficients a and b
        grad = gradientMSE(*X,data) # Compute the gradient

        # Save the coefficients, loss and gradient of the current iteration (epoch) in the lists
        coefHistory.append(X)
        lossHistory.append(loss)
        gradientHistory.append(grad)

        X = X-learningRate*grad # Update the coefficients a and b

    return coefHistory, lossHistory, gradientHistory



# This is the stochastic gradient descent algorithm
def stochasticGradientDescent(E, gradientE,data, startingCoefficients, learningRate,nbIteration):

    coefHistory = [] # Used to save the value of a and b at each iteration
    lossHistory = [] # Used to save the value of the loss at each iteration
    gradientHistory = [] # Used to save the value of the gradient at each iteration
    X = startingCoefficients # This is a tuple (a,b) with a and b the starting coefficients

    for _ in range((int)(nbIteration/len(data))): # nbIterations/len(data) = number of epochs

        for i in range(len(data)): # len(data) iterations are made here

            loss = E(*X,data[i]) # Compute the loss for one data and the coefficients a and b
            grad = gradientE(*X,data[i]) # Compute the gradient for one data

            # Save the coefficients, loss and gradient of the current data in the lists
            coefHistory.append(X)
            lossHistory.append(loss)
            gradientHistory.append(grad)

            X = X-learningRate*grad # Update the coefficients a and b

    return coefHistory, lossHistory, gradientHistory



# This is mini-batch stochastic gradient descent algorithm
def miniBatchSGD(E, gradientE,data, startingCoefficients, learningRate,nbEpoch,batchSize):

    coefHistory = [] # Used to save the value of a and b at each iteration
    lossHistory = [] # Used to save the value of the loss at each iteration
    gradientHistory = [] # Used to save the value of the gradient at each iteration
    X = startingCoefficients # This is a tuple (a,b) with a and b the starting coefficients

    for _ in range(nbEpoch):

        for i in range((int)(np.ceil(len(data)/batchSize))): # Number of steps (iterations)
            batchData = data[i*batchSize:i*batchSize+batchSize] #Create the sublist that contains batchSize data

            loss = MSE(*X,batchData) # Compute the loss thanks to the data and the coefficients a and b
            grad = gradientMSE(*X,batchData) # Compute the gradient

            # Save the coefficients, loss and gradient of the current iteration (epoch) in the lists
            coefHistory.append(X)
            lossHistory.append(loss)
            gradientHistory.append(grad)

            X = X-learningRate*grad # Update the coefficients a and b

    return coefHistory, lossHistory, gradientHistory


"""
For more examples, check the notebook
"""
