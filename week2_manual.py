import numpy as np  # this lets us do math with arrays


# This is our training data. Each row is a house, each column is a feature (size, bedrooms, floors)
X_NonScaled = np.array([[2104, 3, 2], [1600, 3, 1], [2400, 3, 2], [1416, 2, 1], [3000, 4, 2]])

# Find the average (mean) and spread (standard deviation) for each feature
X_mean = np.mean(X_NonScaled, axis=0)  # mean for each column
X_std = np.std(X_NonScaled, axis=0)    # std for each column
# Make all features have mean 0 and std 1 (feature scaling)
X = (X_NonScaled - X_mean) / X_std


# These are the actual prices for each house
y_NonScaled = np.array([399900, 329900, 369000, 232000, 539900])

# Find the mean and std for the prices
y_mean = np.mean(y_NonScaled)
y_std = np.std(y_NonScaled)
# Scale the prices so they have mean 0 and std 1
y = (y_NonScaled - y_mean) / y_std

m = y.shape[0]  # number of training examples (how many houses)

print("Number of training examples:", m)  # show how many houses we have


# Start with a bias (intercept) term, set to 1 (just for now)
bias = 1
# Start with all weights at zero (one for each feature)
weights = np.zeros(X.shape[1])
# Make a place to store our predictions
predictions = np.zeros(m)


# This function calculates predictions for all houses
def getPredictions(X, weights, bias):
    predictions = np.zeros(X.shape[0])  # make a place for predictions
    for i in range(m):  # for each house
        for j in range(X.shape[1]):  # for each feature
            predictions[i] += X[i][j] * weights[j]  # add up feature * weight
        predictions[i] += bias  # add the bias term
    return predictions


# This function calculates how wrong our predictions are (cost)
def compute_cost(predictions, y):
    m = len(y)
    cost = 0
    for i in range(m):
        cost += (predictions[i] - y[i]) ** 2  # squared error for each house
    return cost / (2 * m)  # average cost
    

# Make predictions with our starting weights and bias
predictions = getPredictions(X, weights, bias)
print("predictions:", predictions)  # show starting predictions


# Calculate how wrong our starting predictions are
mean_squared_error = compute_cost(predictions, y)


# Now let's learn! (Gradient Descent)
learningRate = 0.01  # how big a step to take each time
nIterations = 10000  # how many times to update weights
bias = 0  # start bias at zero
weights = np.zeros(X.shape[1])  # start weights at zero
predictions = np.zeros(m)  # reset predictions


# Repeat this many times to learn better weights
for iteration in range(nIterations):
    predictions = getPredictions(X, weights, bias)  # make predictions
    mean_squared_error = compute_cost(predictions, y)  # see how wrong we are

    # Make places to store how much to change weights and bias
    dW = np.zeros(X.shape[1])  # gradient for weights
    dB = 0  # gradient for bias

    # Figure out how much to change each weight
    for j in range(X.shape[1]):  # for each feature
        for i in range(m):  # for each house
            dW[j] += (predictions[i] - y[i]) * X[i][j]  # error * feature value
        dW[j] = dW[j] / m  # average over all houses

    # Figure out how much to change the bias
    for i in range(m):
        dB += (predictions[i] - y[i])  # error for each house
    dB = dB / m  # average over all houses

    # Actually update weights and bias
    for j in range(X.shape[1]):
        weights[j] = weights[j] - learningRate * dW[j]  # move weights
    bias = bias - learningRate * dB  # move bias

    # Show what is happening each time
    print("Iteration", iteration, "predictions:", predictions, "cost:", mean_squared_error, "weights:", weights, "bias:", bias)
    predictions_unscaled = predictions * y_std + y_mean  # turn predictions back to real prices
    print("Unscaled predictions:", predictions_unscaled)



