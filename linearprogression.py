import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Nairobi Office Price Ex.csv'  # Ensure this file is in the same directory
data = pd.read_csv(file_path)

# Extract relevant columns: SIZE (feature) and PRICE (target)
x = data['SIZE'].values
y = data['PRICE'].values

# Define the Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the Gradient Descent function to update m (slope) and c (y-intercept)
def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c  # Predicted values
    # Calculate gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    # Update parameters
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize random values for slope (m) and intercept (c)
np.random.seed(0)
m, c = np.random.rand(), np.random.rand()

# Set training parameters
epochs = 10
learning_rate = 0.0001
errors = []

# Training loop for 10 epochs
for epoch in range(epochs):
    # Predict y with current m and c
    y_pred = m * x + c
    # Compute mean squared error
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    print(f"Epoch {epoch+1}: MSE = {error:.4f}")
    # Update m and c using gradient descent
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plot the line of best fit after the final epoch
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m * x + c, color='red', label='Best Fit Line')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.legend()
plt.title('Best Fit Line After 10 Epochs')
plt.show()

# Predict the price for an office size of 100 sq. ft.
predicted_price = m * 100 + c
print(f"Predicted price for 100 sq. ft. office: {predicted_price:.2f}")
