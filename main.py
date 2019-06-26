from numpy import genfromtxt

# error_function =>  += 1/N*(y - (m*x+b))**2

# calculate the error with mean squared method
def calculate_error(points,m,b):
    error = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        error += (y - (m*x+b))**2
    return error/len(points)

def step_gradient(points,m,b,learning_rate):
    gradient_m = 0
    gradient_b = 0
    for i in range(len(points)) :
        x = points[i,0]
        y = points[i,1]
        gradient_m += -2/len(points)*x*(y-(m*x + b)) # partial derivative respect to m
        gradient_b +=  -2/len(points)*(y-(m*x + b)) # partial derivative respect to b
    new_m = m - (learning_rate*gradient_m) # update m
    new_b = b - (learning_rate*gradient_b) # update b
    return new_m,new_b


def train(points,initial_m,initial_b,learning_rate,iterations):
    predicted_m = initial_m
    predicted_b = initial_b
    for i in range(0,iterations):
        predicted_m , predicted_b = step_gradient(points,predicted_m,predicted_b,learning_rate)

    return predicted_m,predicted_b

path = 'data.csv'
points = genfromtxt(path, delimiter=",")
learning_rate = 0.000001
initial_m = 0
initial_b = 0
iterations = 1000
predicted_m,predicted_b = train(points,initial_m,initial_b,learning_rate,iterations)
print('after training our model predicted that m = {m} and b= {b} '.format(m=predicted_m,b=predicted_b))
initial_error = calculate_error(points,initial_m,initial_b)
error = calculate_error(points,predicted_m,predicted_b)
print('the error for this model is ',error,' instead of ',initial_error)
