# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# points = pd.read_csv('data.csv')

def y_function(x):
    return x**2

def y_derative(x):
    return 2*x

x = np.arange(-100, 100, 0.1)
y = y_function(x)

current_position = (80, y_function(80))
learning_rate = 0.005

for it in range(1000):
    new_x = current_position[0] - learning_rate * (y_derative(current_position[0]))
    new_y = y_function(new_x)
    current_position = (new_x, new_y)
    


    plt.plot(x,y)
    plt.scatter(current_position[0], current_position[1], color = "red")
    plt.pause(0.001)
    plt.clf()