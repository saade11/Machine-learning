import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')



def gradient_descent(m_now, b_now, points,l):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i, 0]  # First column as x
        y = points.iloc[i, 1]  # Second column as y
        
        # Calculate the gradients
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m_updated = m_now - (l * m_gradient)
    b_updated = b_now - (l * b_gradient)
    
    return m_updated, b_updated


m = 0
b = 0
l = 0.0001
epochs = 200

for i in range(epochs):
    if i % 50 ==0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m,b,data,l)
    

print(m,b)

plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color="black")  # Scatter plot
plt.plot(list(range(20,80)), [m*x + b for x in range(20,80)], color ="red")    # Line plot (optional)
plt.show()
# # Add labels and title
# plt.xlabel('X Coordinates')
# plt.ylabel('Y Coordinates')
# plt.title('X vs Y Coordinates')

# # Show the plot
# plt.show()
