  # Machine-learning

  Different type of machine learning:

	1) Suppervised Learning
	2) Unsupervised Learning
	3) Reinforcement Learning

  Supervised Learning: Supervised learning is the most basic and widely used type of machine learning. In supervised learning, a model is trained on a dataset where the correct output or “label” is already provided for each input. For example, imagine you have a dataset of pictures of cats and dogs. The labels for the dataset would be “cat” and “dog.” The model is then able to use this information to make predictions about new pictures of cats and dogs it has never seen before.
## **Supervised Learning algorithms**

There are several popular supervised learning algorithms that are widely used in the field of machine learning. Some of the most well-known and commonly used algorithms include:

1. Linear Regression
2. Logistic Regression
3. Decision Trees
4. Random Forest
5. Support Vector Machines (SVM)
6. k-Nearest Neighbours (kNN)

 Unsupervised Learning: Unsupervised learning, on the other hand, is when the model is given a dataset without any labels or output. The model must then find patterns and structure within the data on its own. A common example of unsupervised learning is clustering, where a model groups similar data points together. Imagine you have a dataset of customer data. The model would group customers based on similar characteristics such as age, location, and spending habits.
## Unsupervised Learning algorithms

There are several popular unsupervised learning algorithms:

1. K-Means
2. Hierarchical Clustering
3. PCA(Principal Component Analysis)
4. t-SNE ( t-Distributed Stochastic Neighbour Embedding)

Reinforcement Learning: Reinforcement learning is a bit different from supervised and unsupervised learning. In reinforcement learning, the model learns from the consequences of its actions. The model receives feedback on its performance, and uses that information to adjust its actions and improve its performance over time. A classic example of reinforcement learning is training a model to play a game like chess or Go. The model receives feedback on its performance in the form of win or loss, and then adjusts its strategy to improve its chances of winning.

## Reinforcement Learning algorithms

There are several popular reinforcement learning algorithms:

1. Q-Learning
2. SARSA
3. DQN
4. A3C

### Linear regression:
___
Linear regression is a type of [supervised machine learning](https://www.geeksforgeeks.org/supervised-machine-learning/) algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data.

When there is only one independent feature, it is known as [Simple Linear Regression](https://www.geeksforgeeks.org/simple-linear-regression-using-r/), and when there are more than one feature, it is known as [Multiple Linear Regression](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/).

Similarly, when there is only one dependent variable, it is considered [Univariate Linear Regression](https://www.geeksforgeeks.org/univariate-linear-regression-in-python/), while when there are more than one dependent variables, it is known as [Multivariate Regression](https://www.geeksforgeeks.org/multivariate-regression/).

### Why Linear Regression is Important?
____
The interpretability of linear regression is a notable strength. The model’s equation provides clear coefficients that elucidate the impact of each independent variable on the dependent variable, facilitating a deeper understanding of the underlying dynamics. Its simplicity is a virtue, as linear regression is transparent, easy to implement, and serves as a foundational concept for more complex algorithms.

Linear regression is not merely a predictive tool; it forms the basis for various advanced models. Techniques like regularization and support vector machines draw inspiration from linear regression, expanding its utility. Additionally, linear regression is a cornerstone in assumption testing, enabling researchers to validate key assumptions about the data.


In linear regression we are somehow predicting the output y according to a parameter x if we are in the case of a single valued linear regression.

So put a line that best fits the cluster of points we would have to play with the coefficients.

$y=ax+b$  
where here $a$ is the slope of the line and $b$ is the y-intercept. so we need to predict the best a and b for our model to give the best results. 
And how do we do that? using **Cost functions** 

A Cost function is a function that will compute the difference between the actual output and the predicted output. Usually with single valued linear regression we would use the mean square error as our Cost Function which can be denoted by the following:

$$
J(\theta_0,\theta_1) = (1/2m) \sum_{i=1}^{i=m}(h_\theta(x^i)-y^i)^2 
$$
Our goal is to minimize as much as possible the value of the function above.
How do we do that? From basic Calculus we know that to get the minimal point of a function we need to compute the derivative of this function and set equal to 0, but here we do that by computing the partial derivative for the respective variable and choosing values at randoms for $\theta_1$ and $\theta_2$ after that we update them respectively using a smart algorithm called gradient Descent.

but first lets us find the partial derivatives for the above:

Assume our hypothesis function is $h_\theta(x) = \theta_1(x) + \theta_0$ .    

$$
\partial/\partial\theta_j (J(\theta_0,\theta_1)) = \partial/\partial\theta_j ((1/2m) \sum_{i=1}^m(\theta_1x_i+\theta_0 -y_i )^2
$$

with respect to $\theta_0$:
$$
\partial/\partial\theta_j (J(\theta_0,\theta_1)) = 1/m (\sum_{i=1}^m h_\theta(x_i)  - y_i) 
$$

with respect to $\theta_1$ :
$$
\partial/\partial\theta_j (J(\theta_0,\theta_1)) = 1/m (\sum_{i=1}^m h_\theta(x_i)  - y_i) x_i
$$

and now we would apply the gradient descent algorithm:

![image](https://github.com/user-attachments/assets/1f46d0fe-b5d7-4443-9446-9fe55ed29202)

Note: effectively here we are computing the gradient (derivative) of the cost function and finding its minimum and when this gradient is >0 what does it mean? we are moving in the direction of the steepest assent which what we are trying to do the complete opposite, thus this is why we put the "-" thus making $\theta$ smaller. Assume now the other case around, the gradient is < 0 if we follow the direction of the gradient we will go up and make $\theta$ smaller and again the "-" comes to rescue, thus in this case it will become a "+" making $\theta$ bigger and moving towards a local minima.   

Now I will be coding this in python.


`import pandas as pd`

`import matplotlib.pyplot as plt`

  

`data = pd.read_csv('data.csv')`

  
  
  

`def gradient_descent(m_now, b_now, points,l):`

    `m_gradient = 0`

    `b_gradient = 0`

    `n = len(points)`

    `for i in range(n):`

        `x = points.iloc[i, 0]  # First column as x`

        `y = points.iloc[i, 1]  # Second column as y`

        `# Calculate the gradients`

        `m_gradient += -(2/n) * x * (y - (m_now * x + b_now))`

        `b_gradient += -(2/n) * (y - (m_now * x + b_now))`

    `m_updated = m_now - (l * m_gradient)`

    `b_updated = b_now - (l * b_gradient)`

    `return m_updated, b_updated`

  
  

`m = 0`

`b = 0`

`l = 0.0001`

`epochs = 200`

  

`for i in range(epochs):`

    `if i % 50 ==0:`

        `print(f"Epoch: {i}")`

    `m, b = gradient_descent(m,b,data,l)`

  

`print(m,b)`

  

`plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color="black")  # Scatter plot`

`plt.plot(list(range(20,80)), [m*x + b for x in range(20,80)], color ="red")    # Line plot (optional)`

`plt.show()`


![image](https://github.com/user-attachments/assets/265d7f0d-4955-465e-ab0d-7cc80bb793a4)


# K-Nearest Neighbors Classification
____


The ****K-Nearest Neighbors (KNN) algorithm**** is a supervised machine learning method employed to tackle classification and regression problems. Evelyn Fix and Joseph Hodges developed this algorithm in 1951, which was subsequently expanded by Thomas Cover. The article explores the fundamentals, workings, and implementation of the KNN algorithm.

## What is the K-Nearest Neighbors Algorithm?

KNN is one of the most basic yet essential classification algorithms in machine learning. It belongs to the [supervised learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning) domain and finds intense application in pattern recognition, [data mining](https://www.geeksforgeeks.org/data-mining), and intrusion detection.

It is widely disposable in real-life scenarios since it is non-parametric, meaning it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a [Gaussian distribution](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-3-normal-distribution) of the given data). We are given some prior data (also called training data), which classifies coordinates into groups identified by an attribute.


![image](https://github.com/user-attachments/assets/68db6e36-65a9-4f28-a823-921da08f0c0e)


## Why do we need a KNN algorithm?

(K-NN) algorithm is a versatile and widely used machine learning algorithm that is primarily used for its simplicity and ease of implementation. It does not require any assumptions about the underlying data distribution. It can also handle both numerical and categorical data, making it a flexible choice for various types of datasets in classification and regression tasks. It is a non-parametric method that makes predictions based on the similarity of data points in a given dataset. K-NN is less sensitive to outliers compared to other algorithms.

The K-NN algorithm works by finding the K nearest neighbors to a given data point based on a distance metric, such as Euclidean distance. The class or value of the data point is then determined by the majority vote or average of the K neighbors. This approach allows the algorithm to adapt to different patterns and make predictions based on the local structure of the data.

## Distance Metrics Used in KNN Algorithm

As we know that the KNN algorithm helps us identify the nearest points or the groups for a query point. But to determine the closest groups or the nearest points for a query point we need some metric. For this purpose, we use below distance metrics:


$$
distance(x,X_i) = \sqrt{\sum_{i=0}^n(x_j-X_i{_{_j}})^2 } 
$$

![image](https://github.com/user-attachments/assets/74faea30-7a47-4f77-a523-07487fba74db)

### Step 1: Selecting the optimal value of K

- K represents the number of nearest neighbors that needs to be considered while making prediction.

### Step 2: Calculating distance

- To measure the similarity between target and training data points, Euclidean distance is used. Distance is calculated between each of the data points in the dataset and target point.

### Step 3: Finding Nearest Neighbors

- The k data points with the smallest distances to the target point are the nearest neighbors.

### Step 4: Voting for Classification or Taking Average for Regression

- In the classification problem, the class labels of K-nearest neighbors are determined by performing majority voting. The class with the most occurrences among the neighbors becomes the predicted class for the target data point.
- In the regression problem, the class label is calculated by taking average of the target values of K nearest neighbors. The calculated average value becomes the predicted output for the target data point.

The ‘k’ in KNN stands for the number of nearest neighbors you want to consider for making your prediction. This choice is vital because a small ‘k’ makes the model sensitive to noise, while a large ‘k’ might smooth out the prediction too much, possibly leading to misclassification or inaccurate predictions. There’s no one-size-fits-all ‘k’; it often requires trial and error or techniques like cross-validation to find the best ‘k’ for your specific dataset.

[knn](https://medium.com/@sahin.samia/demystifying-k-neighbors-classifier-knn-theory-and-python-implementation-from-scratch-f5e76d6f2d48) article (need to revise)

the hard thing in this approach we must correctly choose "K" and it is always better to choose an odd K for the voting part.

# Gradient Descent vs Stochastic Gradient Descent
_____

Lets first try to understand why do we need any alternative for Gradient Descent, is it not good enough?

Lets illustrate the problem through an example below:

| Area | Bedrooms | price  |
| ---- | -------- | ------ |
| 2600 | 3        | 550000 |
| 3000 | 4        | 565000 |
| 3200 | 3        | 610000 |
| 3600 | 3        | 595000 |
| 4100 | 6        | 810000 |

If we approach this problem using the standard way, we would have first a hypothesis function $h_\theta (x)$ and then we would compute the error by comparing with the actual prices. So we would compute the error of each row until we reach the last row (this is known as an epoch) **finally** now we adjust the weights ie ($\theta_0$ ,$\theta_1$ and $\theta_2$), $\theta_2$ here is the bias term. So in this example we have 5 rows and 2 features where are (Area and Bedrooms). assume we have initialized $\theta_0$ ,$\theta_1$ and $\theta_2$ to all be 1. We have to calculate here the error for 5 rows but assume we have 20 zillions rows this would not be computationally feasible keeping in mind also we have to evaluate the derivatives  which would require double because we have 2 features. Thus we can see the need for an alternative which is stochastic gradient where we will proceed as follows: we will sample(pick at random a row from the dataset) evaluate $h_\theta (x)$ get the error and directly updating the weight, thus we will evaluate much faster and concerning the convergence we will also converge much faster with no remarkable cost at the most optimal weight.





