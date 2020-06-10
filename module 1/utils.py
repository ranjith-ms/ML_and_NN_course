#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[2]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[1]:


def gradientDescent(x, y, theta, num_iters,lr):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `lr`.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m , n+1).
    
    y : arra_like
        Value at given features. A vector of shape (m,1 ).
    
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (1,n+1 ).
        
    lr : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (1,n+1 ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(num_iters)
    
    for i in range(num_iters):
        h = np.matmul(x, theta.T)
        error = h-y
        for j in range(2):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((lr/ len(x)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(x, y, theta)
        
    return theta, cost


# In[2]:


def computeCost(x, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m , n+1) <Here n is 1 and we added one more column of ones>, where m is the number of examples,
        and n is the number of features. <Hence the dimension is (46,2)
    
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, 1).
    
    theta : array_like
        The parameters for the regression function. This is a vector of 
        shape (1,n+1 ).
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. 
    You should set J to the cost.
    """
    
    # initialize some useful values
    m =46  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
   
    h = np.matmul(x, theta.T)
    
    J = (1/(2 * m)) * np.sum(np.square(h - y))
    
   
    return J

