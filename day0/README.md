# Classification with Logistic Regression

The purpose of this exercise to gain an appreciation of what happens under the hood in an autograd framework like PyTorch.

First we draw samples from two different 2D gaussian distributions.
The goal is to create a linear regression model, that classifies each sample as belonging to distribution 0 or 1.

In `0_logistic_regression.ipynb` there are placeholders and hints marked with pounds (`#`) for where you can add your code.
In `0_logistic_regression_solution.ipynb` you can find a reference solution.

## Sigmoid
Compute the sigmoid function and plot it in the given range.

## Logistic Regression

Implement the logistic regression prediction. To implement the bias, you can concatenate a `1.0` to the input data.

## Loss

Compute the MSE loss and calculate the gradient of the loss with respect to the weights.
This is the part that PyTorch will do for you in the future (Unless you are doing something *exotic*/interesting).

## Train

Train the model by computing the gradient and using it to update the model weights.

# Acknowledgement

This exercise was adapted from the Workshop "Machine Learning with Neural Networks" at the GridKa Summer School 2019. See: [https://github.com/Markus-Goetz/gks-2019](https://github.com/Markus-Goetz/gks-2019).
