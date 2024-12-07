# Regularization

## Ridge Regression

Regulating overfitting when using many features.

## Overfitting of polynomial regression

Consider the overfitting phenomenon of polynomial regression. Often, overfitting associated with very large estimated parameters $\hat{\mathbf{w}}$. To avoid model chossing large $\hat{\mathbf{w}}$ when training, we can add term loss function to prefer somall coefficients.

### Desired Total Cost format

Want to balance:

- How well function fits data
- Magnitude of coefficients

$$
\text{TotalCost} = \text{MeasureOfFit} + \text{MeasureOfMagnitudeOfCoefficients}
$$

### Measure of fit to training data

Consider the loss function we learned before:

$$
\text{RSS}(\mathbf{w}) = (\mathbf{y} - \Phi \mathbf{w})^{\mathbb{T}}(\mathbf{y} - \Phi \mathbf{w})
$$

This loss can measure how well the function fits training data.

### Measure of Magnitude of Regression Coefficients

We can use $L_2$ norm to measure the coefficient:

$$
\lVert \mathbf{w} \rVert^2_2 = \sum_{i = 1}^{D}w_i^2
$$

### Consider resulting objective

$$
\text{RSS}(\mathbf{w}) + \lambda \lVert \mathbf{w} \rVert^2_2
$$

A.K.A $L_2$ regularization. $\lambda$ is a hyper param that controls the model complexity.

- Large $\lambda$: high bias, low variance
- Low $\lambda$: low bias, high variance

## Optimization of ridge regression

Consider the optimization method of our model.

### Gradient of ridge regression cost

$$
\nabla \left[\text{RSS}(\mathbf{w}) + \lambda \lVert \mathbf{w} \rVert^2_2\right] = \nabla \left[(\mathbf{y} - \Phi \mathbf{w})^{\mathbb{T}}(\mathbf{y} - \Phi \mathbf{w})\right] + \lambda \nabla (\mathbf{w}^{\mathbb{T}}\mathbf{w}) = -2\Phi^{\mathbb{T}}(y - \Phi \mathbf{w}) + 2\lambda \mathbf{w}
$$

### Ridge closed-form solution
