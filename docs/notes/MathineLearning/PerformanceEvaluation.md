# Performance Evaluation

After we trained a model, we should evaluate the model to find out how good it is.

## Training Loss

There are two dimension for us to consider when designing  our model: **Training Error** VS **Model complexity**.

<img src="./Trainning error vs model complexity.png" width="60%">

- Increasing model complexity $\to$ Lower training error.

- Thinking: **Is this a desirable thing?**

Because the params of model are only fit for **training data**, so unless the training data includes everything we can see for the task, **small training error not equals to  Good predictions**.

## Generalization Error

Ideally, we want to estimate loss over all possible data $(x, y)$, then the train loss should be the true error. But apparently it is impossible. Our training loos assume that all datas have the same possibility to appear, actually it is not.

Assume we know the possibility of the appearance of each data point, the training could be:

$$
\text{GeneralizationError} = \arg\min_{\mathbf{w}}\sum_{(x, y)}(y - f_{\mathbf{w}}(x))^2 \cdot p(x, y)
$$

<img src="./generalization error vs model complexity.png" width="60%">

As the increasing of model complexity, the generalization error would go through down first and up at last.

### Approximating Generalization Error

- Not possible to look at all $(x, y)$ to compute generalization error.
- Instead, approximate generalization error by looking at data which is not in the training set.
- Split the available data into Train/Test and **never, ever, train your models on test data!**
- Test Set becomes a Proxy for “everything one might see”

## Test Error

The format of test error is similar with training error:

$$
\text{TestError} = \frac{1}{|\mathcal{D}_{\text{Test}}|}\sum_{\forall (x, y) \in \mathcal{D}_{\text{Test}}} L(y, f_{\mathbf{w^*}}(x))
$$

The $\mathbf{w}^*$ is the outcome of training on training data set.

### Training and test

- Split data to hold out “test” data.
- Train the model (estimate the params) using only the training data.
- Test Set becomes a Proxy for “everything one might see”

<img src="./errors.png" width="80%">

### Overfitting

## Bias and Variance

### Source of Error

- Noise
- Bias
- Variance

$$
\text{ExpectedPredictionError}(x) = \text{Noise} + \text{Bias}^2 + \text{Variance}
$$

$$
\text{MSE}(x) = \sigma^2 + \text{Bias}^2 + \text{Variance}
$$

### Noise

- Data is inherently noisy.
- Irreducible error.
- Regression Model: The $\varepsilon$ in $y = f(x) + \varepsilon$.
- Even if we estimate f exactly, there will still be some error (noise).

### Setup

- We create N different training sets by sampling.
- Each training set produces estimated model params.
- Use the “average” predictions of all the N estimated models, denoted by $f_{\bar{\mathbf{w}}}$.
- The “average” fit is akin to an expected fit.

### Bias

$$
\text{Bias}(x) = f_{\text{true}}(x) - f_{\bar{\mathbf{w}}}(x)
$$

- Is our approach flexible enough to capture $f_{\text{true}}$.
- Bias is high when the hypothesis class is unable to capture $f_{\text{true}}$.

<img src="./bias.png" width="60%">

- Constant Function has Low Model Complexity.
- Usually not capable of capturing the true relationship.
- Low complexity models lead to High Bias.

### Variance

### Why 3 sources of error

First of all, consider the definition of EPE:

$$
\begin{align}
\text{EPE} &= E_{\text{train}}\left[\text{GeneralizationError of }\mathbf{w}^*(\text{train})\right]\\
&= E_{\text{train}}
\end{align}
$$
