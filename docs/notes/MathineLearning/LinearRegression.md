# Linear Regression

Linear regressing is a base model of mechine learning.

## What is machine learning?

- The process of **turning data into a model**.
- Perform inference to make **predictions**.

Below is a prediction process expressed in math notation:

$$
f_{\theta}(x) = y \in \mathbb{R} \qquad f_{\theta}(x) = y \in \mathbb{Z} \qquad f_{\theta}(x) = y \text{ is a complex object}
$$

The input $x$ is usually a vector, and $y$ could be many formats depend on the type of task. When the output $y \in \mathbb{R}$, the 
task is called **regression task**. When the output $y$ is in a distribution area, the tasks will be called **classification task**. When the output is a complex object, the task will be called **structured prediction**.

## Linear regression framework

<img src="./frame work.png" width="60%">

### Design decisions

- **Hypothesis class**: Which predictor are possible?
- **Loss function**: How good is a predictor?
- **Optimization algorithm**: How do we compute the best predictor?

## Hypothesis class

- **Model**: Our **assumption** about how the world works.

- Expected relationship between $x$ and $y$.

- **Regression Model**:

$$
y = f(x) + \varepsilon, \qquad E(\varepsilon) = 0
$$

  The $y$ denotes the target or **ground-truth** to be predicted. The $\hat{y} = f(x)$ is the prediction or estimation of our model. The $\varepsilon = y - \hat{y}$ denotes the error or residual.

### Linear Regression Model

#### Vector notation

- **Weight vector**: $\mathbf{w} = [w_1, w_2]$
- **Feature vector**: $[1, x]$
- **Feature extractor**: $\phi(x) = [1, x]$

$$
f_{\mathbf{w}}(x) = \mathbf{w} \cdot \phi(x)
$$

#### Hypothesis class

$$
\mathcal{F} = \{f_{\mathbf{w}}: \mathbf{w} \in \mathbb{R}^2\}
$$

### Polynomial Regression Model

In some complex situation, we can change the feature vector to construct polynomial regression.

$$
\phi(x) = [\phi_1, \phi_2, \phi_3, \cdots]
$$

### Linear in where?

- Linear in $\mathbf{w}$
- Linear in $\phi$
- Non-linear in $x$

## Loss Function

Measure how good a model is.

### MSE: Mean squared error

<img src="./loss function.png" width="40%">

For the squared loss:

$$
\text{Loss}(x, y, \mathbf{w}) = (f_{\mathbf{w}}(x) - y)^2
$$

Calculate the mean of all datas in $\mathcal{D}_{\text{train}}$, we can get the MSE:

$$
\text{TrainLoss}(\mathbf{w}) = \frac{1}{|\mathcal{D}_{\text{train}}|}\sum_{(x, y) \in \mathcal{D}_{\text{train}}}\text{Loss}(x, y, \mathbf{w})
$$

### Matrix Form

$$
\begin{align}
	\text{TrainLoss}(\mathbf{w}) &= \frac{1}{|\mathcal{D}_{\text{train}}|}\sum_{(x, y) \in \mathcal{D}_{\text{train}}}(\mathbf{w} \cdot \phi(x) - y)^2\\
	&= \frac{1}{|\mathcal{D}_{\text{train}}|}(\mathbf{y} - \Phi \mathbf{w})^{\mathbb{T}}(\mathbf{y} - \Phi\mathbf{w})
\end{align}
$$

<img src="./matrix form.png" width="60%">

### Why mean squared loss?

Consider the relationship between $y$ and $x$:

$$
y = f_{\mathbf{w}}(x) + \varepsilon \qquad E(\varepsilon) = 0
$$

Assuming that $\varepsilon \sim N(\mu = 0, \sigma^2)$, which means $y \sim N(\phi(x)\cdot \mathbf{w}, \sigma^2)$.

Maximum likelihood estimate of params:

$$
\arg\max_{\mathbf{w}} p(\mathbf{y|_{\mathbf{x}, \mathbf{w}}}) = \arg\max_{\mathbf{w}}\Pi_{(x, y) \in \mathcal{D}_{\text{train}}} p(y|_{x, w}) = \arg\max_{\mathbf{w}}\Pi_{(x, y)\in \mathcal{D}_{\text{train}}} \ln p(y|_{x, w})
$$

Maximize log-likelihood:

$$
\begin{align}
\arg\max_{\mathbf{w}}\Pi_{(x, y)\in \mathcal{D}_{\text{train}}}\ln p(y|_{x, w}) &= \arg\max_{\mathbf{w}}\Pi_{(x, y)\in \mathcal{D}_{\text{train}}}\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2\sigma^2}(y - \phi(x)\mathbf{w})^2}\\
&=\arg\min_{\mathbf{w}}\sum_{(x, y) \in \mathcal{D}_{\text{train}}}(y - \phi(x)\cdot \mathbf{w})^2
\end{align}
$$

## Optimization algorithm

### Gradient deccent

- Initialize $\mathbf{w} = [0, 0, \cdots, 0]$

- For every epoch:

$$
\mathbf{w} \gets \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w})
$$

  In the up equation, arg $\eta$ is called **learning rate**.

### When to stop/convergence

- For Convex functions, convergence happens when gradient = 0.

- In practice, stop when

$$
\lVert \nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w})\rVert \leqslant \varepsilon
$$

- Or when a maximum number if iterations is done.

### Example: Linear Regression

For linear regression task, we can conclude target loss as:

$$
\arg\min_{\mathbf{w}}\text{TrainLoss}(\mathbf{w}) =\arg\min_{\mathbf{w}}(\mathbf{y}\cdot \Phi \mathbf{w})^{\mathbb{T}}(\mathbf{y}\cdot \Phi \mathbf{w})
$$

Try to solve the gradient of RSS:

$$
\nabla_{\mathbf{w}}\text{RSS}(\mathbf{w}) = \nabla_{\mathbf{w}}(\mathbf{y} - \Phi \mathbf{w})^{\mathbb{T}}(\mathbf{y} - \Phi \mathbf{w}) = -2\Phi^\mathbb{T}(\mathbf{y} - \Phi \mathbf{w})
$$

So that the gradient of train loss would be:

$$
\nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w}) = \frac{-2}{|\mathcal{D}_{\text{train}}|}\Phi^{\mathbb{T}}(\mathbf{y} - \Phi\mathbf{w})
$$

#### Approach 1: Set the gradient equals to zero

Set $\nabla_{\mathbf{w}}\text{TrainLoss}$ to zero:

$$
\nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w}) = 0 \implies \mathbf{w^*} = (\Phi^{\mathbb{T}}\Phi)^{-1}\Phi^{\mathbb{T}}\mathbf{y}
$$

Which is difficult to calculate.

#### Approach 2: Gradient descent

<img src="./gradient descent.png" width="80%">

### Stochastic gradient descent

There is a problem about gradient descent. It is: expensive when have a lots of data!!! So we can use stochastic gradient descent:

- Initialize $\mathbf{w} = [0, 0, \cdots, 0]$

- For each epoch:

  - For $(x, y) \in \mathcal{D}_{\text{train}}$ (Stochasticly choose a batch of data to calculate)
  
$$
\mathbf{w} \gets \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}}\text{Loss}(x, y, \mathbf{w})
$$

## Appendix: Inequalities in linear regression

Some times, the prediction of model may be unfair. We need to adjust our loss policy to avoid this phenomenon.

### Average Loss

<img src="./average loss.png" width="80%">

This policy is the common policy to calculate the loss.

### Group DRO: Group distributionally robust optimization

There are several steps to adjust our loss calculation policy.

- Calculate per-group loss
- Get the maximum group loss
- Training via gradient descent

#### Per-group loss

For different group, we can calculate their average loss separately:

<img src="./per-group loss.png" width="70%">

#### Maximum group loss

After calculate per-group loss seprately, we can get the maximum group loss:

<img src="./maximum gtoup loss.png" width="80%">

#### Train via gradient descent

Using gradient descent to find the best $\mathbf{w}$.

$$
\text{TrainLoss}_{\max}(\mathbf{w}) = \max_g \text{TrainLoss}_g(\mathbf{w})
$$

$$
\nabla \text{TrainLoss}_\max(\mathbf{w}) = \nabla \text{TrainLoss}_{g^*}(\mathbf{w})\\
\text{where}\, g^* = \arg\max_g\text{TrainLoss}_g(\mathbf{w})
$$

