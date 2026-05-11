# Module 05 — Deep Learning & MLOps

> **Run:**
> ```bash
> cd src/05-deep-learning
> python nn_numpy.py        # 2-layer NN from scratch — forward + backprop
> python optimizers.py      # SGD, Momentum, RMSProp, Adam from scratch
> python mlflow_demo.py     # experiment tracking, model registry
> python monitoring.py      # data drift detection
> ```

```bash
cd src/05-deep-learning && python nn_numpy.py
cd src/05-deep-learning && python optimizers.py
cd src/05-deep-learning && python mlflow_demo.py
cd src/05-deep-learning && python monitoring.py
```

---

## Prerequisites & Overview

**Prerequisites:** Modules 01–02 (matrix math, partial derivatives, chain rule, gradient descent). NumPy. No PyTorch required — all scripts run in pure NumPy.
**Estimated time:** 10–15 hours (the backprop derivation alone deserves 2+ hours)

### Why This Module Matters
Every modern AI system — LLMs, diffusion models, vision transformers — is a deep neural network trained with backpropagation and an adaptive optimizer. Understanding these from scratch means you can debug training instability, choose the right optimizer, and explain gradient flow in interviews. The MLOps half (MLflow + drift detection) covers what happens after training — equally critical for production roles.

### Module Map

| Section | Core Concept | Practical Payoff |
|---------|-------------|-----------------|
| Neural network math | Forward pass, backprop derivation | Debug NaNs, understand gradient flow |
| Initialisation | Xavier, He, symmetry breaking | Fix vanishing/exploding gradients |
| Activations | ReLU, GELU, sigmoid, softmax | Choose the right activation per layer |
| Optimizers (6) | SGD → Adam → AdamW | Know when Adam fails, when SGD wins |
| LR Scheduling | Warmup, cosine, plateau | Match scheduler to training dynamics |
| Regularisation | Dropout, BatchNorm, LayerNorm | Control overfitting without losing capacity |
| MLflow | Experiment tracking, model registry | Reproduce any experiment from a run ID |
| Drift Detection | KS, PSI, MMD, JS | Catch distribution shift before accuracy drops |

### Before You Start
- Implement a dot product in NumPy from scratch (Module 01)
- Know what a partial derivative is (Module 01)
- Understand gradient descent update rule: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$ (Module 01-02)
- Install MLflow: `pip install mlflow` (optional — demo degrades gracefully without it)

### Intuition First: What Is Backpropagation?
Backpropagation is **the chain rule applied systematically from the output layer backward to the input layer**. Every layer computes: "by how much does changing my input change the loss?" That number is the gradient. Layer by layer, from loss back to weights, gradients tell the optimizer which direction to step.

---

## 1. Neural Networks — Mathematical Foundation

A feedforward neural network with $L$ layers maps input $\mathbf{x} \in \mathbb{R}^{n_0}$ to output $\hat{\mathbf{y}} \in \mathbb{R}^{n_L}$ through a composition of affine transformations and nonlinearities:

$$\mathbf{a}^{(l)} = g^{(l)}\!\left(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1, \ldots, L$$

where $\mathbf{a}^{(0)} = \mathbf{x}$, $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$, $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$, and $g^{(l)}$ is the activation function.

**Pre-activation (logit):**

$$\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

$$\mathbf{a}^{(l)} = g^{(l)}(\mathbf{z}^{(l)})$$

---

## 2. Activation Functions

| Function | Formula | Derivative | Dead neurons? | Use case |
|----------|---------|-----------|---------------|----------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | No | Output (binary) |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ | No | RNN hidden |
| ReLU | $\max(0, z)$ | $\mathbf{1}[z>0]$ | Yes (z<0) | Default hidden |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha\approx0.01$ | $\mathbf{1}[z>0] + \alpha\mathbf{1}[z\leq0]$ | Rarely | Sparse features |
| ELU | $z$ if $z>0$, else $\alpha(e^z-1)$ | $\mathbf{1}[z>0] + g(z)+\alpha$ if $z\leq0$ | No | Negative saturation |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$ | N/A | Output (multiclass) |

**ReLU dead neuron problem:** If $z^{(l)} < 0$ for all training examples, the gradient is 0 and the neuron never updates. Mitigated by small initialization, proper LR, or Leaky ReLU.

**Numerical stability of softmax:** Compute $z_k - \max_j z_j$ before exponentiating to prevent overflow.

---

## 3. Backpropagation — Full Derivation

### Loss function

Binary cross-entropy (BCE) for output $\hat{y} = \sigma(z^{(L)})$:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

Categorical cross-entropy for softmax output:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log\hat{y}_{ik}$$

### Chain rule — layer-by-layer

Define the upstream gradient at layer $l$:

$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

**Output layer** (sigmoid + BCE):

$$\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y} \quad \in \mathbb{R}^{n_L \times m}$$

This is the "miracle" of cross-entropy + sigmoid/softmax: the Jacobian of the activation cancels with the derivative of the loss, leaving a clean residual.

**Hidden layer** (chain rule backward through $g$):

$$\delta^{(l)} = \left(W^{(l+1)T}\delta^{(l+1)}\right) \odot g'^{(l)}(\mathbf{z}^{(l)})$$

where $\odot$ is elementwise multiplication and $g'^{(l)}$ is the elementwise derivative of the activation.

**Parameter gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m}\delta^{(l)}\mathbf{a}^{(l-1)T}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \frac{1}{m}\sum_{i=1}^{m}\delta^{(l)}_i = \frac{1}{m}\delta^{(l)}\mathbf{1}$$

**Update rule** (gradient descent):

$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$

### Computational complexity

Forward pass: $O\!\left(\sum_{l=1}^{L} n_l \cdot n_{l-1} \cdot m\right)$

Backward pass: same asymptotic cost as forward (two matrix multiplies per layer). Total: $\approx 2\times$ forward cost.

---

## 4. Weight Initialization

Bad initialization → vanishing or exploding gradients during backprop.

**Zero initialization:** All neurons in each layer compute the same gradient → symmetry breaking fails → equivalent to a single neuron per layer.

**Random initialization (too large):** Activations saturate (sigmoid/tanh) → gradients → 0 (vanishing gradient).

| Scheme | Formula | Use with |
|--------|---------|----------|
| Xavier / Glorot | $\mathcal{N}\!\left(0, \frac{2}{n_{in}+n_{out}}\right)$ | Sigmoid, Tanh |
| He (Kaiming) | $\mathcal{N}\!\left(0, \frac{2}{n_{in}}\right)$ | ReLU, Leaky ReLU |
| Lecun | $\mathcal{N}\!\left(0, \frac{1}{n_{in}}\right)$ | SELU |

**Derivation sketch (Xavier):** To keep variance stable across layers, we want $\text{Var}[\mathbf{a}^{(l)}] = \text{Var}[\mathbf{a}^{(l-1)}]$. For a linear layer with $n_{in}$ inputs:

$$\text{Var}[z] = n_{in} \cdot \text{Var}[w] \cdot \text{Var}[a]$$

Setting $\text{Var}[z] = \text{Var}[a]$ gives $\text{Var}[w] = 1/n_{in}$. Averaging the forward and backward constraints gives $1/(n_{in}+n_{out})/2 \cdot 2 = 2/(n_{in}+n_{out})$.

**He initialization adds factor of 2** because ReLU zeros out half the neurons, halving the effective variance:

$$\text{Var}[w] = \frac{2}{n_{in}}$$

---

## 5. Optimization Algorithms

All algorithms maintain a parameter vector $\theta$ and minimize $\mathcal{L}(\theta)$ using gradient information $g_t = \nabla_\theta \mathcal{L}$.

### SGD (Stochastic Gradient Descent)

$$\theta_{t+1} = \theta_t - \eta g_t$$

Batch SGD uses all data; mini-batch SGD uses a random subset $B \ll m$. Mini-batch is the default — balances gradient noise (good for escaping local minima) with computation efficiency.

### SGD + Momentum (Heavy Ball)

Accumulates an exponentially decaying moving average of past gradients:

$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \eta g_t$$
$$\theta_{t+1} = \theta_t + \mathbf{v}_{t+1}$$

$\mu \in [0.9, 0.99]$ typically. Effective learning rate for a constant gradient: $\eta / (1-\mu)$.

**Nesterov momentum** computes gradient at the "lookahead" position:

$$\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \eta \nabla_\theta \mathcal{L}(\theta_t + \mu\mathbf{v}_t)$$

### RMSProp

Divides per-parameter LR by a running estimate of the root-mean-square gradient:

$$s_{t+1} = \beta s_t + (1-\beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1} + \epsilon}} g_t$$

$\beta = 0.99$, $\epsilon = 10^{-8}$. Adapts LR per parameter — large gradient parameters get smaller LR; small gradient parameters get larger LR. Designed for non-stationary objectives (RNNs).

### Adam (Adaptive Moment Estimation)

Combines momentum (1st moment) with RMSProp (2nd moment):

$$m_{t+1} = \beta_1 m_t + (1-\beta_1)g_t \quad \text{(1st moment)}$$
$$v_{t+1} = \beta_2 v_t + (1-\beta_2)g_t^2 \quad \text{(2nd moment)}$$

**Bias correction** (critical — $m_0 = v_0 = 0$ biases estimates toward 0 early in training):

$$\hat{m}_{t+1} = \frac{m_{t+1}}{1-\beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{v_{t+1}}{1-\beta_2^{t+1}}$$

**Update:**

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$.

**Effective step size** bounded by $\approx \eta \cdot \frac{1-\beta_2^{t+1}}{1-\beta_1^{t+1}}$ early in training; converges to $\eta$ later.

### AdamW (Adam + Weight Decay)

Standard L2 regularization adds $\lambda\theta$ to the gradient before Adam — but Adam's adaptive scaling makes the effective weight decay vary per parameter. AdamW decouples weight decay from the gradient update:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon}\hat{m}_{t+1} - \eta\lambda\theta_t$$

Weight decay applied directly to parameters, not through gradient. Standard choice for transformers.

### Optimizer Comparison

| Optimizer | Adaptive LR | Momentum | Memory | Default LR |
|-----------|------------|---------|--------|-----------|
| SGD | No | No | $O(1)$ | 0.01–0.1 |
| SGD+Momentum | No | Yes | $O(d)$ | 0.01 |
| RMSProp | Yes (2nd moment) | No | $O(d)$ | 1e-3 |
| Adam | Yes (1st + 2nd) | Yes | $O(2d)$ | 1e-3 |
| AdamW | Yes + decoupled WD | Yes | $O(2d)$ | 1e-4 |

---

## 6. Learning Rate Scheduling

### Warmup + Cosine Decay

Used in transformer training (BERT, GPT). Avoids large gradients early when parameters are random:

$$\eta_t = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warm}} & t \leq T_{warm} \\ \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\!\left(\pi\frac{t-T_{warm}}{T-T_{warm}}\right)\right) & t > T_{warm} \end{cases}$$

### Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

$\gamma = 0.1$, $s = 30$ epochs (ImageNet ResNet schedule).

### Reduce on Plateau

Monitor validation loss; if no improvement after `patience` epochs, multiply LR by `factor` (e.g., 0.5). Implemented in PyTorch as `ReduceLROnPlateau`.

---

## 7. Regularization Techniques

### Dropout

During training, independently zero each neuron's output with probability $p$:

$$\tilde{\mathbf{a}}^{(l)} = \mathbf{m}^{(l)} \odot \mathbf{a}^{(l)}, \quad m_j^{(l)} \sim \text{Bernoulli}(1-p)$$

Scale by $\frac{1}{1-p}$ (inverted dropout) so expected value is unchanged. **At inference:** disable dropout (no masking, no scaling).

**Why it works:** prevents co-adaptation of neurons — each neuron must learn features useful without relying on specific other neurons being active. Equivalent to averaging exponentially many thinned networks.

### Batch Normalization

For a mini-batch $\mathcal{B} = \{z_1, \ldots, z_m\}$:

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m z_i, \quad \sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (z_i - \mu_\mathcal{B})^2$$

$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}}, \quad y_i = \gamma\hat{z}_i + \beta$$

$\gamma, \beta$ are learnable per-feature parameters. At inference, use exponential moving averages of $\mu$ and $\sigma^2$ accumulated during training.

**Benefits:** reduces internal covariate shift, allows higher LR, reduces sensitivity to initialization, mild regularization effect.

**Placement:** original paper applies BN before activation; later practice (ResNets) often applies after or uses Pre-LN (before attention in transformers).

### Layer Normalization

Normalizes across features for a single example (not across the batch):

$$\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \mu = \frac{1}{H}\sum_{j=1}^H z_j, \quad \sigma^2 = \frac{1}{H}\sum_{j=1}^H (z_j - \mu)^2$$

Independent of batch size → works for batch size 1, RNNs, transformers (standard in attention architectures).

---

## 8. MLOps — Experiment Tracking with MLflow

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Experiment** | Named group of runs (e.g., "iris_classification") |
| **Run** | Single execution — logs params, metrics, artifacts |
| **Parameters** | Hyperparameters (LR, epochs, hidden_dim) — logged once |
| **Metrics** | Scalar values per step (train_loss, val_acc) — logged per epoch |
| **Artifacts** | Files (model.pkl, confusion_matrix.png, requirements.txt) |
| **Model** | Logged model in MLflow flavor (sklearn, pytorch, pyfunc) |
| **Registry** | Versioned model store with Staging/Production lifecycle |

### Run Lifecycle

```
mlflow.start_run()
  → mlflow.log_param("lr", 1e-3)
  → mlflow.log_metric("train_loss", loss, step=epoch)
  → mlflow.log_artifact("model.pkl")
  → mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

### Model Registry Stages

```
None → Staging → Production → Archived
```

A model version in `Production` can be loaded anywhere:

```python
model = mlflow.sklearn.load_model("models:/iris_rf/Production")
```

### Autolog

```python
mlflow.sklearn.autolog()   # automatically logs params, metrics, model
mlflow.pytorch.autolog()   # logs training loop metrics, model checkpoints
```

Autolog hooks into the library's fitting code — no manual `log_param` calls needed.

---

## 9. Data Drift & Model Monitoring

### Why models degrade in production

1. **Data drift (covariate shift):** $P(\mathbf{x})$ changes — input distribution shifts from training.
2. **Concept drift:** $P(y|\mathbf{x})$ changes — relationship between features and label changes.
3. **Label drift (prior probability shift):** $P(y)$ changes without $P(\mathbf{x}|y)$ changing.

### Statistical Drift Tests

| Test | Data Type | Null Hypothesis | Interpretation |
|------|----------|-----------------|----------------|
| Kolmogorov-Smirnov | Continuous univariate | Distributions identical | p < 0.05 → drift |
| Chi-Squared | Categorical | Same proportions | p < 0.05 → drift |
| Population Stability Index (PSI) | Continuous/categorical | Same distribution | PSI > 0.2 → major drift |
| Maximum Mean Discrepancy (MMD) | Multivariate | Same distribution | MMD > threshold → drift |
| Jensen-Shannon Divergence | Any | Same distribution | JS > 0.1 → notable |

### Population Stability Index (PSI)

$$\text{PSI} = \sum_{j=1}^{J} (A_j - E_j) \ln\!\left(\frac{A_j}{E_j}\right)$$

where $A_j$ is the actual (production) fraction in bin $j$, $E_j$ is the expected (reference) fraction. Rule of thumb: PSI < 0.1 no change, 0.1–0.2 slight change, > 0.2 major change.

### Kolmogorov-Smirnov Statistic

$$D = \sup_x |F_{ref}(x) - F_{prod}(x)|$$

Maximum absolute difference between empirical CDFs. $p$-value derived from $D$ and sample sizes via the KS distribution.

### Monitoring Pipeline

```
Production predictions
  → Feature store snapshot (hourly/daily)
    → Drift detector (KS, PSI, MMD)
      → Alert if threshold exceeded
        → Retrain trigger or manual review
```

---

## 10. Interview Reference — Deep Learning

### Q: What is the vanishing gradient problem?

In deep networks with sigmoid/tanh activations, $|g'(z)| < 1$ everywhere ($\sigma' \leq 0.25$, $\tanh' \leq 1$). Repeated multiplication across $L$ layers: $\prod_{l=1}^{L} g'^{(l)} \approx c^L$ for $c < 1$ → exponentially small gradients in early layers → slow or no learning. Solutions: ReLU activations, residual connections (gradient highway), careful initialization (He/Xavier), gradient clipping.

### Q: What is the exploding gradient problem?

Opposite: large weight matrices cause gradients to grow exponentially across layers. Common in RNNs with long sequences. Solution: gradient clipping (clip gradient norm to max value):

$$g \leftarrow g \cdot \frac{\text{clip\_norm}}{\max(\|g\|_2, \text{clip\_norm})}$$

### Q: Why does Adam use bias correction?

$m_0 = v_0 = \mathbf{0}$. At step 1: $m_1 = (1-\beta_1)g_1 \approx 0.1 g_1$ (if $\beta_1=0.9$) — severely underestimates the true gradient. Dividing by $(1-\beta_1^t)$ corrects this; as $t \to \infty$, the correction factor $\to 1$.

### Q: What is gradient clipping and when do you use it?

Rescales gradient vector when its L2 norm exceeds a threshold. Used in: RNN/LSTM training (long sequences amplify gradients), early transformer training, any time loss spikes are observed. Does not change direction, only magnitude.

### Q: BatchNorm vs LayerNorm?

BatchNorm normalizes across the batch dimension — requires batch size > 1, accumulates running stats, computationally cheap. LayerNorm normalizes across feature dimension for each sample independently — works with batch size 1, no running stats needed, used in transformers.

### Q: What does MLflow track vs what does a feature store track?

MLflow tracks **model artifacts, training metrics, hyperparameters, and model versions**. A feature store (Feast, Tecton) tracks **feature definitions, feature values, and serves consistent features** between training and serving (prevents training-serving skew).

### Q: What is the difference between data drift and concept drift?

Data drift: input feature distribution $P(\mathbf{x})$ changes (e.g., user demographics shift). Concept drift: the mapping $P(y|\mathbf{x})$ changes (e.g., fraud patterns evolve). Data drift is detectable from features alone; concept drift requires labeled production data to detect.

---

## Resources

### Books
- **Deep Learning** — Goodfellow, Bengio, Courville. Free online at `deeplearningbook.org`. Chapters 6 (feedforward networks), 7 (regularization), 8 (optimization) map directly to this module.
- **Dive into Deep Learning** (`d2l.ai`): Interactive Jupyter book with PyTorch and NumPy implementations. Covers backprop, optimizers, and normalization in depth.

### Video & Courses
- **Andrej Karpathy — Neural Networks: Zero to Hero** (YouTube): Six videos building everything from scratch in Python. The micrograd video directly matches `nn_numpy.py` in this module.
- **Stanford CS231n — Convolutional Neural Networks for Visual Recognition**: Full lecture slides and notes free at `cs231n.stanford.edu`. Backprop notes in Assignment 2 are excellent.
- **fast.ai Practical Deep Learning for Coders**: Top-down, code-first. Complements this module's bottom-up approach.

### Papers
- **Adam: A Method for Stochastic Optimization** — Kingma & Ba (2015): `arxiv.org/abs/1412.6980`
- **Batch Normalization: Accelerating Deep Network Training** — Ioffe & Szegedy (2015): `arxiv.org/abs/1502.03167`
- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** — Srivastava et al. (2014): JMLR.
- **Decoupled Weight Decay Regularization (AdamW)** — Loshchilov & Hutter (2019): `arxiv.org/abs/1711.05101`

### MLOps Tooling
- MLflow documentation (`mlflow.org/docs`): Tracking, models, registry APIs.
- Weights & Biases (`wandb.ai/site`): Alternative experiment tracker with richer visualization. Same concepts as MLflow.
- Evidently AI (`evidentlyai.com`): Production drift detection library — same metrics as `monitoring.py` but with a dashboard.

---

> **Next module:** [Module 06 — GenAI Core](06-genai-core.md)
