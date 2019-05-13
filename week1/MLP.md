## Multi-Layer Percptron

## Summary

- Just a basic feed-forward neural network, _fully_ connected.
- It can do the supervised learning as well as the unsupervised one.
- Motivation: neurons & function approximation
- Some Limitations
  - Too may parameters...
  - Prone to overfit (so lets bring in regularization!)
  - Cannot do parallel computation
- Training: use the backpropagation approach, which is merely multivariate chain rule of (partial) derivtives.

### Algorithm

### Implementation

#### Hints

- For dot product $x \cdot y$: `x.dot(y)`
- For matrix transpose $ x^\top$ use `x.transpose`
- Keep track of the dimensions of all numpy arrays
- Start with a simpleMLP, then to a standard one.
-

#### Coding

See the MLP.py file

#### Reflection

### Note Behind
