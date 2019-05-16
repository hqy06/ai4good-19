## k-Means

### Summary

- the simplest unsupervised learning algorithm possible
- Remember that all clustering problems are NP-hard
- The heuristic of "start randomly and alternate b/w assign/refit"
- Soft k-Means vs. Hard k-Means
  - hard: assign data point to cluster
  - soft: each cluster have responsibility (b/w 0-1) for each point
  - Intuitively, the "soft" version of k-Means allows a cluster to use more info on data in the "refit" step
- Objective for optimization: min distance

### Convergence of k-Means

When assignment $r^{(n)}$ changed, the objective function $J$ reduced.

When cluster center $m_k$ updated, the objective $J$ also reduced.

So if no $r^{(n)}$ changed at the "assign" step, we are at local minimum.

Note $J$ is may not be convex so $J$ may not converge to global minimum. So try many random starting point & try merge-split cluster

### Algorithm

For the normal version of k-Means

```
1. Randomly init cluster centers {m_k}
2. Repeat until converge:
  3. Assign responsibility to each point in one-hot fashion
  4. Refit by update m_k to the center of pt assigned to k
```

For the soft version of k-Means

```
1. Randomly init cluster centers {m_k}
2. Repeat until converge:
  3. Assign responsibility to each point using
     r_k = e^(beta* dist(x,k)) / SUM_j(e^beta*dist(x, j))
  4. Refit by update m_k to its expectation.
```

### Implementation

- One nice trick: for classification dataset in the [UCI repo](http://archive.ics.uci.edu/ml/index.php), remove their label and you get datasets for clustering.

### Hyperparameters

$k$ the number of clusters: the model becomes more detailed as $k$ grows larger.

$\beta$ the coefficient usded for responsibility distro: hard to tone and analyze.

### Note to Self

- Epoch, batch & iteration:
  https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks
