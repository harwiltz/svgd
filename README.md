# `svgd`

Implementation of Stein Variational Gradient Descent:

```
@misc{liu2019stein,
  title = {Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm},
  author = {Qiang Liu and Dilin Wang},
  year = {2019},
  eprint = {1608.04471},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML},
}
```

## Mode Collapse

It seems the algorithm is struggling with mode collapse even on a simple mixture of Gaussians experiment.

![](result.png)
