# Practice Machine Learning

This repo is a personal sandbox and correspond to personal implementation
 and interpretation of ML algorithms.

# Rules

- Only use `numpy`. No `scikit-learn`, `tensorflow` and co.

# Components

See the documentation and tests inside each file for more details.
- kmeans : K-Means. (Lloyd's algorithm with pure numpy). 
K-Median and K-Medoids can be made out of kmean as an arbitrary 
metric can be provided.
- knn : K-Nearest Neighbors (pure numpy)
- pcn : Perceptron Neural Network
- mlp : Multi-layers Perceptron
- rbf : Radial Basis Function
- linreg : Linear Regresion
- dtree : Decision tree (ID3 heuristics)

# ToDo:

- Add description to linreg (z-test, p-test, ...)
- Add classifier evaluations (confusion matrix, ...)
- Dtree : improve prunning algorithm
- knn : norme as a parameter
- mlp : sigmoid as a parameter

# Licence

MIT Licence is used. See the LICENCE file for mode details.
