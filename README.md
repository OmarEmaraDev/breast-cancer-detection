# Breast Cancer Detection

A Fine Needle Aspiration (FNA) breast cancer binary classifier implemented using
both neural networks and support vector machines. A college assignment.

## Dataset

The Fine Needle Aspiration (FNA) Wisconsin dataset available in the
[UCI Archive](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

## Dependencies

- Numpy.
- Tensorflow.
- Scikit-learn.
- TeX-Distribution. (Documentation)

## Run

Run both the neural network and SVM classifiers. The code will write the results
needed by the documentation.

```
python neuralNetwork.py
python supportVectorMachine.py
```

Compile the documentation using latexmk if available.

```
cd documentation/
latexmk
```
