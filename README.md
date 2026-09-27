# Micrograd-Style Autograd Implementation

![Micrograd reference image](./micrograd-reference.jpeg)

This project is a small scalar automatic-differentiation engine and
multi-layer perceptron written from scratch in Python. It was created while
following Andrej Karpathy's **micrograd** video, with the goal of making
forward evaluation and backpropagation easy to inspect.

The main implementation is [`implementation.py`](./implementation.py).
[`implementation.ipynb`](./implementation.ipynb) contains the notebook
version from which the Python script was generated.

## What is included

### `Value`

`Value` represents one scalar in a computation graph. Each value stores:

- `data`: the scalar's numerical value
- `grad`: the derivative accumulated during backpropagation
- `_prev`: parent values used to build the graph
- `_backward`: a local gradient function for the operation that created it

The class implements scalar addition, multiplication, powers, division,
negation, subtraction, exponentiation, and hyperbolic tangent. Calling
`backward()` builds a topological ordering of the graph and applies the local
derivatives in reverse order.

This version intentionally does not include a `relu` method or `__radd__`.
The network uses `tanh` activations, and reductions in the training code
start with a `Value` object explicitly.

### Neural-network building blocks

- `Neuron` stores weights and a bias, computes a weighted sum, and applies
  `tanh`.
- `Layer` evaluates several neurons in parallel.
- `MLP` combines layers and exposes all trainable parameters through
  `parameters()`.

The example model is an MLP with the architecture:

```text
2 inputs -> 16 neurons -> 16 neurons -> 1 output
```

## Dataset and training

The script initially creates a 200-sample two-moons dataset with
`sklearn.datasets.make_moons`. It then trains the MLP for 100 epochs using:

- Mean squared error
- Manual reverse-mode backpropagation
- A learning rate of `0.5`
- Plain gradient-descent parameter updates

The script prints the loss for each epoch and finishes by printing the
classification accuracy on the embedded two-moons data.

The training example currently includes a fixed list of feature values and
labels for the final accuracy calculation. The generated two-moons data is
used for training before that final evaluation.

## Run the implementation

From the project root, run:

```powershell
python implementation.py
```

The output includes the generated dataset shapes, one loss line per epoch,
and a final accuracy summary.

To explore the implementation interactively, open
`implementation.ipynb` in Jupyter Notebook or JupyterLab and run its cells in
order.

## Learning resources

The implementation was inspired by Andrej Karpathy's
[micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0). This is an
educational implementation intended for learning; it is not a replacement for
production autodiff or neural-network libraries such as PyTorch.
