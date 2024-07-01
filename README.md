# MIA Attention Mechanism implementation experiment with PyTorch

Implementation of Attention Mechanism presented in the seminal paper - [Attention is All what you need](https://arxiv.org/abs/1706.03762). An explanation about inspiration behind this core mechanism is presented in my post - [Atención en el Modelo de Lenguaje MIA](https://edwinmoo.substack.com/p/embeddings-en-el-modelo-de-lenguaje?r=233vmr).

## MIA Attention Mechanism experiment details

Before anything else we analized what is the inspiration behind this core feature in the Transformer Architecture and implement a very first version of the Attention Equation using Numpy Library:

- A notebook was created where we describe our analysis about the Attention Equation and the origin of its independent matrix variables Q, K, and V.
- The notebook provides a basic implementation of this equation using Numpy Library.
- Because MIA is based in Pytorch Framework I provide an implementation of Attention Mechanism using this Deep Learning Framework taking advantage of Pytorch nn.Module class as base abstraction.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── model
│   └── miaattentionmechanism.py
└── notebooks
    └── miaattentionintuition.py
```

- **model/miaattentionmechanism.py** - Defines the module classes that implement the Multi-Head Attention Layer and Attention Head Layer.
- **notebooks/miaattentionmechanism.py** - Notebook that using code explains the origin and nature of matrix Q, K, and V and its relationship with Retrival Information Systems.

## How to start the notebook

You can open the notebook from Google Collab or using VSCode with the Jupyter plug-in already installed, each section of this notebook explains our understanding about from where the matrix Q, K and V come from together with python code for demostrative purposes.

Note: It is important to note that the MIA's Pythorch module clases (layers) that implement the Attention Mechanism can't be executed in isolation but integrated only, however, this code is totally functional and it is part of the MIA base code.

```
In case you want to experiment and/or integrate MIA's attention layer in your own code please

Run below commands

1.- python -m venv miaattmecenv
2.- source ./miaattmecenv/bin/activate
3.- pip install -r requirements.txt

This is in order to prepare your own development environment.

```

