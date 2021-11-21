# tddl (TéDDLe)
Tensor Decomposition for Deep Learning


## Install Guide

Install pytorch `pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

Install tddl `pip install -e ."[dev]"`


## Getting started

    from tddl.factorizations import factorize_network

    factorize_network(      
        model,                  # Changes your pytorch model inplace.
        layers,                 # Modifies only layers (or layer type) you specify,
        factorization='tucker', # into specific factorization,
        rank=0.5,               # with a given (fractional) rank.
        decompose_weights=True, # Decompose the weights of the model.
    )

To know how the layers are numbered, we provide the utility function `number_layers`.

    from tddl.factorizations import number_layers

    number_layers(model)


## Hyperparameter Tuning with RayTune
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html