# BrainPy Handbook

English online version of *BrainPy Handbook* is available at https://pku-nip-lab.github.io/BrainPyHandbook/en/. Chinese online version is available at https://pku-nip-lab.github.io/BrainPyHandbook/zh. To download PDF version, see path `./pdf/book_<language>.pdf` is this repository (https://github.com/PKU-NIP-Lab/BrainPyHandbook).



#### Handbook introduction

------

In this handbook, we will introduce a series of canonical computational neuroscience models, including neuron models, synapse models and network models. We also provide their realization of BrainPy ---- a Python platform for computational neuroscience and brain-inspired computing.

We hope that, other than listing the models' definitions and functions, this handbook can also  provide an overview on the context and thoughts of the discipline of computational neuroscience. Through reading *BrainPy Handbook*, if our readers can establish a basic understanding of computational neuroscience modeling, know how to choose appropriate models in research and in application, or how to properly model biophysical phenomena, that is what we expected when editing this handbook.

The BrainPy realization is attached to each model in our handbook to help beginners understanding the models and how to run their first simulation. For readers that are familiar with computational neuroscience, we also hope these codes can tell you the features and advantages of BrainPy.



#### Environment

------

Readers should be able to get our newest web version and PDF version on our website, and so not need to generate the handbook from .md files on their own.

We provide this environment requirements for running the codes attached in our handbook. We suggest students and researchers to see BrainPy's [repository](https://github.com/PKU-NIP-Lab/BrainPy) and [documentation](https://brainpy.readthedocs.io/en/latest/)，BrainModels' [repository](https://github.com/PKU-NIP-Lab/BrainModels) and [documentation](https://brainmodels.readthedocs.io/en/latest/), in which the codes are more effective. However, if you only need to run a simple simulation of our code, please install the requirements:

```
pip install -r requirements.txt
```

Attached coded are integrated in the path `./<laguage>/appendix/`.



#### Catalog

------

* [0. Introduction](README.md)
* [1. Neuron models](neurons.md)
  * [1.1 Biological background](neurons/biological_background.md)
  * [1.2 Biophysical models](neurons/biophysical_models.md)
  * [1.3 Reduced models](neurons/reduced_models.md)
  * [1.4 Firing rate models](neurons/firing_rate_models.md)
* [2. Synapse models](synapses.md)
  * [2.1 Synaptic models](synapses/dynamics.md)
  * [2.2 Plasticity models](synapses/plasticity.md)
* [3. Network models](networks.md)
  * [3.1 Spiking neural networks](networks/spiking_neural_networks.md)
  * [3.2 Firing rate networks](networks/rate_models.md)
* Appendix
  * [Neuron models](appendix/neurons.md)
  * [Synapse models](appendix/synapses.md)
  * [Network models](appendix/networks.md)



#### Note

------

Please raise issues if you have suggestions about *BrainPy Handbook*.