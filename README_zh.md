## BrainPy中文手册（BrainPy Handbook）

> 英文版README.md [click here](README.md)

《BrainPy中文手册》网页版请见https://pku-nip-lab.github.io/BrainPyHandbook/zh。如需查看英文网页版，请见https://pku-nip-lab.github.io/BrainPyHandbook/en/。本手册亦提供PDF下载版本，见本仓库（https://github.com/PKU-NIP-Lab/BrainPyHandbook）下`./pdf/book_<language>.pdf`。



------

#### 书籍介绍

在本手册中，我们将介绍一系列经典的计算神经科学模型，包括神经元模型、突触模型和网络模型，并提供它们的BrainPy——基于Python的计算神经科学及类脑计算平台——实现。

我们希望，本手册不仅能列出模型的定义、功能，也能对计算神经科学这一学科的脉络和思想有所涉及。通过阅读本手册，读者若能建立对计算神经科学的基本认识，知道如何在学术或应用场景中选择合适的模型、或对现象进行适当的建模，那就是我们在编辑本书时所期望的。

此外，模型后附BrainPy实现代码，帮助初学者快速上手，完成第一次仿真。对于熟悉计算神经科学的读者，我们也希望书中的例子能帮助大家了解BrainPy的优势、学习BrainPy的使用。



------

#### 环境安装

理想情况下，用户应可在我们的网页版和PDF下载页面方便地获取最新版本，而不需从.md文件自行生成手册文本。

我们在这里提供的安装环境步骤是为手册中附有的BrainPy代码而准备的。我们推荐有意进一步了解的学生和研究者参考BrainPy的[仓库](https://github.com/PKU-NIP-Lab/BrainPy)、[文档](https://brainpy.readthedocs.io/en/latest/)，以及BrainModels的[仓库](https://github.com/PKU-NIP-Lab/BrainModels)、[文档](https://brainmodels.readthedocs.io/en/latest/)，那里的代码更加高效，但若读者只希望运行模型后附的代码查看效果，则请在本目录下安装前置包：

```
pip install -r requirements.txt
```

模型代码存储在`./<laguage>/appendix/`下，神经元、突触、网络模型分别存储在相应命名的.md文件和.ipynb文件中，请读者选择熟悉的方式运行代码。



------

#### 目录

* [0. 简介](README.md)
* [1. 神经元模型](neurons.md)
  * [1.1 生物背景](neurons/biological_background.md)
  * [1.2 生理模型](neurons/biophysical_models.md)
  * [1.3 简化模型](neurons/reduced_models.md)
  * [1.4 发放率模型](neurons/firing_rate_models.md)
* [2. 突触模型](synapses.md)
  * [2.1 突触动力学模型](synapses/dynamics.md)
  * [2.2 突触可塑性模型](synapses/plasticity.md)
* [3. 网络模型](networks.md)
  * [3.1 脉冲神经网络](networks/spiking_neural_networks.md)
  * [3.2 发放率神经网络](networks/rate_models.md)
* [附录：模型代码](appendix.md)
  * [神经元模型](appendix/neurons.md)
  * [突触模型](appendix/synapses.md)
  * [网络模型](appendix/networks.md)



------

#### 备注

如对手册内容有建议或意见，欢迎在仓库中提出issues。

