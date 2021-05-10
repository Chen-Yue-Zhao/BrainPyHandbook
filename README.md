# BrainPy介绍

本章将介绍计算神经科学中的一系列神经元模型、突触模型和网络模型。在正式开始之前，我们希望先为读者简单介绍如何使用BrainPy实现计算神经科学模型，以方便读者理解附在每个模型之后的BrainPy实现代码。

`BrainPy`是一个用于计算神经科学和类脑计算的Python平台。要使用BrainPy进行建模，用户通常需要完成以下三个步骤：

1）为神经元和突触模型定义Python类。BrainPy预先定义了数种基类，用户在实现特定模型时，只需继承相应的基类，并在模型的Python类中定义特定的方法来告知BrainPy该模型在仿真的每个时刻所需的操作。在此过程中，BrainPy在微分方程（如ODE、SDE等）的数值积分、多种后端（如`Numpy`、`PyTorch`等）适配等功能上辅助用户，简化实现的代码逻辑。

2）将模型的Python类实例化为代表神经元群或突触群的对象，将这些对象传入到BrainPy的`Network`类的构造函数中，初始化一个网络，并调用`run`方法进行仿真。

3）调用BrainPy的测度模块`measure`或可视化模块`visualize`等，展示仿真结果。

带着上述对BrainPy的粗略理解，我们希望下述各节中的代码实例能够帮助读者更好地理解计算神经科学模型和其中蕴含的思想。下面，我们将按照[神经元模型](neurons.md), [突触模型](synapses.md), and [网络模型](networks.md)的顺序进行介绍。

*BrainPy的文档及更多细节请参考我们的Github仓库：https://github.com/PKU-NIP-Lab/BrainPy和https://github.com/PKU-NIP-Lab/BrainModels。*