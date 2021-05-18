## 3. 网络模型

到此，读者已经了解了几种最常见、最经典的神经元和突触模型，是时候更进一步了。本节中，我们将介绍计算神经科学中两类重要的网络模型：脉冲神经网络（spiking neural networks）和发放率神经网络（firing rate neural networks）。

脉冲神经网络的特点是网络分别建模、计算每个神经元和突触。研究者希望通过这种仿真来观察大规模神经网络的行为，并验证相关理论推导。

而发放率神经网络则将一个神经元群简化为单个的发放率单元，以计算整个神经元群的发放率代替对单神经元的模拟。这种模拟经常可以得到和脉冲神经网络在网络尺度上类似的结果，而计算过程却被大大地简化了。

本章中，我们将介绍分属于这两类网络模型的兴奋-抑制平衡网络、抉择网络和连续吸引子神经网络。

> 注：本章所述模型的完整BrainPy代码请见[附录](appendix/networks.md)，或[右键点此](appendix/networks.ipynb)下载jupyter notebook版本。

### 3.1 [兴奋-抑制平衡网络](networks/EI_balanced_network.md)

### 3.2 [抉择网络](networks/decision_making_networks.md)

### 3.3 [连续吸引子神经网络](networks/continuous_attractor_neural_network.md)