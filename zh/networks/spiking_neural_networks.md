## 3.1 脉冲神经网络

脉冲神经网络的特点是网络分别建模、计算每个神经元和突触。研究者希望通过这种仿真来观察大规模神经网络的行为，并验证相关理论推导。本节将介绍两个经典的脉冲神经网络模型：Vreeswijk和Sompolinsky（1996）提出的兴奋-抑制平衡网络和Wang（2002）提出的决策网络。

### 3.1.1 兴奋-抑制平衡网络

上世纪90年代初，学界发现，在大脑皮层中神经元有时表现出一种在时间上不规则的发放特征。这种特征广泛地存在于脑区中，但当时人们对它的产生机制和主要功能都了解不多。

Vreeswijk和Sompolinsky（1996）提出了**兴奋-抑制平衡网络**（E/I balanced network），希望能够解释神经元这种不规则的发放，并提示了这种结构在功能上可能的优势。

<center><img src="../../figs/snns/3-2.png" width="616" height="300"></center>

<center><b>Fig.3-1 兴奋-抑制平衡网络结构 （<cite id="reffn_1">Vreeswijk and Sompolinsky, 1996 <sup><a href="#fn_1">1</a></sup></cite>）</b></center>

图3-1画出了兴奋-抑制平衡网络的结构。该网络由兴奋性LIF神经元和和抑制性LIF神经元构成，其数量比$$N_E: N_I = 4:1$$。在网络两类神经元之间和同类神经元之内，建立了四组指数型突触连接，分别是兴奋-兴奋连接（E2E conn），兴奋-抑制连接（E2I conn），抑制-兴奋连接（I2E conn），抑制-抑制连接（I2I conn）。在代码中我们通过定义符号相反的突触权重，来指明突触连接的兴奋性或抑制性。

<center><img src="../../figs/snns/codes/EInet1.png">	</center>

*注：LIF神经元和指数型突触的实现请参见第1节《神经元模型》和第2节《突触模型》*

兴奋-抑制平衡网络在结构上最大的特征是神经元间强随机突触连接，连接概率为$$0.1$$，属于稀疏连接。

这种强的突触连接使得网络中每个神经元都会接收到很大的来自网络内部的兴奋性和抑制性输入。但是，这两种输入一正一负相互抵消，最后神经元接收到的总输入将保持在一个相对小的数量级上，仅足以让神经元的膜电位上升到阈值电位，引发其产生动作电位。

由于突触连接和噪声带来的随机性，网络中神经元接收到的输入也在时间和空间上具有一定的随机性（尽管总体保持在阈值电位量级上），这使得神经元的发放也具有随机性，保证兴奋-抑制平衡网络能够自发产生前述的时间上不规则的发放特征。

下述仿真结果中，可以看到网络中的神经元从一开始的强同步发放慢慢变为时间上不规则的发放。

<center><img src="../../figs/snns/codes/EInet2.png">	</center>

<center><img src="../../figs/snns/codes/EInet3.png">	</center>


![png](../../figs/snns/out/output_8_0.png)

与此同时，作者还提出了这种发放特征在大脑中可能提供的功能：兴奋-抑制平衡网络可以快速跟踪外部刺激的变化。假如该网络真的是大脑中神经元产生不规则发放背后的机制，那么真实的神经元网络也可能拥有同样的特性。

如图3-2所示，当没有外部输入时，兴奋-抑制平衡网络中神经元的膜电位相对均匀且随机地分布在静息电位$$V_0$$和阈值电位$$\theta$$之间。当网络接收到一个小的外部恒定输入时，那些膜电位原本就落在阈值电位附近的神经元（图中标为红色）就能很快地发放，在网络尺度上，表现为网络的发放率随输入变化而快速改变。

<center><img src="../../figs/snns/EI_neuron_distribution.jpg" width="352" hight="300"></center>

<center><b>Fig.3-2 兴奋-抑制平衡网络中神经元膜电位的分布（<cite id="reffn_2">Tian et al.，2020 <sup><a href="#fn_2">2</a></sup></cite>）</b></center>

仿真证实，在这种情况下，网络对输入产生反应的延迟时间和突触的延迟时间处于同一量级，并且二者都远小于单神经元从静息电位开始积累同样大小的外部输入直到产生动作电位所需的延迟时间（Vreeswijk和Sompolinsky，1996；Tian et al.，2020）。因此，兴奋-抑制平衡网络面对外部输入的变化可以快速反应，改变自身的活跃水平。

------

### 3.1.2 决策网络

计算神经科学的网络建模也可以对标特定的生理实验任务，比如视觉运动区分实验（Parker和Newsome，1998；Roitman和Shadlen，2002）。

在该实验中，参与实验的猕猴将观看一段随机点的运动展示。在展示过程中，随机点以一定比例（该比例被定义为一致度（coherence））向特定方向运动，其他点则向随机方向运动。猕猴被要求判断随机点一致运动的方向，并通过眼动给出答案。同时，研究者通过电生理手段记录猕猴LIP神经元的活动。

<center><img src="../../figs/snns/3-4.png" width="440" height="200"> </center>

<center><b>Fig.3-4 生理实验中随机点的运动示意图（<cite id="reffn_3">Gerstner et al., 2014<sup><a href="#fn_3">3</a></sup></cite>）</b></center>

Wang（2002）提出了本节所述的决策网络，希望建模在视觉运动区分实验中猕猴大脑新皮层的决策回路的活动。

<center><img src="../../figs/snns/3-5.png" width="440" height="240"></center>

<center><b>Fig.3-5 决策网络结构（<cite id="reffn_4">Wang，2002<sup><a href="#fn_4">4</a></sup></cite>）</b></center>

如图3-5所示，网络同样基于兴奋-抑制平衡网络。兴奋性神经元和抑制型神经元的数量比是$$N_E:N_I = 4:1$$，调整参数使得网络处在平衡状态下。

为了简化模型，实验被设定为一个二选一的任务：在兴奋性神经元群中，特别地标出两个选择性子神经元群A和B，其他的兴奋性神经元称为非选择性神经元，用下标$$_{non}$$表示。A群和B群的数目均为兴奋性神经元的0.15倍（$$N_A = N_B = 0.15N_E$$），它们分别代表着两个相反的运动方向，可以视作随机点要么向左，要么向右，没有第三个方向，网络的决策结果也必须在这两个子神经元群中产生。非选择性神经元的数目为$$N_{non} = (1-2*0.15)N_E$$。

<center><img src="../../figs/snns/codes/DMnet2.PNG">	</center>

决策网络中共有四组突触——E2E，E2I，I2E和I2I突触连接，其中兴奋性突触实现为AMPA突触，抑制性突触实现为GABAa突触。

由于网络需要在A群和B群之间作出决策，所以这两个子神经元群之间必须形成一种竞争关系。一个选择性子神经元群应当激活自身，并同时抑制另一个选择性子神经元群。

因此，网络中的E2E连接被建模为有结构的连接。如表3-1所示，$$w+ > 1 > w-$$。这样，在A群或B群的内部，兴奋性突触连接更强，形成了一种相对的自激活；而在A、B两个选择性子神经元群之间或是A群、B群和非选择性子神经元群之间，兴奋性突触连接较弱，实际上形成了相对的抑制。A和B两个神经元因此产生竞争，迫使网络做出二选一的决策。

<center><b>Sheet 3-1 决策网络中兴奋性神经元间连接权重的分布</b></center>

<center><img src="../../figs/snns/3-6.png"></center>

<center><img src="../../figs/snns/codes/DMnet1.PNG">	</center>

决策网络接受的外部输入可分为两类：

1）所有神经元都收到从其他脑区传来的非特定的背景输入，表示为AMPA突触介导的高频泊松输入（2400Hz）。

<center><img src="../../figs/snns/codes/DMnet3.PNG">	</center>

2）仅选择性的A群和B群收到外部传来的刺激输入，表示为AMPA突触介导的较低频泊松输入（约100Hz内）。

给予A和B神经元群的泊松输入的频率均值（$$\mu_A$$、$$\mu_B$$）有一定差别，对应到生理实验上，代表猕猴看到的随机点朝两个相反方向运动的比例（用coherence表示）不同。这种输入上的差别引导着网络在两个子神经元群中做出决策。
$$
\rho_A = \rho_B = \mu_0/100
$$

$$
\mu_A = \mu_0 + \rho_A * coherence
$$

$$
\mu_B = \mu_0 + \rho_B * coherence
$$

每50毫秒，泊松输入的实际频率$$f_x$$遵循由均值$$\mu_x$$ 和方差$$\delta^2$$定义的高斯分布，重新进行一次采样。
$$
f_A \sim N(\mu_A, \delta^2)
$$

$$
f_B \sim N(\mu_B, \delta^2)
$$

<center><img src="../../figs/snns/codes/DMnet4.PNG">	</center>

<center><img src="../../figs/snns/codes/DMnet5.PNG">	</center>

下图中可以看到，在本次仿真中，子神经元群A收到的刺激输入平均大于B收到的刺激输入。经过一定的延迟时间，A群的活动水平明显高于B群，说明网络做出了正确的选择。

<center><img src = "../../figs/snns/out/output_18_1.png" width="400" height="796"></center>



[^ 1]:Van Vreeswijk, Carl, and Haim Sompolinsky. "Chaos in neuronal networks with balanced excitatory and inhibitory activity." *Science* 274.5293 (1996): 1724-1726.
[^ 2 ]:Tian, Gengshuo, et al. "Excitation-Inhibition Balanced Neural Networks for Fast Signal Detection." *Frontiers in Computational Neuroscience* 14 (2020): 79.
[^ 3 ]:Gerstner, Wulfram, et al. Neuronal dynamics: From single neurons to networks and models of cognition. Cambridge University Press, 2014.
[^ 4 ]:Wang, Xiao-Jing. "Probabilistic decision making by slow reverberation in cortical circuits." *Neuron* 36.5 (2002): 955-968.