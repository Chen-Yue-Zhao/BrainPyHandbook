
## 1.4 发放率模型

发放率模型比简化模型更加简单。在这些模型中，每个计算单元代表一个神经元群，而单神经元模型中的膜电位变量$$V$$也被发放率变量$$a$$（或$$r$$或$$\nu$$）所取代。本节将介绍一个经典的发放率单元。

### 1.4.1 发放率单元

Wilson和Cowan（1972）来表示在兴奋性和抑制性皮层神经元微柱中的活动。变量$$a_e$$和$$a_i$$中的每个元素都表示一个包含复数神经元的皮层微柱中神经元群的平均活动水平。
$$
\tau_e \frac{d a_e(t)}{d t} = - a_e(t) + (k_e - r_e * a_e(t)) * \mathcal{S}(c_1 a_e(t) - c_2 a_i(t) + I_{ext_e}(t))
$$

$$
\tau_i \frac{d a_i(t)}{d t} = - a_i(t) + (k_i - r_i * a_i(t)) * \mathcal{S}(c_3 a_e(t) - c_4 a_i(t) + I_{ext_i}(t))
$$

$$
\mathcal{S}(input) = \frac{1}{1 + exp(- a(input - \theta))} - \frac{1}{1 + exp(a\theta)}
$$

下标$$x\in\{e, i\}$$表示该参数或变量对应兴奋性或抑制性的神经元群。在微分方程中，$$\tau_x$$表示神经元群的时间常数，参数$$k_x$$和$$r_x$$共同控制不应期，$$a_x$$和$$\theta_x$$分别是Sigmoid函数$$\mathcal{S}(input)$$的斜率和相位，且兴奋性和抑制性的神经元群分别收到外界输入$$I_{ext_{x}}$$。

<center><img src="../../figs/neus/codes/frunit1.PNG">	</center>

<center><img src="../../figs/neus/codes/frunit2.PNG">	</center>
