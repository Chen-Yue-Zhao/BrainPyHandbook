import brainpy as bp
from numba import prange

class LIF(bp.NeuGroup):
  target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

  @staticmethod
  def derivative(V, t, Iext, V_rest, R, tau):
    dvdt = (-V + V_rest + R * Iext) / tau
    return dvdt

  def __init__(self, size, t_refractory=1., V_rest=0.,
               V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    num = bp.size2len(size)
    self.t_last_spike = bp.ops.ones(num) * -1e7
    self.input = bp.ops.zeros(num)
    self.spike = bp.ops.zeros(num, dtype=bool)
    self.V = bp.ops.ones(num) * V_rest

    self.integral = bp.odeint(self.derivative)
    super(LIF, self).__init__(size=size, **kwargs)

  def update(self, _t):
    for i in prange(self.size[0]):
      spike = 0.
      V = self.integral(self.V[i], _t, self.input[i],
                        self.V_rest, self.R, self.tau)
      spike = (V >= self.V_th)
      if spike:
        V = self.V_reset
        self.t_last_spike[i] = _t
      self.V[i] = V
      self.spike[i] = spike
      self.input[i] = 0.

import brainpy as bp

dt = 0.1
bp.backend.set('numpy', dt=dt)
neu = LIF(100, monitors=['V', 'refractory', 'spike'])
neu.t_refractory = 5.
net = bp.Network(neu)
net.run(duration=200., inputs=(neu, 'input', 21.), report=True)
fig, gs = bp.visualize.get_figure(1, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V", show=True)
