import brainpy as bp
from numba import prange

class QuaIF(bp.NeuGroup):
  target_backend = 'general'

  @staticmethod
  def derivative(V, t, I_ext, V_rest, V_c, R, tau, a_0):
    dVdt = (a_0 * (V - V_rest) * (V - V_c) + R * I_ext) / tau
    return dVdt

  def __init__(self, size, V_rest=-65., V_reset=-68.,
               V_th=-30., V_c=-50.0, a_0=.07,
               R=1., tau=10., t_refractory=0., **kwargs):
    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.a_0 = a_0
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    num = bp.size2len(size)
    self.V = bp.ops.ones(num) * V_reset
    self.input = bp.ops.zeros(num)
    self.spike = bp.ops.zeros(num, dtype=bool)
    self.refractory = bp.ops.zeros(num, dtype=bool)
    self.t_last_spike = bp.ops.ones(num) * -1e7

    self.integral = bp.odeint(f=self.derivative, method='euler')
    super(QuaIF, self).__init__(size=size, **kwargs)

  def update(self, _t):
    for i in prange(self.size[0]):
      spike = 0.
      refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
      if not refractory:
        V = self.integral(self.V[i], _t, self.input[i],
                          self.V_rest, self.V_c, self.R,
                          self.tau, self.a_0)
        spike = (V >= self.V_th)
        if spike:
          V = self.V_rest
          self.t_last_spike[i] = _t
        self.V[i] = V
      self.spike[i] = spike
      self.refractory[i] = refractory or spike
      self.input[i] = 0.
      
 
dt = 0.1
bp.backend.set('numpy', dt=dt)
neu = QuaIF(100, monitors=['V', 'refractory', 'spike'])
neu.t_refractory = 5.
net = bp.Network(neu)
net.run(duration=200., inputs=(neu, 'input', 21.), report=True)
fig, gs = bp.visualize.get_figure(1, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V", show=True)
