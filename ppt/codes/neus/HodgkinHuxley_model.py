import brainpy as bp
from numba import prange

class HH(bp.NeuGroup):
  target_backend = 'general'

  @staticmethod
  @bp.odeint(method='exponential_euler')
  def integral(V, m, h, n, t, C, gNa, ENa, gK, EK, gL, EL, Iext):
    alpha_m = 0.1*(V+40)/(1-bp.ops.exp(-(V+40)/10))
    beta_m = 4.0*bp.ops.exp(-(V+65)/18)
    dmdt = alpha_m * (1 - m) - beta_m * m

    alpha_h = 0.07*bp.ops.exp(-(V+65)/20)
    beta_h = 1/(1+bp.ops.exp(-(V+35)/10))
    dhdt = alpha_h * (1 - h) - beta_h * h

    alpha_n = 0.01*(V+55)/(1-bp.ops.exp(-(V+55)/10))
    beta_n = 0.125*bp.ops.exp(-(V+65)/80)
    dndt = alpha_n * (1 - n) - beta_n * n

    I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
    I_K = (gK * n ** 4.0) * (V - EK)
    I_leak = gL * (V - EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / C

    return dVdt, dmdt, dhdt, dndt

  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36.,
               EL=-54.387, gL=0.03, V_th=20., C=1.0, **kwargs):
    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

    # variables
    num = bp.size2len(size)
    self.V = -65. * bp.ops.ones(num)
    self.m = 0.5 * bp.ops.ones(num)
    self.h = 0.6 * bp.ops.ones(num)
    self.n = 0.32 * bp.ops.ones(num)
    self.spike = bp.ops.zeros(num, dtype=bool)
    self.input = bp.ops.zeros(num)

    super(HH, self).__init__(size=size, **kwargs)

  def update(self, _t):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                               self.C, self.gNa, self.ENa, self.gK,
                               self.EK, self.gL, self.EL, self.input)
    self.spike = (self.V < self.V_th) * (V >= self.V_th)
    self.V = V
    self.m = m
    self.h = h
    self.n = n
    self.input[:] = 0

import brainpy as bp

dt = 0.1
bp.backend.set('numpy', dt=dt)
neu = HH(100, monitors=['V', 'spike'])
neu.t_refractory = 5.
net = bp.Network(neu)
net.run(duration=200., inputs=(neu, 'input', 21.), report=True)
fig, gs = bp.visualize.get_figure(1, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V,
                       xlabel="t", ylabel="V", show=True)
