import brainpy as bp

class GABAb(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(R, G, t, k3, TT, k4, k1, k2):
    dRdt = k3 * TT * (1 - R) - k4 * R
    dGdt = k1 * R - k2 * G
    return dRdt, dGdt

  def __init__(self, pre, post, conn, delay=0., g_max=0.02, E=-95.,
               k1=0.18, k2=0.034, k3=0.09, k4=0.0012, kd=100., T=0.5,
               T_duration=0.3, **kwargs):
    # params
    self.g_max = g_max
    self.E = E
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.k4 = k4
    self.kd = kd
    self.T = T
    self.T_duration = T_duration

    # conns
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # data
    self.R = bp.ops.zeros(self.size)
    self.G = bp.ops.zeros(self.size)
    self.t_last_pre_spike = bp.ops.ones(self.size) * -1e7
    self.s = bp.ops.zeros(self.size)
    self.g = self.register_constant_delay('g', size=self.size,
                                          delay_time=delay)

    self.integral = bp.odeint(f=self.derivative, method='rk4')
    super(GABAb, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      post_id = self.post_ids[i]

      if self.pre.spike[pre_id]:
        self.t_last_pre_spike[i] = _t
      TT = ((_t - self.t_last_pre_spike[i]) < self.T_duration) * self.T

      self.R[i], G = self.integral(self.R[i], self.G[i], _t, self.k3,
                                   TT, self.k4, self.k1, self.k2)
      self.s[i] = G ** 4 / (G ** 4 + self.kd)
      self.G[i] = G

      self.g.push(i, self.g_max * self.s[i])
      I_syn = self.g.pull(i) * (self.post.V[post_id] - self.E)
      self.post.input[post_id] -= I_syn


import brainmodels as bm

bp.backend.set(backend='numba', dt=0.1)
bm.set_backend(backend='numba')

neu1 = bm.neurons.LIF(2, monitors=['V'])
neu2 = bm.neurons.LIF(3, monitors=['V'])

syn = GABAb(pre=neu1, post=neu2, conn=bp.connect.All2All(), monitors=['s'])

net = bp.Network(neu1, syn, neu2)
I, dur = bp.inputs.constant_current([(25, 20), (0, 1000)])
net.run(dur, inputs=(neu1, 'input', I))
bp.visualize.line_plot(net.ts, syn.mon.s, ylabel='s', show=True)
