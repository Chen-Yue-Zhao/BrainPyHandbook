import brainpy as bp


class NMDA(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, x, t, tau_rise, tau_decay, a):
    dsdt = -s / tau_decay + a * x * (1 - s)
    dxdt = -x / tau_rise
    return dsdt, dxdt

  def __init__(self, pre, post, conn, delay=0., g_max=0.15, E=0., cc_Mg=1.2,
               alpha=0.062, beta=3.57, tau=100, a=0.5, tau_rise=2., **kwargs):
    # parameters
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.cc_Mg = cc_Mg
    self.tau = tau
    self.tau_rise = tau_rise
    self.a = a
    self.delay = delay

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.x = bp.ops.zeros(self.size)
    self.g = self.register_constant_delay('g', size=self.size,
                                          delay_time=delay)

    self.integral = bp.odeint(f=self.derivative, method='rk4')

    super(NMDA, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      post_id = self.post_ids[i]

      self.x[i] += self.pre.spike[pre_id]
      self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t,
                                           self.tau_rise, self.tau,
                                           self.a)

      # output
      g_inf_exp = bp.ops.exp(-self.alpha * self.post.V[post_id])
      g_inf = 1 + g_inf_exp * self.cc_Mg / self.beta

      self.g.push(i, self.g_max * self.s[i] / g_inf)

      I_syn = self.g.pull(i) * (self.post.V[post_id] - self.E)
      self.post.input[post_id] -= I_syn


import brainmodels as bm

bp.backend.set(backend='numba', dt=0.1)


def run_syn(syn_model, **kwargs):
  neu1 = bm.neurons.LIF(2, monitors=['V'])
  neu2 = bm.neurons.LIF(3, monitors=['V'])

  syn = syn_model(pre=neu1, post=neu2, conn=bp.connect.All2All(),
                  monitors=['s'], **kwargs)

  net = bp.Network(neu1, syn, neu2)
  net.run(30., inputs=(neu1, 'input', 35.))
  bp.visualize.line_plot(net.ts, syn.mon.s, ylabel='s', show=True)


run_syn(NMDA)
