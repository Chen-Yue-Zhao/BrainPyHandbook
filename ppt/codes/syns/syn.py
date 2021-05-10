import brainpy as bp


class AMPA(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, t, TT, alpha, beta):
    ds = alpha * TT * (1 - s) - beta * s
    return ds

  def __init__(self, pre, post, conn, alpha=0.98, beta=0.18, T=0.5,
               T_duration=0.5, **kwargs):
    # parameters
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.t_last_pre_spike = -1e7 * bp.ops.ones(self.size)

    self.int_s = bp.odeint(f=self.derivative, method='exponential_euler')
    super(AMPA, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      post_id = self.post_ids[i]

      if self.pre.spike[pre_id]:
        self.t_last_pre_spike[pre_id] = _t
      TT = ((_t - self.t_last_pre_spike[pre_id])
            < self.T_duration) * self.T
      self.s[i] = self.int_s(self.s[i], _t, TT, self.alpha, self.beta)


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


run_syn(AMPA, T_duration=3.)


# two expo
class Two_exponentials(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, x, t, tau1, tau2):
    dxdt = (-(tau1 + tau2) * x - s) / (tau1 * tau2)
    dsdt = x
    return dsdt, dxdt

  def __init__(self, pre, post, conn, tau1=1.0, tau2=3.0, **kwargs):
    # parameters
    self.tau1 = tau1
    self.tau2 = tau2

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.x = bp.ops.zeros(self.size)

    self.integral = bp.odeint(f=self.derivative, method='rk4')

    super(Two_exponentials, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]

      self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t,
                                           self.tau1, self.tau2)
      self.x[i] += self.pre.spike[pre_id]


run_syn(Two_exponentials, tau1=2.)





# alpha
class Alpha(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, x, t, tau):
    dxdt = (-2 * tau * x - s) / (tau ** 2)
    dsdt = x
    return dsdt, dxdt

  def __init__(self, pre, post, conn, tau=3.0, **kwargs):
    # parameters
    self.tau = tau

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.x = bp.ops.zeros(self.size)

    self.integral = bp.odeint(f=self.derivative, method='rk4')

    super(Alpha, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]

      self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t,
                                           self.tau)
      self.x[i] += self.pre.spike[pre_id]


run_syn(Alpha)


# expo

class Exponential(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, t, tau):
    ds = -s / tau
    return ds

  def __init__(self, pre, post, conn, tau=8.0, **kwargs):
    # parameters
    self.tau = tau

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)

    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

    super(Exponential, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]

      self.s[i] = self.integral(self.s[i], _t, self.tau)
      self.s[i] += self.pre.spike[pre_id]


run_syn(Exponential)


# Vj
# only s
class Voltage_jump(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  def __init__(self, pre, post, conn, **kwargs):
    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)

    super(Voltage_jump, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      self.s[i] = self.pre.spike[pre_id]


run_syn(Voltage_jump)


# update V
class Voltage_jump(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  def __init__(self, pre, post, conn, delay=0., post_refractory=False,
               weight=1., **kwargs):
    # parameters
    self.delay = delay
    self.post_has_refractory = post_refractory

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.w = bp.ops.ones(self.size) * weight
    self.I_syn = self.register_constant_delay('I_syn', size=self.size,
                                              delay_time=delay)

    super(Voltage_jump, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      self.s[i] = self.pre.spike[pre_id]

      # output
      post_id = self.post_ids[i]

      self.I_syn.push(i, self.s[i] * self.w[i])
      I_syn = self.I_syn.pull(i)

      if self.post_has_refractory:
        self.post.V += I_syn * (1. - self.post.refractory[post_id])
      else:
        self.post.V += I_syn


run_syn(Voltage_jump)


def __init__(self, pre, post, conn, delay, **kwargs):
  # ...
  self.s = bp.ops.zeros(self.size)
  self.w = bp.ops.ones(self.size) * .2
  self.I_syn = self.register_constant_delay('I_syn', size=self.size,
                                            delay_time=delay)


def update(self, _t):
  for i in range(self.size):
    # ...
    self.I_syn.push(i, self.w[i] * self.s[i])
    self.post.input[post_id] += self.I_syn.pull(i)


def __init__(self, pre, post, conn, g_max, E, delay, **kwargs):
  self.g_max = g_max
  self.E = E
  # ...
  self.s = bp.ops.zeros(self.size)
  self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)


def update(self, _t):
  for i in range(self.size):
    # ...
    self.g.push(i, self.g_max * self.s[i])
    self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] -
                                                  self.E)