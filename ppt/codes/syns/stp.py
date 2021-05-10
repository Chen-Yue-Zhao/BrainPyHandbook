import brainpy as bp
import brainmodels as bm
import matplotlib.pyplot as plt

bp.backend.set(backend='numba', dt=0.1)
bm.set_backend(backend='numba')


class STP(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, u, x, t, tau, tau_d, tau_f):
    dsdt = -s / tau
    dudt = - u / tau_f
    dxdt = (1 - x) / tau_d
    return dsdt, dudt, dxdt

  def __init__(self, pre, post, conn, delay=0., U=0.15, tau_f=1500.,
               tau_d=200., tau=8., **kwargs):
    # parameters
    self.tau_d = tau_d
    self.tau_f = tau_f
    self.tau = tau
    self.U = U
    self.delay = delay

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.x = bp.ops.ones(self.size)
    self.u = bp.ops.zeros(self.size)
    self.w = bp.ops.ones(self.size)
    self.I_syn = self.register_constant_delay('I_syn', size=self.size,
                                              delay_time=delay)

    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

    super(STP, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]

      self.s[i], u, x = self.integral(self.s[i], self.u[i], self.x[i], _t,
                                      self.tau, self.tau_d, self.tau_f)

      if self.pre.spike[pre_id] > 0:
        u += self.U * (1 - self.u[i])
        self.s[i] += self.w[i] * u * self.x[i]
        x -= u * self.x[i]
      self.u[i] = u
      self.x[i] = x

      # output
      post_id = self.post_ids[i]
      self.I_syn.push(i, self.s[i])
      self.post.input[post_id] += self.I_syn.pull(i)


neu1 = bm.neurons.LIF(1, monitors=['V'])
neu2 = bm.neurons.LIF(1, monitors=['V'])

# STD
syn = STP(U=0.2, tau_d=150., tau_f=2., pre=neu1, post=neu2,
          conn=bp.connect.All2All(), monitors=['s', 'u', 'x'])
net = bp.Network(neu1, syn, neu2)
net.run(100., inputs=(neu1, 'input', 28.))

# plot
fig, gs = bp.visualize.get_figure(2, 1, 3, 7)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.u[:, 0], label='u')
plt.plot(net.ts, syn.mon.x[:, 0], label='x')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label='s')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()

neu1 = bm.neurons.LIF(1, monitors=['V'])
neu2 = bm.neurons.LIF(1, monitors=['V'])

# STF
syn = STP(U=0.1, tau_d=10, tau_f=100., pre=neu1, post=neu2,
          conn=bp.connect.All2All(), monitors=['s', 'u', 'x'])
net = bp.Network(neu1, syn, neu2)
net.run(100., inputs=(neu1, 'input', 28.))

# plot
fig, gs = bp.visualize.get_figure(2, 1, 3, 7)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.u[:, 0], label='u')
plt.plot(net.ts, syn.mon.x[:, 0], label='x')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label='s')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()
