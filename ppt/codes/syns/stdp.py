import brainpy as bp
import brainmodels as bm
import matplotlib.pyplot as plt

bp.backend.set(backend='numba', dt=0.1)
bm.set_backend(backend='numba')


class STDP(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s, A_s, A_t, t, tau, tau_s, tau_t):
    dsdt = -s / tau
    dAsdt = - A_s / tau_s
    dAtdt = - A_t / tau_t
    return dsdt, dAsdt, dAtdt

  def __init__(self, pre, post, conn, delay=0., delta_A_s=0.5,
               delta_A_t=0.5, w_min=0., w_max=20., tau_s=10., tau_t=10.,
               tau=10., **kwargs):
    # parameters
    self.tau_s = tau_s
    self.tau_t = tau_t
    self.tau = tau
    self.delta_A_s = delta_A_s
    self.delta_A_t = delta_A_t
    self.w_min = w_min
    self.w_max = w_max
    self.delay = delay

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.s = bp.ops.zeros(self.size)
    self.A_s = bp.ops.zeros(self.size)
    self.A_t = bp.ops.zeros(self.size)
    self.w = bp.ops.ones(self.size) * 1.
    self.I_syn = self.register_constant_delay('I_syn', size=self.size,
                                              delay_time=delay)
    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

    super(STDP, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      post_id = self.post_ids[i]

      self.s[i], A_s, A_t = self.integral(self.s[i], self.A_s[i],
                                          self.A_t[i], _t, self.tau,
                                          self.tau_s, self.tau_t)

      w = self.w[i]
      if self.pre.spike[pre_id] > 0:
        self.s[i] += w
        A_s += self.delta_A_s
        w -= A_t

      if self.post.spike[post_id] > 0:
        A_t += self.delta_A_t
        w += A_s

      self.A_s[i] = A_s
      self.A_t[i] = A_t

      self.w[i] = bp.ops.clip(w, self.w_min, self.w_max)

      # output
      self.I_syn.push(i, self.s[i])
      self.post.input[post_id] += self.I_syn.pull(i)


pre = bm.neurons.LIF(1, monitors=['spike'])
post = bm.neurons.LIF(1, monitors=['spike'])

# pre before post
duration = 300.
(I_pre, _) = bp.inputs.constant_current([(0, 5), (30, 15),  # t_pre=5
                                         (0, 15), (30, 15),
                                         (0, 15), (30, 15),
                                         (0, 98), (30, 15),  # t_pre=(t_post)+3
                                         (0, 15), (30, 15),
                                         (0, 15), (30, 15),
                                         (0, duration - 155 - 98)])
(I_post, _) = bp.inputs.constant_current([(0, 10), (30, 15),  # t_post=t_pre+5
                                          (0, 15), (30, 15),
                                          (0, 15), (30, 15),
                                          (0, 90), (30, 15),  # t_post=(t_pre)-3
                                          (0, 15), (30, 15),
                                          (0, 15), (30, 15),
                                          (0, duration - 160 - 90)])

syn = STDP(pre=pre, post=post, conn=bp.connect.All2All(), monitors=['s', 'w'])
net = bp.Network(pre, syn, post)
net.run(duration, inputs=[(pre, 'input', I_pre), (post, 'input', I_post)])

# plot
fig, gs = bp.visualize.get_figure(4, 1, 2, 7)


def hide_spines(my_ax):
  plt.legend()
  plt.xticks([])
  plt.yticks([])
  my_ax.spines['left'].set_visible(False)
  my_ax.spines['right'].set_visible(False)
  my_ax.spines['bottom'].set_visible(False)
  my_ax.spines['top'].set_visible(False)


ax = fig.add_subplot(gs[0, 0])
plt.plot(net.ts, syn.mon.s[:, 0], label="s")
hide_spines(ax)

ax1 = fig.add_subplot(gs[1, 0])
plt.plot(net.ts, pre.mon.spike[:, 0], label="pre spike")
plt.ylim(0, 2)
hide_spines(ax1)
plt.legend(loc='center right')

ax2 = fig.add_subplot(gs[2, 0])
plt.plot(net.ts, post.mon.spike[:, 0], label="post spike")
plt.ylim(-1, 1)
hide_spines(ax2)

ax3 = fig.add_subplot(gs[3, 0])
plt.plot(net.ts, syn.mon.w[:, 0], label="w")
plt.legend()
# hide spines
plt.yticks([])
ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.xlabel('Time (ms)')
plt.show()
