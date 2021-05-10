import brainpy as bp
import numpy as np
import brainmodels as bm
import matplotlib.pyplot as plt

bp.backend.set(backend='numpy', dt=0.1)


class Oja(bp.TwoEndConn):
  target_backend = 'numpy'

  @staticmethod
  def derivative(w, t, gamma, r_pre, r_post):
    dwdt = gamma * (r_post * r_pre - r_post * r_post * w)
    return dwdt

  def __init__(self, pre, post, conn, gamma=.005, w_max=1., w_min=0., **kwargs):
    # params
    self.gamma = gamma
    self.w_max = w_max
    self.w_min = w_min
    # no delay in firing rate models

    # conns
    self.conn = conn(pre.size, post.size)
    self.conn_mat = conn.requires('conn_mat')
    self.size = bp.ops.shape(self.conn_mat)

    # data
    self.w = bp.ops.ones(self.size) * 0.05

    self.integral = bp.odeint(f=self.derivative)
    super(Oja, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    w = self.conn_mat * self.w
    self.post.r = np.sum(w.T * self.pre.r, axis=1)

    # resize to matrix
    dim = self.size
    r_post = np.vstack((self.post.r,) * dim[0])
    r_pre = np.vstack((self.pre.r,) * dim[1]).T

    self.w = self.integral(w, _t, self.gamma, r_pre, r_post)


class neu(bp.NeuGroup):
  target_backend = 'numpy'

  def __init__(self, size, **kwargs):
    self.r = bp.ops.zeros(size)
    super(neu, self).__init__(size=size, **kwargs)

  def update(self, _t):
    self.r = self.r


# create input
current1, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 3 +
                                         [(0., 20.), (0., 20.)] * 2)
current2, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
current3, _ = bp.inputs.constant_current([(2., 20.), (0., 20.)] * 5)
current_pre = np.vstack((current1, current2))
current_post = np.vstack((current3, current3))

# simulate
neu_pre = neu(2, monitors=['r'])
neu_post = neu(2, monitors=['r'])
syn = Oja(pre=neu_pre, post=neu_post, conn=bp.connect.All2All(), monitors=['w'])
net = bp.Network(neu_pre, syn, neu_post)
net.run(duration=200., inputs=[(neu_pre, 'r', current_pre.T, '='),
                               (neu_post, 'r', current_post.T)])

# plot
fig, gs = bp.visualize.get_figure(4, 1, 2, 6)

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, neu_pre.mon.r[:, 0], 'b', label='pre r1')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, neu_pre.mon.r[:, 1], 'r', label='pre r2')
plt.legend()

fig.add_subplot(gs[2, 0])
plt.plot(net.ts, neu_post.mon.r[:, 0], color='purple', label='post r')
plt.ylim([0, 4])
plt.legend()

fig.add_subplot(gs[3, 0])
plt.plot(net.ts, syn.mon.w[:, 0, 0], 'b', label='syn.w1')
plt.plot(net.ts, syn.mon.w[:, 1, 0], 'r', label='syn.w2')
plt.legend()
plt.show()
