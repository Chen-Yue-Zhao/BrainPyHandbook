import brainpy as bp
import numpy as np
import brainmodels as bm
import matplotlib.pyplot as plt

bp.backend.set(backend='numba', dt=0.1)
bm.set_backend(backend='numba')

class BCM(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(w, t, lr, r_pre, r_post, r_th):
    dwdt = lr * r_post * (r_post - r_th) * r_pre
    return dwdt

  def __init__(self, pre, post, conn, lr=0.005, w_max=2., w_min=0., **kwargs):
    # parameters
    self.lr = lr
    self.w_max = w_max
    self.w_min = w_min
    self.dt = bp.backend.get_dt()

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.w = bp.ops.ones(self.size)
    self.sum_post_r = bp.ops.zeros(post.size[0])

    self.int_w = bp.odeint(f=self.derivative, method='rk4')

    super(BCM, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    # update threshold
    self.sum_post_r += self.post.r
    r_th = self.sum_post_r / (_t / self.dt + 1)

    # update w and post_r
    post_r = bp.ops.zeros(self.post.size[0])

    for i in range(self.size):
      pre_id = self.pre_ids[i]
      post_id = self.post_ids[i]

      post_r[post_id] += self.w[i] * self.pre.r[pre_id]

      w = self.int_w(self.w[i], _t, self.lr, self.pre.r[pre_id],
                     self.post.r[post_id], r_th[post_id])

      self.w[i] = bp.ops.clip(w, self.w_min, self.w_max)

    self.post.r = post_r

class neu(bp.NeuGroup):
  target_backend = ['numpy', 'numba']

  def __init__(self, size, **kwargs):
    self.r = bp.ops.zeros(size)
    super(neu, self).__init__(size=size, **kwargs)

  def update(self, _t):
    self.r = self.r

# create input
group1, duration = bp.inputs.constant_current(([1.5, 1], [0, 1]) * 20)
group2, duration = bp.inputs.constant_current(([0, 1], [1., 1]) * 20)
group1 = bp.ops.vstack(((group1,) * 10))
group2 = bp.ops.vstack(((group2,) * 10))
input_r = bp.ops.vstack((group1, group2))

# simulate
pre = neu(20, monitors=['r'])
post = neu(1, monitors=['r'])
bcm = BCM(pre=pre, post=post, conn=bp.connect.All2All(), monitors=['w'])
net = bp.Network(pre, bcm, post)
net.run(duration, inputs=(pre, 'r', input_r.T, "="))

# plot
fig, gs = bp.visualize.get_figure(2, 1)
fig.add_subplot(gs[1, 0], xlim=(0, duration), ylim=(0, bcm.w_max))
plt.plot(net.ts, bcm.mon.w[:, 0], 'b', label='w1')
plt.plot(net.ts, bcm.mon.w[:, 11], 'r', label='w2')
plt.title("weights")
plt.ylabel("weights")
plt.xlabel("t")
plt.legend()

fig.add_subplot(gs[0, 0], xlim=(0, duration))
plt.plot(net.ts, pre.mon.r[:, 0], 'b', label='r1')
plt.plot(net.ts, pre.mon.r[:, 11], 'r', label='r2')
plt.title("inputs")
plt.ylabel("firing rate")
plt.xlabel("t")
plt.legend()

plt.show()
