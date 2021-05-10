import brainpy as bp
import numpy as np
import brainmodels as bm
import matplotlib.pyplot as plt

bp.backend.set(backend='numpy', dt=0.1)
bm.set_backend(backend='numpy')


class BCM(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(w, t, lr, r_pre, r_post, r_th):
    dwdt = lr * r_post * (r_post - r_th) * r_pre
    return dwdt

  def __init__(self, pre, post, conn, lr=0.005, w_max=1., w_min=0., **kwargs):
    # parameters
    self.lr = lr
    self.w_max = w_max
    self.w_min = w_min
    self.dt = bp.backend.get_dt()

    # connections
    self.conn = conn(pre.size, post.size)
    self.conn_mat = conn.requires('conn_mat')
    self.size = bp.ops.shape(self.conn_mat)

    # variables
    self.w = bp.ops.ones(self.size) * .5
    self.sum_post_r = bp.ops.zeros(post.size[0])
    self.r_th = bp.ops.zeros(post.size[0])

    self.int_w = bp.odeint(f=self.derivative, method='rk4')

    super(BCM, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    # update threshold
    self.sum_post_r += self.post.r
    r_th = self.sum_post_r / (_t / self.dt + 1)
    self.r_th = r_th

    # resize to matrix
    w = self.w * self.conn_mat
    dim = self.size
    r_th = np.vstack((r_th,) * dim[0])
    r_post = np.vstack((self.post.r,) * dim[0])
    r_pre = np.vstack((self.pre.r,) * dim[1]).T

    # update w
    w = self.int_w(w, _t, self.lr, r_pre, r_post, r_th)
    self.w = np.clip(w, self.w_min, self.w_max)

    # output
    self.post.r = np.sum(w.T * self.pre.r, axis=1)


class neu(bp.NeuGroup):
  target_backend = ['numpy', 'numba']

  def __init__(self, size, **kwargs):
    self.r = np.zeros(size)
    super(neu, self).__init__(size=size, **kwargs)

  def update(self, _t):
    self.r = self.r


# create input
group1, _ = bp.inputs.constant_current(([3., 1], [0, 1]) * 10)
group2, duration = bp.inputs.constant_current(([0, 1], [1., 1]) * 10)
input_r = np.vstack((group1, group2))

# simulate
pre = neu(2, monitors=['r'])
post = neu(1, monitors=['r'])
bcm = BCM(pre=pre, post=post, conn=bp.connect.All2All(), lr=0.01,
          monitors=['w', 'r_th'])
net = bp.Network(pre, bcm, post)
net.run(duration, inputs=(pre, 'r', input_r.T, "="))

# plot
fig, gs = bp.visualize.get_figure(3, 1, 2, 6)

fig.add_subplot(gs[0, 0], xlim=(0, duration))
plt.plot(net.ts, pre.mon.r[:, 0], 'b', label='r1')
plt.plot(net.ts, pre.mon.r[:, 1], 'r', label='r2')
plt.title("inputs")
plt.legend()

fig.add_subplot(gs[1, 0], xlim=(0, duration))
plt.plot(net.ts, post.mon.r[:, 0], label='r_post')
plt.plot(net.ts, bcm.mon.r_th[:, 0], label='r_th')
plt.title("response")
plt.legend()

fig.add_subplot(gs[2, 0], xlim=(0, duration), ylim=(0, bcm.w_max))
plt.plot(net.ts, bcm.mon.w[:, 0], 'b', label='w1')
plt.plot(net.ts, bcm.mon.w[:, 1], 'r', label='w2')
plt.title("weights")
plt.legend()
plt.xlabel("t")
plt.show()


## print version

class BCM(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(w, t, lr, r_pre, r_post, r_th):
    dwdt = lr * r_post * (r_post - r_th) * r_pre
    return dwdt

  def __init__(self, pre, post, conn, lr=0.005, w_max=1., w_min=0., **kwargs):
    # parameters
    self.lr = lr
    self.w_max = w_max
    self.w_min = w_min
    self.dt = bp.backend.get_dt()

    # connections
    self.conn = conn(pre.size, post.size)
    self.conn_mat = conn.requires('conn_mat')
    self.size = bp.ops.shape(self.conn_mat)

    # variables
    self.w = bp.ops.ones(self.size) * .5
    self.sum_post_r = np.zeros(post.size[0])

    self.int_w = bp.odeint(f=self.derivative, method='rk4')

    super(BCM, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    # update threshold
    self.sum_post_r += self.post.r
    r_th = self.sum_post_r / (_t / self.dt + 1)
    self.r_th = r_th

    # resize to matrix
    w = self.w * self.conn_mat
    dim = self.size
    r_th = np.vstack((r_th,) * dim[0])
    r_post = np.vstack((self.post.r,) * dim[0])
    r_pre = np.vstack((self.pre.r,) * dim[1]).T

    # update w
    w = self.int_w(w, _t, self.lr, r_pre, r_post, r_th)
    self.w = np.clip(w, self.w_min, self.w_max)

    # output
    self.post.r = np.sum(w.T * self.pre.r, axis=1)


class neu(bp.NeuGroup):
  target_backend = ['numpy', 'numba']

  def __init__(self, size, **kwargs):
    self.r = np.zeros(size)
    super(neu, self).__init__(size=size, **kwargs)

  def update(self, _t):
    self.r = self.r

# create input
group1, _ = bp.inputs.constant_current(([1.5, 1],
                                        [0, 1]) * 10)
group2, duration = bp.inputs.constant_current(([0, 1],
                                               [1., 1]) * 10)
group1 = np.vstack(((group1,) * 10))
group2 = np.vstack(((group2,) * 10))
input_r = np.vstack((group1, group2))

# simulate
pre = neu(20, monitors=['r'])
post = neu(1, monitors=['r'])
bcm = BCM(pre=pre, post=post, conn=bp.connect.All2All(),
          monitors=['w'])
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