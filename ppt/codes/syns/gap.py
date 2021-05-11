import brainpy as bp
import brainmodels as bm

bp.backend.set(backend='numba', dt=0.1)
bm.set_backend(backend='numba')


class Gap_junction(bp.TwoEndConn):
  target_backend = ['numpy', 'numba']

  def __init__(self, pre, post, conn, delay=0., k_spikelet=0.1,
               post_refractory=False, **kwargs):
    self.delay = delay
    self.k_spikelet = k_spikelet
    self.post_has_refractory = post_refractory

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # variables
    self.w = bp.ops.ones(self.size)
    self.spikelet = self.register_constant_delay('spikelet', size=self.size,
                                                 delay_time=self.delay)

    super(Gap_junction, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      post_id = self.post_ids[i]

      self.post.input[post_id] += self.w[i] * (self.pre.V[pre_id] -
                                               self.post.V[post_id])

      self.spikelet.push(i, self.w[i] * self.k_spikelet *
                         self.pre.spike[pre_id])

      out = self.spikelet.pull(i)
      if self.post_has_refractory:
        self.post.V[post_id] += out * (1. - self.post.refractory[post_id])
      else:
        self.post.V[post_id] += out


import matplotlib.pyplot as plt

neu0 = bm.neurons.LIF(1, monitors=['V'], t_refractory=0)
neu0.V = bp.ops.ones(neu0.V.shape) * -10.
neu1 = bm.neurons.LIF(1, monitors=['V'], t_refractory=0)
neu1.V = bp.ops.ones(neu1.V.shape) * -10.
syn = Gap_junction(pre=neu0, post=neu1, conn=bp.connect.All2All(),
                   k_spikelet=5.)
syn.w = bp.ops.ones(syn.w.shape) * .5

net = bp.Network(neu0, neu1, syn)
net.run(100., inputs=(neu0, 'input', 30.))

fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )

fig.add_subplot(gs[1, 0])
plt.plot(net.ts, neu0.mon.V[:, 0], label='V0')
plt.legend()

fig.add_subplot(gs[0, 0])
plt.plot(net.ts, neu1.mon.V[:, 0], label='V1')
plt.legend()
plt.show()
