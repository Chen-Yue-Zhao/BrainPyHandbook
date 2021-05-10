from collections import OrderedDict
import brainpy as bp

bp.backend.set(backend='numba', dt=0.1)


class Decision(bp.NeuGroup):
  target_backend = ['numpy', 'numba']

  @staticmethod
  def derivative(s1, s2, t, I, coh,
                 JAext, J_rec, J_inh, I_0,
                 a, b, d, tau_s, gamma):
    I1 = JAext * I * (1. + coh)
    I2 = JAext * I * (1. - coh)

    I_syn1 = J_rec * s1 - J_inh * s2 + I_0 + I1
    r1 = (a * I_syn1 - b) / (1. - bp.ops.exp(-d * (a * I_syn1 - b)))
    ds1dt = - s1 / tau_s + (1. - s1) * gamma * r1

    I_syn2 = J_rec * s2 - J_inh * s1 + I_0 + I2
    r2 = (a * I_syn2 - b) / (1. - bp.ops.exp(-d * (a * I_syn2 - b)))
    ds2dt = - s2 / tau_s + (1. - s2) * gamma * r2

    return ds1dt, ds2dt

  def __init__(self, size, coh, JAext=.00117, J_rec=.3725, J_inh=.1137,
               I_0=.3297, a=270., b=108., d=0.154, tau_s=.06, gamma=0.641,
               **kwargs):
    # parameters
    self.coh = coh
    self.JAext = JAext
    self.J_rec = J_rec
    self.J_inh = J_inh
    self.I0 = I_0
    self.a = a
    self.b = b
    self.d = d
    self.tau_s = tau_s
    self.gamma = gamma

    # variables
    self.s1 = bp.ops.ones(size) * .06
    self.s2 = bp.ops.ones(size) * .06
    self.input = bp.ops.zeros(size)

    self.integral = bp.odeint(f=self.derivative, method='rk4', dt=0.01)

    super(Decision, self).__init__(size=size, **kwargs)

  def update(self, _t):
    for i in range(self.size):
      self.s1[i], self.s2[i] = self.integral(self.s1[i], self.s2[i], _t,
                                             self.input[i], self.coh,
                                             self.JAext, self.J_rec,
                                             self.J_inh, self.I0,
                                             self.a, self.b, self.d,
                                             self.tau_s, self.gamma)
      self.input[i] = 0.


def phase_analyze(I, coh):
  decision = Decision(1, coh=coh)

  phase = bp.analysis.PhasePlane(decision.integral,
                                 target_vars=OrderedDict(s2=[0., 1.],
                                                         s1=[0., 1.]),
                                 fixed_vars=None,
                                 pars_update=dict(I=I, coh=coh,
                                                  JAext=.00117, J_rec=.3725,
                                                  J_inh=.1137, I_0=.3297,
                                                  a=270., b=108., d=0.154,
                                                  tau_s=.06, gamma=0.641),
                                 numerical_resolution=.001,
                                 options={'escape_sympy_solver': True})

  phase.plot_nullcline()
  phase.plot_fixed_point()
  phase.plot_vector_field(show=True)


# no input
phase_analyze(I=0., coh=0.)

# coherence = 0%
phase_analyze(I=30., coh=0.)

# coherence = 51.2%
phase_analyze(I=30., coh=0.512)

# coherence = 100%
phase_analyze(I=30., coh=1.)
