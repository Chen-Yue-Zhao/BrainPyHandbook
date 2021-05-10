# -*- coding: utf-8 -*-
import brainpy as bp
import brainmodels
import matplotlib.pyplot as plt
import numpy as np

bp.backend.set('numba')

N_E = 500
N_I = 500
prob = 0.1

tau = 10.
V_rest = -52.
V_reset = -60.
V_th = -50.

tau_decay = 2.

neu_E = brainmodels.neurons.LIF(N_E, monitors=['spike'])
neu_I = brainmodels.neurons.LIF(N_I, monitors=['spike'])
neu_E.V = V_rest + np.random.random(N_E) * (V_th - V_rest)
neu_I.V = V_rest + np.random.random(N_I) * (V_th - V_rest)

syn_E2E = brainmodels.synapses.Exponential(pre=neu_E, post=neu_E,
                                           conn=bp.connect.FixedProb(prob=prob))
syn_E2I = brainmodels.synapses.Exponential(pre=neu_E, post=neu_I,
                                           conn=bp.connect.FixedProb(prob=prob))
syn_I2E = brainmodels.synapses.Exponential(pre=neu_I, post=neu_E,
                                           conn=bp.connect.FixedProb(prob=prob))
syn_I2I = brainmodels.synapses.Exponential(pre=neu_I, post=neu_I,
                                           conn=bp.connect.FixedProb(prob=prob))

JE = 1 / np.sqrt(prob * N_E)
JI = 1 / np.sqrt(prob * N_I)
syn_E2E.w = JE
syn_E2I.w = JE
syn_I2E.w = -JI
syn_I2I.w = -JI

net = bp.Network(neu_E, neu_I,
                 syn_E2E, syn_E2I,
                 syn_I2E, syn_I2I)
net.run(500., inputs=[(neu_E, 'input', 3.), (neu_I, 'input', 3.)], report=True)

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)
fig.add_subplot(gs[:3, 0])
bp.visualization.raster_plot(net.ts, neu_E.mon.spike)

fig.add_subplot(gs[3, 0])
rate = bp.measure.firing_rate(neu_E.mon.spike, 5.)
plt.plot(net.ts, rate)
plt.show()
