import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from scipy import signal
import csv

fig = plt.figure(figsize=(10, 8))
ax_a = fig.add_subplot(211)
plt.ylabel('Average Reward')

ax_b = fig.add_subplot(212)
plt.ylabel('Average Reward')
plt.xlabel('Episodes')

iterations = np.arange(1, 26, 1)
kernel = np.zeros((25,))
kernel[0:3] = 1
fitnesses_a = []
fitnesses_b = []
last = 0
bound = 100

with open('../es_approach/results_03_best_yet/fitnesses.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        fitnesses = row[:25]
        fitnesses = [float('%.4f'%(float(fit))) for fit in fitnesses]
        fitnesses_a = np.array(fitnesses)

with open('../es_approach/results_04_only_one_circle/fitnesses.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        fitnesses = row[:25]
        fitnesses = [float('%.4f'%(float(fit))) for fit in fitnesses]
        fitnesses_b = np.array(fitnesses)

ax_a.plot(iterations, fitnesses_a)
ax_b.plot(iterations, fitnesses_b)
fig.savefig('es_graph.png')

fig2 = plt.figure(figsize=(10, 4))
ax_c = fig2.add_subplot(111)
plt.ylabel('Average Q')
plt.xlabel('Episodes')

iterations = np.arange(0, 425, 1) * 23.5
qs = []
with open('../q_exp_01/average_qs.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        qs = row[:425]
        qs = [float('%.4f'%(float(q))) for q in qs]
        qs = np.array(qs)

ax_c.plot(iterations, qs)
fig2.savefig('dqn_graph.png')

