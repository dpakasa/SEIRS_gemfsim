from GEMFsim.GEMFsim import *
from time import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# Set numpy array output options:
np.set_printoptions(precision=4, linewidth=65, suppress=True,
                    threshold=10)

# ----------------------------------------------- SEIRS ----------------------------------------------- #
# Model parameters
n = 500
p = 0.05
radius = 0.112

# Contact Network G
G = nx.powerlaw_cluster_graph(n=n, m=5, p=p)
G1 = nx.random_geometric_graph(n=n, radius=radius)
G2 = nx.barabasi_albert_graph(n=n, m=5)
G3 = nx.erdos_renyi_graph(n=n, p=p)

# Node position dictionary
pos = nx.get_node_attributes(G, 'pos')
# number of compartments
M = 4
print("Number of compartments:", M)
# The neighborhood of each node is determined by a contact network N.
N = G.number_of_nodes()
# Time horizon
T_final = 3
nper = 60
print("Number of periods:", nper)
# The chosen time step
step = .1
print("Step size:", step)
# Number of initially infected agents (nodes)
Init_inf = 2
print("Initial infected (MC): ", Init_inf)
# n is the entire population size
n = n
nsim = 800
# Population: NJ individuals randomly chosen to be in compartment J
NJ = 2
print("Population size: ", n)
print("Number of simulations: ", nsim)
# Fixed initial infected pop
print("Initial infected:", NJ)
# Infection process of a node with one infected neighbor is a Poisson process with transition rate β
# transmission/infection rate (probability of getting infected on contact in unit time δt)
# In the compartmental model, beta is the transmission rate for the entire compartment beta*(K)
# Can also represent the frequency of contacts between nodes/agents
beta = 0.22
# Recovery process with curing rate δ (probability of recovery in unit time δt)
delta = 0.18
print("Delta:", delta)
# Transition rate λ from exposed state to the infected state
Lambda = 1/4
print("Lambda:", Lambda)
# Transition rate from recovered to susceptible
tau = 1/17
print("Tau:", tau)
# Basic reproduction number
r0 = beta / delta
print("Basic reproduction number 1:", r0)


# ------------------------------------ SEIRS Model initialization ---------------------------------- #

# Initial state for NJ number of whole population N
Para = Para_SEIRS(delta, beta, Lambda, tau)

# Net function converts graph information into graph adjacency list format
Net = NetCmbn([MyNet(G)])

# Initial condition such that two nodes are initially in the first inducer compartment
x0 = np.zeros(N)
x0 = Initial_Cond_Gen(N=N, J=Para[1][0], NJ=NJ, x0=x0)
StopCond = ['RunTime', 30]

t0 = datetime.now()
print("Simulation start time: {} ".format(t0))

# ------------------------------------ Monte Carlo ---------------------------------- #
# Monte Carlo simulation
t0 = time()
t_interval, f, H, sol = MonteCarlo_sim(Net=Net, Para=Para,
                           StopCond=StopCond, Init_inf=Init_inf,
                           M=M, step=step, nsim=nsim, N=N, x_init=x0)
t1 = time()
time = t1 - t0
print("Simulation completed in {0:.2f} seconds or {1:.2f} minutes".format(time, time/60))

# Visualizations of Histogram

if nsim <= 50:
    num_bins = 8
else:
    num_bins = 60

fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(111)

n, bins, patches = plt.hist(H, num_bins, facecolor='midnightblue', density=True,
                            edgecolor='black', linewidth=0.8, label='Obsevations', alpha=0.5,
                            histtype="bar", rwidth=0.4)
ax.legend(loc='best')
ax.grid(False, zorder=-5)
plt.rcParams.update({'font.size': 10})
ax.yaxis.set_tick_params(direction='in', length=0)
ax.xaxis.set_tick_params(direction='in', length=5)
ax.set_xlim(0, 1)
ax.set_ylabel('Number of simulations', fontsize=10)
ax.set_xlabel('Proportion of individuals removed', fontsize=10)
plt.savefig("histgram.pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig("histgram.png", bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

# Visualizations of Histogram with KDE

fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(111)

sns.distplot(H, hist=True, kde=False,
             bins=num_bins,
             hist_kws={"histtype": "bar", "linewidth": 0.8, "edgecolor":"black", "alpha":0.7, "color": "red", "label": "Observations"},
             kde_kws={"color": "midnightblue", "lw": 1, "label": "KDE"})
plt.xlim(0, max(H) + 0.2)
ax.set_ylabel('Number of simulations', fontsize=10)
ax.set_xlabel('Proportion of individuals removed', fontsize=10)
plt.savefig("histgram_kde.pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig("histgram_kde.png", bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

# Visualization of the simulation output
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)

plt.plot(t_interval, f[0, :], 'red', label='Susceptible')
plt.plot(t_interval, f[1, :], 'grey', label='Exposed')
plt.plot(t_interval, f[2, :], 'blue', label='Infected')
plt.plot(t_interval, f[3, :], 'black', label='Recovered')
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Proportions', fontsize=10)
ax.set_title(r'Dynamics in time ({0:d} iterations ($R_0$ = {1:.2f})'.format(nsim, r0), fontsize=10)
ax.yaxis.set_tick_params(direction='in', length=0)
ax.xaxis.set_tick_params(direction='out', length=5)
plt.legend(loc='upper right', shadow=False, fontsize=10)
plt.savefig("SEIRS_MonteCarlo.pdf")
plt.savefig("SEIRS_MonteCarlo.png")
plt.show()

# Visualization of the simulation output including all iterations

fig, ax = plt.subplots(figsize=(15, 5))

for i in range(0, nsim):
    plt.plot(t_interval, sol[i], color="grey", alpha=0.3, linewidth=0.3)
plt.plot(t_interval, f[2, :], color='blue')
plt.rcParams.update({'font.size': 10})
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Proportions', fontsize=10)
ax.yaxis.set_tick_params(direction='in', length=0)
ax.xaxis.set_tick_params(direction='out', length=5)
ax.grid(False)
ax.set_title('Dynamics in time (with {0:d} iterations)'.format(nsim), fontsize=10)
plt.savefig("SEIRS_MonteCarlo_sim.pdf")
plt.savefig("SEIRS_MonteCarlo_sim.png")
plt.show()


# ------------------------------------ Comparative visualization ---------------------------------- #

# Graph generator list
graph_list = [G, G1, G2, G3]
name_list = ["Power Law Cluster", "Random Geometric", "Barabasi-ALbert", "Erdos-Renyi"]
color_list = ["black", "red", "blue", "green"]

# Comparative visualization of multiple graph models
fig, ax = plt.subplots(figsize=(15, 5))

k = 0
for g in graph_list:
    Para = Para_SEIRS(delta, beta, Lambda, tau)
    Net = NetCmbn([MyNet(g)])
    x0 = np.zeros(N)
    x0 = Initial_Cond_Gen(N=N, J=Para[1][0], NJ=NJ, x0=x0)
    StopCond = ['RunTime', nper]
    t_interval, f, H, sol = MonteCarlo_sim(Net=Net, Para=Para,
                                           StopCond=StopCond, Init_inf=Init_inf,
                                           M=M, step=step, nsim=20, N=N, x_init=x0)
    plt.plot(t_interval, f[2, :], color=color_list[k], alpha=1, linewidth=1, label=name_list[k])
    k += 1
plt.rcParams.update({'font.size': 10})
plt.legend(loc='upper right', shadow=False, fontsize=10)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Proportions', fontsize=10)
ax.yaxis.set_tick_params(direction='in', length=0)
ax.xaxis.set_tick_params(direction='out', length=5)
ax.grid(False)
ax.set_title('Dynamics in time', fontsize=10)
plt.savefig("SEIRS_compare_all.pdf")
plt.savefig("SEIRS_compare_all.png")
plt.show()

# ------------------------------------ Animation ---------------------------------- #

# Contact Network G (visualization can only be implemented using graphs with node position dictionary (pos))
G = nx.random_geometric_graph(n=n, radius=radius)

# Initial state for NJ number of whole population N
Para = Para_SEIRS(delta, beta, Lambda, tau)

# Net function converts graph information into graph adjacency list format
Net = NetCmbn([MyNet(G)])

# Initial condition such that two nodes are initially in the first inducer compartment
x0 = np.zeros(N)
x0 = Initial_Cond_Gen(N=N, J=Para[1][0], NJ=NJ, x0=x0)
StopCond = ['RunTime', nper]


# Event-driven approach to simulation the process
[ts, n_index, i_index, j_index] = GEMF_SIM(Para, Net, x0, StopCond, N)

t_interval, StateCount = Post_Population(x0=x0, M=M, N=N, ts=ts, i_index=i_index, j_index=j_index)
population = StateCount

t0 = datetime.now()
print("Animation start time: {} ".format(t0))

# Visualization the contact network
fig = plt.figure(figsize=(7, 7))
comp = ['S', 'E', 'I', 'R']
colors = ['white', 'yellow', 'blue', "green"]
col = dict(zip(comp, colors))
model = [x0, n_index, i_index, j_index]
anim = animate_discrete_property_over_graph(g=G, model=model, steps=len(ts)-1, fig=fig,
                                            n_index=n_index, i_index=i_index, j_index=j_index,
                                            comp=comp, population=population, days=t_interval, property='state',
                                            color_mapping=col, colors=colors, pos=pos, Node_radius=.005)
anim.save('seirs_network.gif', writer='imagemagick', fps=30)
plt.show()
plt.close()

t0 = datetime.now()
print("Animation completed at {} ".format(t0))

# ------------------------------------ Comparative visualization (power law models) ---------------------------------- #

# Comparative visualization of multiple power law cluster graphs model
n = 500
m = 5

prob = 0
p = []
graph_list = []
for i in range(0, 4):
    p[i] = prob + 0.1
    graph_list[i] = nx.powerlaw_cluster_graph(n=n, m=m, p=p[i])
    name_list[i] = "$p$ = {}".format(p[i])

fig, ax = plt.subplots(figsize=(15, 5))

k = 0
for g in graph_list:
    Para = Para_SEIRS(delta, beta, Lambda, tau)
    Net = NetCmbn([MyNet(g)])
    x0 = np.zeros(N)
    x0 = Initial_Cond_Gen(N=N, J=Para[1][0], NJ=NJ, x0=x0)
    StopCond = ['RunTime', nper]
    t_interval, f, H, sol = MonteCarlo_sim(Net=Net, Para=Para,
                                           StopCond=StopCond, Init_inf=Init_inf,
                                           M=M, step=step, nsim=20, N=N, x_init=x0)
    plt.plot(t_interval, f[2, :], color=color_list[k], alpha=1, linewidth=1, label=name_list[k])
    k += 1
plt.rcParams.update({'font.size': 10})
plt.legend(loc='upper right', shadow=False, fontsize=10)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Proportions', fontsize=10)
ax.yaxis.set_tick_params(direction='in', length=0)
ax.xaxis.set_tick_params(direction='out', length=5)
ax.grid(False)
ax.set_title(r'Power Law Cluster with varying $p$', fontsize=10)
plt.savefig("SEIRS_compare_graph.pdf")
plt.savefig("SEIRS_compare_graph.png")
plt.show()


# ------------------------------------ GEMF one iteration ---------------------------------- #

# Event-driven approach to simulation the process
[ts, n_index, i_index, j_index] = GEMF_SIM(Para, Net, x0, StopCond, N)

t_interval, StateCount = Post_Population(x0=x0, M=M, N=N, ts=ts, i_index=i_index, j_index=j_index)
population = StateCount

# Visualization of the simulation output
fig = plt.figure(figsize=(15, 5))
plt.plot(t_interval, StateCount[0, :]/N, 'red', label='Susceptible')
plt.plot(t_interval, StateCount[1, :]/N, 'grey', label='Exposed')
plt.plot(t_interval, StateCount[2, :]/N, 'blue', label='Infected')
plt.plot(t_interval, StateCount[3, :]/N, 'black', label='Recovered')
plt.xlabel('Time (days)', fontsize=10)
plt.ylabel('Proportion of Population', fontsize=10)
plt.title('{0:d}-Period SEIRS ({1:d} iterations ($R_0$ = {2:.2f})'.format(StopCond[1], 1, r0), fontsize=10)
plt.legend(loc='upper right', shadow=False)
plt.savefig("SEIRS_compare_one_iter.pdf")
plt.savefig("SEIRS_compare_one_iter.png")
plt.show()
plt.close()