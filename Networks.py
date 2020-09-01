from GEMFsim.GEMFsim import *
import collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

# Set numpy array output options:
np.set_printoptions(precision=4, linewidth=65, suppress=True,
                    threshold=10)

# ----------------------------------------------- SEIRS ----------------------------------------------- #
# Model parameters

n = 500
radius = 0.145
m = 3  # Number of initial links
seed = 500

# Contact Network G
G2 = nx.powerlaw_cluster_graph(n=n, m=m, p=0.02, seed=seed)
G1 = nx.random_geometric_graph(n=n, radius=radius)
G3 = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
G = nx.erdos_renyi_graph(n=n, p=0.02)

d = dict(G.degree)

# Network visualization
pos = nx.random_layout(G)

fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Node color and size coding
ncolor = [v for v in d.values()]
e_size = [v * 2 for v in d.values()]

nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='gray', width=0.8, alpha=0.4)
sc = nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=d.keys(),
                            node_size=e_size, edgecolors='grey', cmap='jet', node_color=ncolor, alpha=0.9)
sc.set_norm(mcolors.Normalize())
fig.colorbar(sc)
ax2.grid(False)                # no grid
ax2.get_xaxis().set_ticks([])  # no ticks on the axes
ax2.get_yaxis().set_ticks([])

# Plot the degree distribution using a histogram
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

ax1.bar(deg, cnt, width=0.80, facecolor='red', edgecolor='k', label='Observations', linewidth=0.8, alpha=1)
ax1.set_ylabel("Count",fontsize=10)
ax1.set_xlabel("Degree", fontsize=10)
ax1.legend(loc='best', fontsize=10)
ax1.yaxis.set_tick_params(direction='in', length=5)
ax1.xaxis.set_tick_params(direction='in', length=0)
plt.rcParams.update({'font.size': 10})
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.tick_params(axis='both', which='minor', labelsize=8)
plt.subplots_adjust(left=0.15)
plt.savefig("network_ER.pdf")
plt.savefig("network_ER.png")
plt.show()

# Plot a simple graph with 3 nodes and 4 edges

g = nx.DiGraph()

g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 1)
g.add_edge(1, 3)

pos = nx.spring_layout(g)

fig, ax = plt.subplots(figsize=(5, 2.5))
nx.draw_networkx_edges(g, pos, ax=ax, edge_color='k', width=1, alpha=1, arrows=True, arrowsize=20, connectionstyle='arc3,rad=-0.2')
nodes = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=700, edgecolors='k', cmap='jet', node_color='blue', alpha=1)
nodes.set_edgecolor('k')
nx.draw_networkx_labels(g, pos, font_size=9, font_color="white")
ax.grid(False)                # no grid
ax.get_xaxis().set_ticks([])  # no ticks on the axes
ax.get_yaxis().set_ticks([])
plt.savefig("networkx.png")
plt.show()