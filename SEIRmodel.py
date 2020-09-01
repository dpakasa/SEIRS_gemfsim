import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D

# ODE's

def SEIR_model(y, t, beta, gamma, alpha):
    S, E, I, R = y

    dS_dt = - beta * S * I
    dE_dt = beta * S * I - alpha * E
    dI_dt = alpha * E - gamma * I
    dR_dt = gamma * I

    return ([dS_dt, dE_dt, dI_dt, dR_dt])


def SEIR_model_vitalDynamics(y, t, beta, gamma, alpha, mu):
    S, E, I, R = y

    dS_dt = mu - mu * S - beta * S * I
    dE_dt = beta * S * I - mu * E - alpha * E
    dI_dt = alpha * E - gamma * I - mu * I
    dR_dt = gamma * I - mu * R

    return ([dS_dt, dE_dt, dI_dt, dR_dt])

# Initial conditions

S0 = 0.99
I0 = 1 - S0
R0 = 0.0
E0 = 0.0

# Model parameters

r0 = 4.4
alpha = 1.0 / 4.0 # incubation period
gamma = 1.0 / 3.0 # recovery rate /time
beta = r0 * gamma # infection rate /time
mu = 1 / 240 # Population turnover rate

r0 = beta * (alpha / (alpha + mu)) * (1 / (gamma + mu))
print('r0 = ', r0)

# Time vector
t = np.linspace(0, 500, 10000)

# ----------------------------------- SEIRS without vital dynamics ------------------------------------ #

# Solution
sol = scipy.integrate.odeint(SEIR_model, [S0, E0, I0, R0], t, args=(beta, alpha, gamma))
sol = np.array(sol)

suscept = sol[:, 0]
infected = sol[:, 2]

# Plot results

fig = plt.figure(figsize=(15, 5))
plt.plot(t, sol[:, 0], label=" Susceptible ", color="red")
plt.plot(t, sol[:, 1], label=" Exposed ", color="grey")
plt.plot(t, sol[:, 2], label=" Infected ", color="blue")
plt.plot(t, sol[:, 3], label=" Recovered ", color="black")
plt.xlim(0, 100)
plt.rcParams.update({'font.size': 10})
plt.legend(loc='best', fontsize=10)
plt.grid(False)
plt.xlabel("Time", fontsize=10)
plt.ylabel("Proportions", fontsize=10)
plt.savefig("det_SEIR_model.png")
plt.show()
plt.close()

# ---------------------------- phase plane 1 ------------------------------ #

fig = plt.figure(figsize=(20, 5))
fig.subplots_adjust(wspace=0.5, hspace=0.3)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


for S0 in range(0, 100, 5):
    S0 = S0 / 100
    I0 = 1 - S0
    sol = scipy.integrate.odeint(SEIR_model, [S0, E0, I0, R0], t, args=(beta, alpha, gamma))
    sol = np.array(sol)
    ax1.plot(sol[:, 0], sol[:, 2], color="grey", alpha=0.3)
ax1.plot([0, 1], [1, 0], color="black", linewidth=0.3)
ax1.plot(suscept, infected, color="midnightblue")
ax1.set_xlabel(" Susceptible fraction ", fontsize=10)
ax1.set_ylabel(" Infected fraction ", fontsize=10)
ax1.xaxis.set_ticks(np.arange(0, 1, step=0.2))
ax1.yaxis.set_ticks(np.arange(0, 1, step=0.2))
plt.rcParams.update({'font.size': 10})
ax1.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax1.set_title(r"Phase portrait (varying $S(0)$)", fontsize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True)

S0 = 0.99
I0 = 1 - S0
for beta_loop in range(0, 20, 1):
    beta_loop = beta_loop / 10
    sol = scipy.integrate.odeint(SEIR_model, [S0, E0, I0, R0], t, args=(beta_loop, alpha, gamma))
    sol = np.array(sol)
    ax2.plot(sol[:, 2], 'r-', label=" I(t) ", alpha=0.3, color="grey")
ax2.plot(infected, 'r-', label=" I(t) ", color="midnightblue", linewidth=1)
plt.rcParams.update({'font.size': 10})
ax2.set_xlabel('Time /days', fontsize=10)
ax2.set_ylabel('Number (1000s)', fontsize=10)
ax2.yaxis.set_tick_params(length=0)
ax2.xaxis.set_tick_params(length=0)
ax2.grid(True)
ax2.set_xlim(0, 3000)
ax2.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax2.set_title(r'Dynamics in time (with varying $\beta$)', fontsize=10)
plt.savefig("phase_SEIR_model.png")
plt.show()



# --------------------------------- SEIRS with vital dynamics --------------------------------- #

# Initial conditions

S0 = 0.99
I0 = 1 - S0
R0 = 0.0
E0 = 0.0

# Model parameters

r0 = 4.4
alpha = 1.0 / 4.0 # incubation period
gamma = 1.0 / 3.0 # recovery rate /time
beta = r0 * gamma # infection rate /time
mu = 1 / 240 # Population turnover rate
rho = gamma / beta

# Solution
sol = scipy.integrate.odeint(SEIR_model_vitalDynamics, [S0, E0, I0, R0], t, args=(beta, gamma, alpha, mu))
sol = np.array(sol)

suscept = sol[:, 0]
infected = sol[:, 2]
recovered = sol[:, 3]

# Plot results

fig = plt.figure(figsize=(15, 5))
plt.plot(t, sol[:, 0], label=" Susceptible ", color="red")
plt.plot(t, sol[:, 1], label=" Exposed ", color="grey")
plt.plot(t, sol[:, 2], label=" Infected ", color="blue")
plt.plot(t, sol[:, 3], label=" Recovered ", color="black")
plt.legend(loc='best', fontsize=10)
plt.grid(False)
plt.xlabel("Time", fontsize=10)
plt.ylabel("Proportions", fontsize=10)
plt.savefig("det_SEIR_model_vital.png")
plt.show()

# ---------------------------- phase plane 1 ------------------------------ #

fig = plt.figure(figsize=(20, 5))
fig.subplots_adjust(wspace=0.5, hspace=0.3)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(suscept, infected, color="midnightblue")
ax1.set_xlabel("Susceptible fraction", fontsize=10)
ax1.set_ylabel("Infected fraction", fontsize=10)
ax1.text(0.72, 0.14, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax1.set_title('Phase Portrait', fontsize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.2)
ax1.grid(True)


for beta_loop in range(0, 20, 1):
    beta_loop = beta_loop / 10
    sol = scipy.integrate.odeint(SEIR_model_vitalDynamics, [S0, E0, I0, R0], t, args=(beta_loop, gamma, alpha, mu))
    sol = np.array(sol)
    ax2.plot(sol[:, 2], 'r-', label=" I(t) ", alpha=0.3, color="grey")
ax2.plot(infected, 'r-', label=" I(t) ", color="midnightblue", linewidth=1)
ax2.set_xlabel('Time', fontsize=10)
ax2.set_ylabel('Infected fraction', fontsize=10)
ax2.yaxis.set_tick_params(length=0)
ax2.xaxis.set_tick_params(length=0)
ax2.grid(True)
ax2.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax2.set_title(r'Dynamics in time (with varying $\beta$)', fontsize=10)
plt.savefig("phase_SEIR_model_vital.png")
plt.show()
plt.close()

# ---------------------------- phase plane 2 ------------------------------ #

sol = scipy.integrate.odeint(SEIR_model_vitalDynamics, [S0, E0, I0, R0], t, args=(beta, gamma, alpha, mu))
sol = np.array(sol)

suscept = sol[:, 0]
exposed = sol[:, 1]
infected = sol[:, 2]
recovered = sol[:, 3]

fig, ax = plt.subplots()

for S0 in range(100, 900, 50):
    S0 = S0 / 1000
    I0 = 1 - S0
    sol = scipy.integrate.odeint(SEIR_model_vitalDynamics, [S0, E0, I0, R0], t, args=(beta, gamma, alpha, mu))
    sol = np.array(sol)
    ax.plot(sol[:, 0], sol[:, 2], color="grey", alpha=0.3)
ax.plot([0, 1], [1, 0], color="black", linewidth=0.3)
ax.plot(suscept, infected, color="midnightblue")
ax.set_xlabel("Susceptible fraction", fontsize=10)
ax.set_ylabel("Infected fraction", fontsize=10)
plt.rcParams.update({'font.size': 10})
ax.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax.set_title(r"Phase portrait (varying $S(0)$)", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True)
plt.savefig("phase_SEIR_model_vital2.png")
plt.show()
plt.close()


# ---------------------------- 3D plot ------------------------------ #

fig = plt.figure()
ax = Axes3D(fig=fig)

for mu in range(0, 100, 10):
    mu = mu / 10000
    nu = mu
    sol = scipy.integrate.odeint(SEIR_model_vitalDynamics, [S0, E0, I0, R0], t, args=(beta, gamma, alpha, mu))
    sol = np.array(sol)
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], color="grey", alpha=0.3)
ax.plot(suscept, exposed, infected, color="midnightblue")
plt.rcParams.update({'font.size': 10})
ax.set_xlabel(" Susceptible", fontsize=10)
ax.set_ylabel(" Exposed", fontsize=10)
ax.set_zlabel(" Infected", fontsize=10)
ax.set_title("SEIR 3D Phase Portrait", fontsize=10)
ax.grid(False)
plt.savefig("phase_SEIR_model_vital_3D.png")
plt.show()
plt.close()
