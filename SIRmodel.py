import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D

# ODE's

def SIR_model(y, t, beta, gamma):
    S, I, R = y

    dS_dt = - beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I

    return ([dS_dt, dI_dt, dR_dt])


def SIR_model_vitalDynamics(y, t, beta, gamma, mu):
    S, I, R = y

    dS_dt = mu - mu * S - beta * S * I
    dI_dt = beta * S * I - gamma * I - mu * I
    dR_dt = gamma * I - mu * R

    return ([dS_dt, dI_dt, dR_dt])

# Initial conditions

S0 = 0.99
I0 = 1 - S0
R0 = 0.0

# Model parameters

r0 = 4.4
gamma = 1.0 / 3.0 # recovery rate /time
beta = r0 * gamma # infection rate /time
mu = 1 / 240 # Population turnover rate

# Plotting parameters
rho = gamma / beta
Imax = rho * np.log(rho) - rho + I0 + S0 - rho * np.log(S0)

# Time vector
t = np.linspace(0, 500, 10000)

# ----------------------------------- SEIRS without vital dynamics ------------------------------------ #

# Solution
sol = scipy.integrate.odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))
sol = np.array(sol)

suscept = sol[:, 0]
infected = sol[:, 1]

# Plot results

fig = plt.figure(figsize=(15, 5))
plt.plot(t, sol[:, 0], label=" Susceptible ", color="red")
plt.plot(t, sol[:, 1], label=" Infected ", color="black")
plt.plot(t, sol[:, 2], label=" Recovered ", color="blue")
plt.xlim(0, 40)
plt.rcParams.update({'font.size': 10})
plt.legend(loc='best', fontsize=10)
plt.grid(False)
plt.xlabel("Time", fontsize=10)
plt.ylabel("Proportions", fontsize=10)
plt.savefig("det_SIR_model.png")
plt.show()
plt.close()

# ---------------------------- phase plane 1 ------------------------------ #

fig = plt.figure(figsize=(21, 6))
fig.subplots_adjust(wspace=0.5, hspace=0.3)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


for S0_loop in range(0, 100, 10):
    S0_loop = S0_loop / 100
    I0_loop = 1 - S0_loop
    sol = scipy.integrate.odeint(SIR_model, [S0_loop, I0_loop, R0], t, args=(beta, gamma))
    sol = np.array(sol)
    ax1.plot(sol[:, 0], sol[:, 1], color="grey", alpha=0.3)
ax1.plot([0, 1], [1, 0], color="black", linewidth=0.3)
ax1.plot(suscept, infected, color="midnightblue")
ax1.set_xlabel(" Susceptible fraction ", fontsize=10)
ax1.set_ylabel(" Infected fraction ", fontsize=10)
ax1.xaxis.set_ticks(np.arange(0, 1, step=0.2))
ax1.yaxis.set_ticks(np.arange(0, 1, step=0.2))
plt.rcParams.update({'font.size': 10})
ax1.axvline(x=rho, ymax=Imax, color="black", linestyle="dotted", linewidth=1)
ax1.axhline(y=Imax, xmin=0, xmax=rho, color="black", linestyle="dotted", linewidth=1)
ax1.text(rho - 0.05, -0.1, r'$\rho = \frac{\gamma}{\beta} $', fontsize=10, color="midnightblue")
ax1.text(-0.08, Imax + 0.03, r'$I_{max}$', fontsize=10, color="midnightblue")
ax1.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax1.set_title(r"Phase portrait (varying $S(0)$)", fontsize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True)


for beta_loop in range(0, 20, 1):
    beta_loop = beta_loop / 10
    sol = scipy.integrate.odeint(SIR_model, [S0, I0, R0], t, args=(beta_loop, gamma))
    sol = np.array(sol)
    ax2.plot(sol[:, 1], 'r-', label=" I(t) ", alpha=0.3, color="grey")
ax2.plot(infected, 'r-', label=" I(t) ", color="midnightblue", linewidth=1)
plt.rcParams.update({'font.size': 10})
ax2.set_xlabel('Time /days', fontsize=10)
ax2.set_ylabel('Number (1000s)', fontsize=10)
ax2.yaxis.set_tick_params(length=0)
ax2.xaxis.set_tick_params(length=0)
ax2.grid(True)
ax2.set_xlim(0, 1500)
ax2.set_title(r'Dynamics in time (with varying $\beta$)', fontsize=10)
plt.savefig("phase_SIR_model.png")
plt.show()

# --------------------------------- SEIRS with vital dynamics --------------------------------- #

# Solution
sol = scipy.integrate.odeint(SIR_model_vitalDynamics, [S0, I0, R0], t, args=(beta, gamma, mu))
sol = np.array(sol)

suscept = sol[:, 0]
infected = sol[:, 1]
recovered = sol[:, 2]

# Plot results

fig = plt.figure(figsize=(15, 5))
plt.plot(t, sol[:, 0], label=" Susceptible ", color="red")
plt.plot(t, sol[:, 1], label=" Infected ", color="grey")
plt.plot(t, sol[:, 2], label=" Recovered ", color="blue")
plt.legend(loc='best', fontsize=10)
plt.grid(False)
plt.xlabel("Time", fontsize=10)
plt.ylabel("Proportions", fontsize=10)
plt.savefig("det_SIR_model_vital.png")
plt.show()

# ---------------------------- phase plane 1 ------------------------------ #

fig = plt.figure(figsize=(20, 6))
fig.subplots_adjust(wspace=0.5, hspace=0.3)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(suscept, infected, color="midnightblue", linewidth=1)
ax1.set_xlabel(" Susceptible fraction ", fontsize=10)
ax1.set_ylabel(" Infected fraction ", fontsize=10)
ax1.xaxis.set_ticks(np.arange(0, 1, step=0.2))
ax1.yaxis.set_ticks(np.arange(0, 1, step=0.2))
plt.rcParams.update({'font.size': 10})
ax1.axhline(y=Imax, xmin=0, xmax=rho, color="black", linestyle="dotted", linewidth=1)
ax1.axvline(x=rho, ymax=Imax, color="black", linestyle="dotted", linewidth=1)
ax1.text(rho - 0.05, -0.04, r'$\rho = \frac{\gamma}{\beta} $', fontsize=10, color="midnightblue")
ax1.text(-0.08, Imax, r'$I_{max}$', fontsize=10, color="midnightblue")
ax1.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax1.set_title(r'Phase Portrait $\beta$ = {0:.2f}'.format(beta), fontsize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.5)
ax1.grid(True)


for S0_loop in range(0, 100, 10):
    S0_loop = S0_loop / 100
    I0_loop = 1 - S0_loop
    sol = scipy.integrate.odeint(SIR_model_vitalDynamics, [S0_loop, I0_loop, R0], t, args=(beta, gamma, mu))
    sol = np.array(sol)
    ax2.plot(sol[:, 0], sol[:, 1], color="grey", alpha=0.3)
ax2.plot([0, 1], [1, 0], color="black", linewidth=0.3)
ax2.plot(suscept, infected, color="midnightblue")
ax2.set_xlabel(" Susceptible fraction ", fontsize=10)
ax2.set_ylabel(" Infected fraction ", fontsize=10)
ax2.xaxis.set_ticks(np.arange(0, 1, step=0.2))
ax2.yaxis.set_ticks(np.arange(0, 1, step=0.2))
plt.rcParams.update({'font.size': 10})
ax2.axvline(x=rho, ymax=Imax, color="black", linestyle="dotted", linewidth=1)
ax2.axhline(y=Imax, xmin=0, xmax=rho, color="black", linestyle="dotted", linewidth=1)
ax2.text(rho - 0.05, -0.1, r'$\rho = \frac{\gamma}{\beta} $', fontsize=10, color="midnightblue")
ax2.text(-0.08, Imax + 0.03, r'$I_{max}$', fontsize=10, color="midnightblue")
ax2.text(0.72, 0.85, '$R_0$ = {0:.2f}'.format(r0), fontsize=10, color="midnightblue")
ax2.set_title(r"Phase portrait (varying $S(0)$)", fontsize=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True)
plt.savefig("phase_SIR_model_vital.png")
plt.show()
plt.close()


# ---------------------------- 3D plot ------------------------------ #

fig = plt.figure()
ax = Axes3D(fig=fig)

for S0_loop in range(0, 100, 10):
    S0_loop = S0_loop / 100
    I0_loop = 1 - S0_loop
    sol = scipy.integrate.odeint(SIR_model_vitalDynamics, [S0_loop, I0_loop, R0], t, args=(beta, gamma, mu))
    sol = np.array(sol)
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], color="grey", alpha=0.3)
ax.plot(suscept, infected, recovered, color="midnightblue")
plt.rcParams.update({'font.size': 10})
ax.set_xlabel(" Susceptible fraction ", fontsize=10)
ax.set_ylabel(" Infected fraction ", fontsize=10)
ax.set_zlabel(" Recovered ", fontsize=10)
ax.set_title("3D Phase Portrait", fontsize=10)
ax.grid(False)
plt.savefig("phase_SIR_model_vital_3D.png")
plt.show()
plt.close()
