import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

def SystemEnergy(x, v):
    return 0.5* (k*x**2 + m*v**2 )

# System Parameters
k = 5.0
m = 2.0
x0, v0 = 1.0, 0.0

# Numerical parameters
deltaT = [0.0005, 0.005, 0.05]
tMax = 60

""" Euler Scheme (Questions 4-6) """
eulerResults = {}
for dt in deltaT:  # Loop over the different values of dt
    nSteps = int(tMax / dt) # Calculate number of steps
    timeVector = np.linspace(0, tMax, nSteps)  # Create timesteps

    # Initalize storage vectors
    xEuler = np.zeros_like(timeVector)
    vEuler = np.zeros_like(timeVector)

    # Setting initial conditions
    xEuler[0], vEuler[0] = x0, v0
    
    # Running simulation
    for i in range(len(timeVector)-1):
        xEuler[i+1] = xEuler[i] + vEuler[i] * dt
        vEuler[i+1] = vEuler[i] + ((-k*xEuler[i])/m) * dt
    eEuler = SystemEnergy(x=xEuler, v=vEuler)

    # Storing results
    eulerResults[str(dt)] = {
        't': timeVector,
        'x': xEuler,
        'v': vEuler,
        'e': eEuler}

# Plotting results
fig, axs = plt.subplots(1, len(eulerResults), figsize=(12, 4))
fig.suptitle('Euler')
for idx, ax in enumerate(axs):
    key = str(deltaT[idx])
    results = eulerResults[key]
    ax.plot(results['t'], results['e'])
    ax.set_title(f'Delta t: {key}s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
plt.tight_layout()
plt.show()

""" Leap Frog Scheme (Question 7-8)"""
leapfrogResults = {}
for dt in deltaT:  # Loop over the different values of dt
    nSteps = int(tMax / dt) # Calculate number of steps
    timeVector = np.linspace(0, tMax, nSteps)  # Create timesteps

    # Initalize storage vectors
    xLeap = np.zeros_like(timeVector)
    wLeap = np.zeros_like(timeVector)
    vLeap = np.zeros_like(timeVector) # For revised energy calculation

    # Setting initial conditions
    xLeap[0], wLeap[0], vLeap[0] = x0, v0, v0

    # Running simulation
    for i in range(len(timeVector)-1):
        wLeap[i+1] = wLeap[i] + ((-k*xLeap[i])/m) * dt
        xLeap[i+1] = xLeap[i] + wLeap[i+1]*dt
        if i != 0:
            vLeap[i] = (wLeap[i] + wLeap[i-1]) / 2
    vLeap[-1] = (wLeap[-1] + wLeap[-2]) / 2  # Final one since the loop runs out
    

    # Initial Energy Calculation
    eLeap = SystemEnergy(x=xLeap, v=wLeap)

    # Revised Energy Calculation
    eLeapCorrected = SystemEnergy(x=xLeap, v=vLeap)

    # Storing results
    leapfrogResults[str(dt)] = {
        't': timeVector,
        'x': xLeap,
        'w': wLeap,
        'e initial': eLeap,
        'e revised': eLeapCorrected}

# Plotting Results (original energy)
fig, axs = plt.subplots(1, len(eulerResults), figsize=(12, 4))
fig.suptitle('Leap-Frog')
for idx, ax in enumerate(axs):
    key = str(deltaT[idx])
    results = leapfrogResults[key]
    ax.plot(results['t'], results['e initial'])
    ax.set_title(f'Delta t: {key}s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
plt.tight_layout()
plt.show()

# Plotting Results (revised energy)
fig, axs = plt.subplots(1, len(eulerResults), figsize=(12, 4))
fig.suptitle('Leap-Frog(Revised Energy)')
for idx, ax in enumerate(axs):
    key = str(deltaT[idx])
    results = leapfrogResults[key]
    ax.plot(results['t'], results['e revised'])
    ax.set_title(f'Delta t: {key}s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
plt.tight_layout()
plt.show()


""" Question 10 """
gamma = 0.45
def FreeOscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-k*x) / m
    return [dxdt, dvdt]

# Running Simulation
freeSolutionIVP= solve_ivp(FreeOscillator, (0, tMax), [x0, v0])
# Extracting results
xIVPFree, vIVPFree = freeSolutionIVP.y
tIVPFree = freeSolutionIVP.t
eIVPFree = SystemEnergy(x=xIVPFree, v=vIVPFree)

# Plotting results
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle('Runge-Kutta (Free Oscillator)')
ax.plot(tIVPFree, eIVPFree)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy (J)')
plt.show()

""" Question 12a """
def DampedOscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-k*x - gamma*v) / m
    return [dxdt, dvdt]

""" Question 12b """
dt = 0.005  # dt is set to one value for leap frog
nSteps = int(tMax / dt) # Calculate number of steps
timeVector = np.linspace(0, tMax, nSteps)  # Create timesteps

# Initalize storage vectors
xLeap = np.zeros_like(timeVector)
wLeap = np.zeros_like(timeVector)

# Setting initial conditions
xLeap[0], wLeap[0] = x0, v0

# Running simulation
for i in range(len(timeVector)-1):
    wLeap[i+1] = wLeap[i] + (( -k*xLeap[i] - gamma*wLeap[i] ) / m) * dt
    xLeap[i+1] = xLeap[i] + wLeap[i+1]*dt
eLeap = SystemEnergy(x=xLeap, v=wLeap)

""" Question 13 """
# Running simulation
dampedSolutionIVP = solve_ivp(DampedOscillator, (0, tMax), [x0, v0])
# Extracting results
xIVPDamped, vIVPDamped = dampedSolutionIVP.y
tIVPDamped = dampedSolutionIVP.t
eIVPDamped = SystemEnergy(x=xIVPDamped, v=vIVPDamped)

# Plotting results
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Runge-Kutta (Damped Oscillator)')
axs[0].plot(tIVPDamped, xIVPDamped)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position (m)')

axs[1].plot(tIVPDamped, eIVPDamped)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Energy (J)')
plt.tight_layout()
plt.show()

""" Question 14 """
f = 1
omega = 4
# Runge-Kutta
def ForcedOscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-k*x - gamma*v + f*math.cos(omega*t) / m)
    return [dxdt, dvdt]

# Leap Frog
dt = 0.05  # dt is set to one value for this
nSteps = int(tMax / dt) # Calculate number of steps
timeVector = np.linspace(0, tMax, nSteps)  # Create timesteps

# Initalize storage vectors
xLeap = np.zeros_like(timeVector)
wLeap = np.zeros_like(timeVector)
# Setting initial conditions
xLeap[0], wLeap[0] = x0, v0

# Running simulation
for i in range(len(timeVector)-1):
    wLeap[i+1] = wLeap[i] + (( -k*xLeap[i] - gamma*wLeap[i] + f*math.cos(omega*timeVector[i]) ) / m) * dt
    xLeap[i+1] = xLeap[i] + wLeap[i+1]*dt
eLeap = SystemEnergy(x=xLeap, v=wLeap)

""" Question 15 """
# Initialize a result storage dictionary
forcedSolutions = {}

# Initialize 9 combinations of f and omega
fVector = [1, 2, 4]
naturalFrequency = math.sqrt(k/m)
omegaVector = [naturalFrequency/10, naturalFrequency, naturalFrequency*10]

# Looping over the combinatoins
for f in fVector:
    forcedSolutions[str(f)] = {}
    for omega in omegaVector:
        # Running simulation
        solution = solve_ivp(ForcedOscillator, (0, tMax), [x0, v0], max_step=dt)
        # Extracting results
        x, v = solution.y
        t = solution.t
        # Storing results
        forcedSolutions[str(f)][str(round(omega, 2))] = {
            't': t,
            'x': x,
            'v': v}

# Plotting results
fig, axs = plt.subplots(len(fVector), len(omegaVector), figsize=(12,8))
fig.suptitle('Runge-Kutta (Forced Oscillator)')
for rowIdx, f in enumerate(fVector):
    fKey = str(f)
    for colIdx, omega in enumerate(omegaVector):
        omegaKey = str(round(omega, 2))
        solutions = forcedSolutions[fKey][omegaKey]
        ax = axs[rowIdx, colIdx]
        ax.plot(solutions['t'], solutions['x'])
        ax.set_title(f'Amplitude: {fKey}, Frequency: {omegaKey}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
plt.tight_layout()
plt.show()
