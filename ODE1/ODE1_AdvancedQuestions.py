import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

""" FRF """
k = 5
m = 2
gamma = 0.45
f = 1
x0, v0 = [0, 0]
naturalFrequency = math.sqrt(k/m)
zeta = gamma / (2*math.sqrt(k*m))

def ForcedOscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-k*x - gamma*v + f*math.cos(omega*t)) / m
    return [dxdt, dvdt]

# Creating a logarithmicly spaced points across frequency range
omegaVector = np.logspace(-2, 2, num=25)

# Initialise storage vectors for results
amplitudesVector = np.zeros_like(omegaVector)
theoryAmplitudesVector = np.zeros_like(omegaVector)

for i, omega in enumerate(omegaVector):
    period = 2 * np.pi / omega  # Compute period of force
    maxTime = max(300, 10 * period)  # Simulate the behaviour for 10 periods

    # Solve the system
    solution = solve_ivp(ForcedOscillator, (0, maxTime), [x0, v0])
    x, y = solution.y
    
    # Extracting the steady state amplitude peaks
    steadyStart = int(0.5*len(x))
    xSteady = x[steadyStart:]
    peaksIdx, _ = find_peaks(xSteady)
    peakAmplitudes = xSteady[peaksIdx]

    # Computing the median of these amplitudes
    medianAmplitude = np.median(peakAmplitudes)
    amplitudesVector[i] = medianAmplitude

    # Computing the theoretical response
    bigOmega = omega / naturalFrequency
    denominator = (1 - bigOmega**2)**2 + (2*zeta*bigOmega)**2
    theoryAmplitudesVector[i] = (1/k)*math.sqrt(1/denominator)
    print(f'Done with {omega}')

# Plotting
plt.loglog(omegaVector, amplitudesVector, label='Simulated',linewidth=4)
plt.loglog(omegaVector, theoryAmplitudesVector, label='Theoretical', linewidth=2)
plt.xlabel('Frequency (Rad/s)')
plt.ylabel('A / f')
plt.title('Steady State Amplitude Ratio (Forced Oscillator)')
plt.legend()
plt.show()

""" Diatomic Molecules """
m = 1  # Mass of each particle
k = 1  # Spring stifness
l = 2  # Undeformed length of spring

# Initial conditions
x0 = np.array([1, 1, 1, -1, -1, -1])  # Initial positions (1-3 x1, 4-6 x2)
v0 = np.random.uniform(size=(len(x0))) # Initial velocities (1-3 v1, 4-6 v2)
#v0 = np.array([0.3, -0.5, 0.4, -0.3, 0.5, -0.4]) #* For cool trajectory
y0 = np.concat([x0, v0])

maxTime = 50  # Simulation time
def DiatomicDynamics(t, y):
    x1, x2 = y[0:3], y[3:6]
    v1, v2 = y[6:9], y[9:12]
    d = np.linalg.norm(x1-x2)  # Distance between particles
    forceMagnitude = k*(d-l)  # Magnitude of the force 

    # Force on each particle
    F1 = ((x2-x1) / d)*forceMagnitude
    F2 = ((x1-x2) / d)*forceMagnitude

    # Solving the state-space system
    dx1dt, dx2dt = v1, v2
    dv1dt, dv2dt = (1/m)*F1, (1/m)*F2
    totalState = np.concat([dx1dt, dx2dt, dv1dt, dv2dt])
    return totalState

solution = solve_ivp(DiatomicDynamics, (0, maxTime), y0)
# Extract quantities from solution
t = solution.t
x1 = solution.y[0:3, :]  # Position of particle 1
x2 = solution.y[3:6, :]  # Position of particle 2
v1 = solution.y[6:9, :]
v2 = solution.y[9:12, :]

# Computing convervation quantities
# Linear Momentum
totalMomentum = m*(v1+v2)
magnitudeMomentum = np.linalg.norm(totalMomentum, axis=0)

# Energy
kineticEnergy = 0.5*m*(np.square(np.linalg.norm(v1, axis=0)) + np.square(np.linalg.norm(v2, axis=0)))
distance = np.linalg.norm(x1-x2, axis=0)
potentialEnergy = 0.5*k*np.square(distance-l)
totalEnergy = kineticEnergy + potentialEnergy

# Plotting conservation quantities
fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].plot(t, magnitudeMomentum)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Total Linear Momentum')
axs[0].set_title('Consvervation of Linear Momentum')

axs[1].plot(t, totalEnergy)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Total Energy')
axs[1].set_title('Consvervation of Energy')
plt.tight_layout()
plt.show()

# Plotting the 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1[0, :], x1[1, :], x1[2, :], linewidth=2, label='Particle 1')
ax.plot(x2[0, :], x2[1, :], x2[2, :], linewidth=2, label='Particle 2')

ax.scatter(x1[0, 0], x1[1, 0], x1[2, 0], s=100, marker='o', label='Start 1')
ax.scatter(x2[0, 0], x2[1, 0], x2[2, 0], s=100, marker='o', label='Start 2')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Diatomic Molecule Trajectories')
ax.legend()

ax.grid(True, alpha=0.3)
plt.show()