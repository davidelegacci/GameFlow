import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import random


# Set up a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define spherical coordinates
theta = np.linspace(0, 2 * np.pi, 100)  # angle around the Z-axis
phi = np.linspace(0, np.pi, 100)        # angle from the Z-axis

random_theta = random.choice(theta)
random_phi = random.choice(phi)

theta, phi = np.meshgrid(theta, phi)

# Set radius of the sphere
r = 1

# Convert spherical coordinates to Cartesian coordinates

def sphere_parametrization(theta, phi):
	x = r * np.sin(phi) * np.cos(theta)
	y = r * np.sin(phi) * np.sin(theta)
	z = r * np.cos(phi)
	return np.array([x, y, z])


x_sphere, y_sphere, z_sphere = sphere_parametrization(theta, phi)

# Plot the surface
ax.plot_surface(x_sphere, y_sphere, z_sphere, cmap='viridis', alpha=0.1)

# Set plot limits for equal aspect ratio
ax.set_xlim([-r, r])
ax.set_ylim([-r, r])
ax.set_zlim([-r, r])


# ANY function of x only
def f(x, y, z):
	return x**2

# ANY function of y only
def g(x, y, z):
	return np.exp(y) * np.sin(y)


# Field
def field(t, var):
	x, y, z = var
	return f(x, y, z) * np.array([0, z, -y]) + g(x, y, z) * np.array([z, 0, -x])

# initial_point = 1 / np.sqrt(3) * np.array([1, 1, 1])
initial_point = sphere_parametrization(random_theta, random_phi)


initial_time = 0
final_time = 100
time_resolution = 10000
time_span = [initial_time, final_time]
time = np.linspace(initial_time, final_time, time_resolution)

sol = solve_ivp(field, t_span = time_span, y0 = initial_point, dense_output=True)
trajectory = sol.sol(time)

ax.plot(*initial_point, 'ro')

ax.plot(*trajectory)


# Display the plot
plt.show()