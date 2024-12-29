import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

def squared(x):
  return x*x

pow2 = make_function(function=squared, name='pow2', arity=1)
function_set = ['add', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 
                'cos', 'tan', 'div', 'sqrt', 'sub', 'mul', pow2]

# Step 1: Generate and show synthetic dataset of positive spherical quadrant surface

def generate_sphere_data(num_points, radius):
    """
    Generate synthetic dataset of positive points on the surface of a sphere.
    """
    points = []
    for _ in range(num_points):
        phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        theta = np.random.uniform(0, np.pi)    # Polar angle
        x = np.abs(radius * np.sin(theta) * np.cos(phi))
        y = np.abs(radius * np.sin(theta) * np.sin(phi))
        z = np.abs(radius * np.cos(theta))
        points.append([x, y, z])
    return np.array(points)

# Constants
radius = 1.0  # Radius of the sphere
num_points = 1000  # Number of points to generate

# Generate data
sphere_points = generate_sphere_data(num_points, radius)
X = sphere_points[:, :2]  # x and y coordinates as input
y = sphere_points[:, 2]   # z coordinate as output

# Plot the dataset
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], c='b', marker='o', alpha=0.6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Points on the Surface of a Sphere")
plt.show()

# Step 2: Utilize Genetic Programming for Symbolic Regression
# Using gplearn to rediscover governing equation

# Create a symbolic regressor
est_gp = SymbolicRegressor(
    population_size=10_000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.8,
    verbose=1,
    random_state=42,
    function_set=function_set
)

# Train the symbolic regressor
est_gp.fit(X, y)

# Step 3: Extract the Discovered Equation
print("Discovered equation:", est_gp._program)

# Step 4: Evaluate the Discovered Model
predicted_y = est_gp.predict(X)

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_y, alpha=0.6, label="Predicted vs True")
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Ideal")
plt.xlabel("True Z Values")
plt.ylabel("Predicted Z Values")
plt.title("True vs Predicted Z Values on the Sphere")
plt.legend()
plt.show()