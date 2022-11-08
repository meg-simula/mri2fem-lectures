# Install Docker and download fenicsproject launch script
# fenicsproject run dev # Launch Docker
# python3 biot_example.py

from dolfin import *

# Define mesh
n = 8
mesh = UnitSquareMesh(n, n)
d = mesh.topology().dim()

# Define VxQ (mixed) element
cell = mesh.ufl_cell()
V = VectorElement("CG", cell, 2)
Q = FiniteElement("CG", cell, 1)
M = MixedElement([V, Q])

# Define mixed function space,
# test- and trial functions
W = FunctionSpace(mesh, M)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Material parameters
mu = Constant(1.0)
lmbda = Constant(100.0)
alpha = Constant(1.0)
s = Constant(0.1)
kappa = Expression("sin(pi*x[0]) + 0.1", degree=2)

# Sources
time = Constant(0.0)
f = Expression(("A*t", "0.0"), t=time, A=1.0, degree=1)
g = Constant(0.0)
time.assign(1.0)

# Previous solutions
up_ = Function(W)
(u_, p_) = split(up_)

# Define strain and stress
eps = lambda u : sym(grad(u))
I = Identity(d)
sigma = lambda u : 2*mu*eps(u) + lmbda*div(u)*I

# Define variational forms
a = (inner(sigma(u), eps(v)) + alpha*div(v)*p
     + alpha*div(u)*q - s*p*q - inner(kappa*grad(p), grad(q)))*dx()
L = (dot(f, v) + g*q + s*p_*q + alpha*div(u_)*q)*dx()
     
# Boundary conditions
bc0 = DirichletBC(W.sub(0), (0.0, )*d, "on_boundary")
bc1 = DirichletBC(W.sub(1), 0.0, "on_boundary")
bcs = [bc0, bc1]

# Assemble and apply boundary conditions
A = assemble(a)
b = assemble(L)
for bc in bcs:
    bc.apply(A, b)

# Solve it
up = Function(W)
solve(A, up.vector(), b)
(u, p) = up.split(deepcopy=True)
print(u.vector().get_local())
