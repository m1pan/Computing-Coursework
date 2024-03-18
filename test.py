# In this section I am importing all the libraries I will need
import numpy as np
import matplotlib.pyplot as plt

# Expiration time in years
T = 1
# Strike price 
K = 50
# Risk-free interest rate
r = 0.05
# Volatility of stock price
sigma = 0.2


#coefficients of u
alpha = (r/(sigma**2)) - 0.5
beta = (r**2 / (2 * sigma**2)) + (r / 2) + (sigma**2 / 8)

# sigma^2 / 2
sigma2 = sigma**2 / 2


# In this section I am setting the domain of solution and the discretised grid
# The domain of x
L = 1
h = 0.1     # step size


# The domain of tau
T = 5 
p = 0.005     # step size


# function takes in the domain of x xi, the step size hi, the domain of tau ti and the step size pi and returns the discretised grid
def grid(xi,hi,ti,pi):
    # initialising discretised grid
    x = np.arange(-xi,xi+hi,hi)
    tau = np.arange(0,ti+pi,pi)
    return x,tau

x, tau = grid(L,h,T,p)
n = len(x)
m = len(tau)
print(n,m)
# In this section I am defining arrays I would need (if neeeded)
# initialising the solution matrix u(tau, x)
u = np.zeros((int(m),int(n)))

# In this section I am setting the boundary conditions/initial values
def boundary_conditions(u,x,tau):
    # setting the initial condition at tau = 0
    u[0,:] = np.maximum(np.exp(x)*K - K,0)*np.exp(alpha*x)
    # setting the boundary condition at x = -L
    u[:,0] = 0
    # setting the boundary condition at x = L
    u[:,-1] = np.exp(alpha*x[-1] + beta*tau) * (np.exp(x[-1])*K - K*np.exp(-r*tau))
    return u

u = boundary_conditions(u,x,tau)
print(u)

# In this section I am implementing the numerical method using an explicit finite difference scheme
# explicit takes in matrix u, the number of rows m and the number of columns n and returns the solution matrix
def explicit(u,m,n,p,h):
    # initialise explicit solution matrix
    u_explicit = u.copy()
    # populating the solution matrix from second row to last row
    for i in range(m-1):        # looping through tau
        for j in range(1,n-1):    # looping through x
            u_explicit[i+1,j] = u_explicit[i,j] + ((sigma2*p/(h**2)) * (u_explicit[i,j+1] - 2*u_explicit[i,j] + u_explicit[i,j-1]))
    return u_explicit
u_explicit = explicit(u,m,n,p,h)


#In this section I am implementing the numerical method using an implicit finite difference scheme
# implicit takes in matrix u, the number of rows m and the number of columns n and returns the solution matrix
def implicit(u,m,n,p,h):
    # initialise implicit solution matrix
    u_implicit = u.copy()
    # diagonal coefficient
    a = (1 + (sigma2*p/h**2))
    # upper and lower diagonal coefficient
    b = -sigma2*p/(2*h**2)
    for i in range(1,m):       # looping through tau
        # initialise tridiagonal matrix A
        A = np.zeros((n-2,n-2))
        # initialise known solution matrix C
        C = np.zeros((n-2,1))
        
        C[0,0] = -b*(u_implicit[i-1,0]+u_implicit[i-1,2]-2*u_implicit[i-1,1]) + u_implicit[i-1,1] - b*u_implicit[i,0]
        C[-1,0] = -b*(u_implicit[i-1,-1]+u_implicit[i-1,-3]-2*u_implicit[i-1,-2]) + u_implicit[i-1,-2] - b*u_implicit[i,-1]
        A[0,:2] = [a,b]
        A[-1,-2:] = [b,a]
        # populate A and C
        for j in range(1,n-3):
            A[j,j-1] = b
            A[j,j] = a
            A[j,j+1] = b         
            C[j,0] = -b*(u_implicit[i-1,j-1]+u_implicit[i-1,j+1]-2*u_implicit[i-1,j]) + u_implicit[i-1,j]
        
        # solve for u(tau_i, x)
        B = np.linalg.solve(A,C)
        u_implicit[i,1:-1] = B.flatten()

    return u_implicit

u_implicit = implicit(u,m,n,p,h)
    
# In this section I am showing the results
# Convert the solutions into options value
v_explicit = np.zeros((m,n))
v_implicit = np.zeros((m,n))
v = np.zeros((m,n))
for i in range(m):
    v_explicit[i] = u_explicit[i] / np.exp(alpha*x+beta*tau[i])
    v_implicit[i] = u_implicit[i] / np.exp(alpha*x+beta*tau[i])
    v[i] = u[i] / np.exp(alpha*x+beta*tau[i])
# convert x domain into stock price
S = K * np.exp(x)
s_grid, tau_grid = np.meshgrid(S,tau)


# surface plot of the explicit solution
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(s_grid,tau_grid,v_explicit,cmap='viridis')
ax.set_title('Explicit solution')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiration')
ax.set_zlabel('Option Value')
ax.view_init(30, -150)
plt.show()

# surface plot of the implicit solution
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(s_grid,tau_grid,v_implicit,cmap='viridis')
ax.set_title('Implicit solution')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiration')
ax.set_zlabel('Option Value')
ax.view_init(30, -150)
plt.show()