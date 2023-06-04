import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def pquad(con): #Create quadratic penalty
    p = 0
    for i in con:
        p = p + (max(i,0))**2
    return p

def basis(i,n): #Create a basis vector given input dimension and i
    v = []
    for num in range(n):
        if i == num:
            v = v + [1]
        else:
            v = v + [0]
    return v


def cross_entropy_method(f, P, c, rho, k_max, l, m=30, m_elite=15):
    Xvals = np.zeros(l)
    for k in range(k_max):
        #Gather samples and sort elite samples
        samples = P.rvs(m)
        order = np.argsort([f(sample)+rho*pquad(c(sample)) for sample in samples])
        elite_samples = samples[order][:m_elite]

        #Calculate the new mean
        mu = np.mean(elite_samples, axis=0)

        # Calculate the covariance matrix
        sigma = np.cov(elite_samples, rowvar=False)

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(sigma)

        # Adjust eigenvalues
        min_eigenvalue = 1e-6  # Set a minimum threshold for eigenvalues
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        # Reconstruct the positive definite covariance matrix
        sigma_pd = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))

        #Set the new distribution
        P = multivariate_normal(mean=mu, cov=sigma_pd)
        Xvals = np.vstack((Xvals, [P.mean]))
    return P, Xvals

"""Objective function outlining the profits from one year of crop planting and harvest not taking
into consideration the maturation time of trees. """
def f(x): #Objective function for one year of mature crop harvest
    price = np.array([2.0, 25.0, 1.35, 1.32, 10.0]) # dollars per lb
    crop_yield = np.array([2000.0, 300.0, 80000.0, 16222.0, 10000.0]) #lb per acre
    cost = np.array([3897.0, 1800.0, 4000.0, 16233.0, 2500.0]) # Cost per acre 
    y = 0.0
    for i in range(len(x)):
        y = y - (crop_yield[i]*price[i]-cost[i])*x[i] 
    return y

"""Objective function outlining profits from 10 years of farming, taking into account time for some 
crops to mature for harvest. Subtracted cost once for initial planting and upkeep while garlic grows once every
9 months. """

def f_2(x): 
    price = np.array([2.0, 25.0, 1.35, 1.32, 10.0]) # dollars per lb
    crop_yield = np.array([2000.0, 300.0, 80000.0, 16222.0, 10000.0]) #lb per acre
    cost = np.array([3897.0, 1800.0, 4000.0, 16233.0, 2500.0]) # Cost per acre 
    y = 5*crop_yield[0]*price[0]*x[0]-cost[0]*x[0]
    y = y - 6*crop_yield[1]*price[1]*x[1]-cost[1]*x[1]
    return y

"""Constraint function utilizing only two dimensions/crops in order to plot and visualize optimizaiton algorithm
May seem confusing but was easier to plan this way"""
def c(x):
    net_sown_area = 330.0
    Amin = np.array([50.0, 30.0])
    Amax = np.array([300.0, 300.0])
    water_needed = np.array([4.49, 4.23]) #Acre-feet per acre of crop in one growing season
    total_ground_water = 1168.75
    #g(x) <= 0
    #h(x) = 0
    c1 = 0.0
    for i in range(len(x)):
        c1 = c1 + x[i]
    c1 = c1-(net_sown_area)
    c2 = Amin[0]-x[0]
    c3 = Amin[1]-x[1]
    c7 = x[0]-Amax[0]
    c8 = x[1]-Amax[1]
    c12 = 0.0
    for i in range(len(x)):
        c12 = c12 + water_needed[i]*x[i]
    c12 = c12 - total_ground_water
    return np.array([c1, c2, c3, c7, c8, c12])

"""Same constraints but in five dimensions"""
def c5(x):
    net_sown_area = 330.0
    Amin = np.array([50.0, 30.0, 30.0, 30.0, 30.0])
    Amax = np.array([150.0, 150.0, 150.0, 150.0, 150.0])
    water_needed = np.array([4.49, 4.23, 3.7, 4.23, 2.96]) #Acre-feet per acre of crop in one growing season
    total_ground_water = 1168.75
    #g(x) <= 0
    #h(x) = 0
    c1 = 0.0
    for i in range(len(x)):
        c1 = c1 + x[i]
    c1 = c1-(net_sown_area)
    c2 = Amin[0]-x[0]
    c3 = Amin[1]-x[1]
    c4 = Amin[2]-x[2]
    c5 = Amin[3]-x[3]
    c6 = Amin[4]-x[4]
    c7 = x[0]-Amax[0]
    c8 = x[1]-Amax[1]
    c9 = x[2]-Amax[2]
    c10 = x[3]-Amax[3]
    c11 = x[4]-Amax[4]
    c12 = 0.0
    for i in range(len(x)):
        c12 = c12 + water_needed[i]*x[i]
    c12 = c12 - total_ground_water
    return np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

"""Main Optimization Function using the Cross-Entropy_Method"""
def optimize_CEM(f, c, x0, n):
    #Initialize counts, solution array, and penalty value
    l = len(x0)
    Xvals = np.zeros(l)
    count = 0 
    rho = 1e10
    #Initialize the first distribution
    mu = np.array(x0)
    sigma = np.array(np.identity(l))*(30**2)
    P = multivariate_normal(mean=mu, cov=sigma)
    k_max = 100
    #Repeat distribution k_max times and repeat n times
    while count < n:
        P, Xvals = cross_entropy_method(f, P, c, rho, k_max, l)
        count += 1
    x0 = P.mean
    print(x0)
    Xvals = np.delete(Xvals, 0, 0)
    return Xvals

"""Main Optimization Function using the Hooke_Jeeves method"""
def optimize_HJ(f, c, x0, n):
    #Initialize variables and solution array
    l = len(x0)
    Xvals = np.zeros(l)
    rho = 1e10
    alpha = 150
    xbest = x0
    count = 0
    while count < (n):
        improved = False
        #Find initial y with penalty
        con = c(x0)
        y = f(x0)+rho*pquad(con)
        for i in range(l):
            for j in [-1, 1]:
                #Search area in l dimensions for improvement with penalty
                xnew = x0 + j*alpha*np.transpose(basis(i, l))
                con = c(xnew)
                ynew = f(xnew)+rho*pquad(con)
                if ynew < y:
                    #Move to next step if improvement
                    Xvals = np.vstack((Xvals, [xnew]))
                    xbest = xnew
                    y = ynew
                    improved = True
        if improved == False:
            #Shorten step size if no improvement
            alpha *= 0.5
        x0 = xbest
        count += 1
    x_best = x0
    print(x_best)
    Xvals = np.delete(Xvals, 0, 0)
    return Xvals



#Plotting
x0 = [0, 0]
xs = np.arange(0, 300,1)
ys = np.arange(0, 300,1)
rows = len(xs)
objective_contour = np.array([])
for i in range(len(xs)):
    for j in range(len(ys)):
        number = f([xs[i],ys[j]])
        objective_contour= np.append(objective_contour, number)
objective_contour = np.reshape(objective_contour, [rows, rows])

fig, ax = plt.subplots()
levels = np.linspace(np.min(objective_contour), np.max(objective_contour), 200)
p = ax.contour(xs, ys, objective_contour, levels = levels)

water_needed = np.array([4.49, 4.23]) #Acre-feet per acre of crop in one growing season
total_ground_water = 1168.75
xspos = np.arange(0,300,1)
constraint1 = np.array([])
constraint6 = np.array([])
for i in range(len(xspos)):
    constraint1 = np.append(constraint1,330-xspos[i])
for i in range(len(xspos)):
    constraint6 = np.append(constraint6,(total_ground_water-xspos[i]*water_needed[0])/water_needed[1])

x_ = np.arange(0,330,10)
y_ = np.arange(330, 0, -10)
plt.plot(x_,y_, color = 'black')
plt.plot(xspos,constraint6, color = 'black')
plt.plot([50,50],[0,300], color = 'black')
plt.plot([0,300],[30,30], color = 'black')

ax.set_title('Contour with CEM optimization')

Optimums1 = optimize_CEM(f, c, np.array([100,100]), 1)
X = Optimums1[:,0]
Y = Optimums1[:,1]

plt.plot(X,Y, color = "blue", label = 'One Year')

Optimums1 = optimize_CEM(f_2, c, np.array([100 ,100]), 1)
X = Optimums1[:,0]
Y = Optimums1[:,1]

plt.plot(X,Y, color = "red", label = 'Ten Years')

plt.xlabel('Acres of Almonds')
plt.ylabel('Acres of Oranges')
plt.xlim(0,300)
plt.ylim(0,300)
plt.legend()

plt.show()


fig, ax = plt.subplots()
p = ax.contour(xs, ys, objective_contour, levels = levels)
ax.set_title('Contour with Simplex Algorithm')
plt.plot(x_,y_, color = 'black')
plt.plot(xspos,constraint6, color = 'black')
plt.plot([50,50],[0,300], color = 'black')
plt.plot([0,300],[30,30], color = 'black')

Optimums1 = optimize_HJ(f, c, np.array([0,0]), 200)
X = Optimums1[:,0]
Y = Optimums1[:,1]

plt.plot(X,Y, color = "blue", label = 'One Year')

Optimums1 = optimize_HJ(f_2, c, np.array([0,0]), 200)
X = Optimums1[:,0]
Y = Optimums1[:,1]

plt.plot(X,Y, color = "red", label = 'Ten Years')

plt.xlabel('Acres of Almonds')
plt.ylabel('Acres of Oranges')
plt.xlim(0,300)
plt.ylim(0,300)
plt.legend()
plt.show()


fig, ax = plt.subplots()
ax.set_title('Objective Function vs. Iteration CEM')
plt.ylabel('Function Value')
plt.xlabel('Iteration')


Optimums1 = optimize_CEM(f, c5, np.array([0,0,0,0,0]), 1)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,f(val))

plt.plot(range(len(Optimums1)),fun, color = "orange", label = 'One Year')

Optimums1 = optimize_CEM(f_2, c5, np.array([0,0,0,0,0]), 1)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,f_2(val))

plt.plot(range(len(Optimums1)),fun, color = "green", label = 'Ten Years')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('Objective Function vs. Iteration Simplex')
plt.ylabel('Function Value')
plt.xlabel('Iteration')

Optimums1 = optimize_HJ(f, c5, np.array([0,0,0,0,0]), 200)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,f(val))

plt.plot(range(len(Optimums1)),fun, color = "orange", label = 'One Year')

Optimums1 = optimize_HJ(f_2, c5, np.array([0,0,0,0,0]), 200)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,f_2(val))

plt.plot(range(len(Optimums1)),fun, color = "green", label = 'Ten Years')

plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('Maximum Constraint Violation vs. Iteration CEM')
plt.ylabel('Constraint Violation')
plt.xlabel('Iteration')


Optimums1 = optimize_CEM(f, c5, np.array([0,0,0,0,0]), 1)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,(max(c5(val)[0],0)+max(c5(val)[1],0)+max(c5(val)[2],0)+max(c5(val)[3],0)+max(c5(val)[4],0)+max(c5(val)[5],0)))

plt.plot(range(len(Optimums1)),fun, color = "orange", label = 'One Year')


Optimums1 = optimize_CEM(f_2, c5, np.array([0,0,0,0,0]), 1)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,(max(c5(val)[0],0)+max(c5(val)[1],0)+max(c5(val)[2],0)+max(c5(val)[3],0)+max(c5(val)[4],0)+max(c5(val)[5],0)))

plt.plot(range(len(Optimums1)),fun, color = "green", label = 'Ten Years')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('Maximum Constraint Violation vs. Iteration Simplex')
plt.ylabel('Constraint Violation')
plt.xlabel('Iteration')

Optimums1 = optimize_HJ(f, c5, np.array([0,0,0,0,0]), 200)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,(max(c5(val)[0],0)+max(c5(val)[1],0)+max(c5(val)[2],0)+max(c5(val)[3],0)+max(c5(val)[4],0)+max(c5(val)[5],0)))

plt.plot(range(len(Optimums1)),fun, color = "orange", label = 'One Year')

Optimums1 = optimize_HJ(f_2, c5, np.array([0,0,0,0,0]), 200)
fun = np.array([])
for val in Optimums1:
    fun = np.append(fun,(max(c5(val)[0],0)+max(c5(val)[1],0)+max(c5(val)[2],0)+max(c5(val)[3],0)+max(c5(val)[4],0)+max(c5(val)[5],0)))

plt.plot(range(len(Optimums1)),fun, color = "green", label = 'Ten Years')
plt.legend()
plt.show()