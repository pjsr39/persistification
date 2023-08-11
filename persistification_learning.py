import numpy as np
from scipy.optimize import minimize

t_span = 30
dt = 0.01
t = np.arange(0, t_span + dt, dt)

# Initial position and velocity of intelligent robot 
x1i = np.array([4])
y1i = np.array([10.5])
x2i = np.array([1])
y2i = np.array([1])

# Initial position of dumb robot 
x1d = np.array([10])
y1d = np.array([6])

# position of charging station
x_cs = np.array([6.5])
y_cs = np.array([10])

# initial battery level of intelligent and dumb robots
E1 = np.array([5500])
Ed = np.array([5200])

E_charge = 5000
E_lower = 4100
E_min = 3700

# constants
k = 3
k_cs = 2
kv = 1.5
K_x = 1
K_y = 1
r = 1
B_d = 18
B_c = 100
k_rho = 6
d_charge = 0.5
alpha = 4

# centeres of circular trajectory
Xc1 = 6.5
Yc1 = 14

Xc2 = 6.5
Yc2 = 7

# parameters for circle outside the charging station
r_cs = d_charge

for n in range(len(t)):
    print("n value after the loop:", n)
    barrier_x = r_cs * np.cos(t) + x_cs
    barrier_y = r_cs * np.sin(t) + y_cs

    vi = np.sqrt(x2i ** 2 + y2i ** 2) # absolute velocity of the int. robot

    # desired circular trajectory for int. robot
    xdes1 = r * np.cos(t) + Xc1
    ydes1 = r * np.sin(t) + Yc1

    # desired circular trajectory for dumb robot
    xdes2 = r * np.cos(t) + Xc2
    ydes2 = r * np.sin(t) + Yc2

    # Arrays for position and velocity of int. robot
    X_i = np.array([x1i[n], y1i[n]])
    v_i = np.array([x2i[n], y2i[n]])

    # Arrays for position of dumb robot
    X_d = np.array([x1d[n], y1d[n]])

    # Array for circular trajectory of int. robot
    Xdes = np.array([xdes1[n], ydes1[n]])
    
    # Array for charging station
    X_cs = np.array([x_cs[0], y_cs[0]])

    # controller for int. robot to track the desired circular trajectory
    e1 = Xdes[0] - X_i[0]
    alpha1 = -r * np.sin(t) + k * e1
    z1 = v_i[0] - alpha1
    alpha1_dot = -r * np.cos(t) - k * z1 - k ** 2 * e1
    uxi = e1 + alpha1_dot - k * z1

    e2 = Xdes[1] - X_i[1]
    alpha2 = r * np.cos(t) + k * e2
    z2 = v_i[1] - alpha2
    alpha2_dot = -r * np.sin(t) - k * z2 - k ** 2 * e2
    uyi = e2 + alpha2_dot - k * z2

    U_nom = np.array([uxi[n], uyi[n]])

    # barrier function and controller for int. robot to maintain the battery level
    norm_rob_cs = np.sqrt((X_i[0] - X_cs[0]) ** 2 + (X_i[1] - X_cs[1]) ** 2)
    rho = (B_d / k_rho) * np.log(norm_rob_cs / d_charge)
    h2 = (-B_d / k_rho * norm_rob_cs ** 2) * ((X_i[0] - X_cs[0]) * v_i[0] + (X_i[1] - X_cs[1]) * v_i[1]) - B_d * np.abs(vi ** 2) + alpha * (E1[n] - E_min - rho)
    partialh2_partialx = (-B_d / k_rho) * (norm_rob_cs ** 2 * v_i[0] - 2 * ((X_i[0] - X_cs[0]) * v_i[0] + (X_i[1] - X_cs[1]) * v_i[1]) * (X_i[0] - X_cs[0])) / norm_rob_cs ** 4 - alpha * (B_d / k_rho) * (X_i[0] - X_cs[0]) / norm_rob_cs ** 2
    partialh2_partialy = (-B_d / k_rho) * (norm_rob_cs ** 2 * v_i[1] - 2 * ((X_i[0] - X_cs[0]) * v_i[0] + (X_i[1] - X_cs[1]) * v_i[1]) * (X_i[1] - X_cs[1])) / norm_rob_cs ** 4 - alpha * (B_d / k_rho) * (X_i[1] - X_cs[1]) / norm_rob_cs ** 2
    Lfh2 = partialh2_partialx * v_i[0] + partialh2_partialy * v_i[1] + alpha * (-B_d * np.abs(vi ** 2))
    Lgh2 = (-B_d / k_rho * (norm_rob_cs ** 2)) * np.array([X_i[0] - X_cs[0] - 2 * B_d * v_i[0], X_i[1] - X_cs[1] - 2 * B_d * v_i[1]])

    H = np.eye(2)
    f = -U_nom
    A = -Lgh2
    b = alpha * h2 + Lfh2
    res = minimize(lambda u: 0.5 * np.dot(u, np.dot(H, u)) + np.dot(f, u),x0=np.zeros(2),constraints={'type': 'ineq', 'fun': lambda u: np.dot(A, u) - b},options={'disp': False})
    u = res.x

    U_track = np.array([u[0], u[1]]) # combined controller for tracking the desired trajectory and maintaining battery level

    # controller to take the int. robot to the CS
    e_csx = X_cs[0] - X_i[0]
    sigma_cs1 = k_cs * e_csx
    z_csx = v_i[0] - sigma_cs1
    sigma_cs1dot = -k_cs * z_csx - k ** 2 * e_csx
    u_csxi = e_csx + sigma_cs1dot - k * z_csx

    e_csy = X_cs[1] - X_i[1]
    sigma_cs2 = k_cs * e_csy
    z_csy = v_i[1] - sigma_cs2
    sigma_cs2dot = -k_cs * z_csy - k ** 2 * e_csy
    u_csyi = e_csy + sigma_cs2dot - k * z_csy

    U_cs = np.array([u_csxi, u_csyi])

    # controller for dumb robot to track the desired trajectory
    ex = X_d[0] - xdes2
    ey = X_d[1] - ydes2

    ux_d = -r * np.sin(t) - K_x * ex
    uy_d = r * np.cos(t) - K_y * ey

    # controller to take dumb robot to CS
    excs = X_d[0] - X_cs[0]
    eycs = X_d[1] - X_cs[1]

    ux_d_cs = -k_cs * excs
    uy_d_cs = -k_cs * eycs

    ######## Put Your Code Here ########

    ####################################

    # dynamic updation of int. robot states and energy
    x1i = np.append(x1i, x1i[n] + dt * x2i[n])
    x2i = np.append(x2i, x2i[n] + dt * U_track[0])

    y1i = np.append(y1i, y1i[n] + dt * y2i[n])
    y2i = np.append(y2i, y2i[n] + dt * U_track[1])

    E1 = np.append(E1, E1[n] - dt * 100)

    # dynamic updation of dumb robot states and energy
    x1d = np.append(x1d, x1d[n] + dt * ux_d[n])
    y1d = np.append(y1d, y1d[n] + dt * uy_d[n])
    Ed = np.append(Ed, Ed[n] - dt * 100)