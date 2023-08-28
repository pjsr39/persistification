import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from matplotlib.animation import FuncAnimation

t_span = 300
dt = 0.01
t = np.arange(0, t_span + dt, dt)
num_steps = len(t)

# Initial conditions
x1i = np.zeros(num_steps + 1)
y1i = np.zeros(num_steps + 1)
x2i = np.zeros(num_steps + 1)
y2i = np.zeros(num_steps + 1)
E = np.zeros(num_steps + 1)

x1d = np.zeros(num_steps + 1)
y1d = np.zeros(num_steps + 1)
Ed = np.zeros(num_steps + 1)

x_cs = np.full(num_steps, 6.5)
y_cs = np.full(num_steps, 10)

x_wa = np.full(num_steps, 6.5)
y_wa = np.full(num_steps, 10.8)

x1i[0] = 10
y1i[0] = 16
x2i[0] = 1
y2i[0] = 1
E[0] = 5200

x1d[0] = 10
y1d[0] = 6
Ed[0] = 5100

Xc1 = 6.5
Yc1 = 14

Xc2 = 6.5
Yc2 = 7

Kd = 1
k = 3
k_cs = 2
kv = 1.5
K_x = 1
K_y = 1
k_wa = 2.5
r = 1
B_d = 25
B_c = 20
B_d_dumb = 25
B_c_dumb = 20
k_rho = 6
d_charge = 0.5
alpha = 4
E_charge = 5000
E_lower = 4100
E_min = 3700

c = 1.5
m = 1

p_d = np.zeros(num_steps + 1)
p_c = np.zeros(num_steps + 1)
needs_charging_d = False
needs_charging_i = False

cs_occupied_d = False

X_cs = np.zeros((2, num_steps))
X_d = np.zeros((2, num_steps))
X_i = np.zeros((2, num_steps))
V_i = np.zeros((2, num_steps))
X_wa_i = np.zeros((2, num_steps))
Xdes_i = np.zeros((2, num_steps))
Xdes_d = np.zeros((2, num_steps))

# parameters for circle outside the charging station
r_cs = d_charge


for n in range(len(t)-1):

    print("Current time =", n * dt)

    # Circle around the charging station
    barrier_x = r_cs * np.cos(t) + x_cs
    barrier_y = r_cs * np.sin(t) + y_cs

    X_cs[:,n] = [x_cs[0], y_cs[0]]

    xdes1 = r * np.cos(c*t) + Xc1
    ydes1 = r * np.sin(c*t) + Yc1
    Xdes_i[:,n] = [xdes1[n], ydes1[n]]

    xdes2 = r * np.cos(m*t) + Xc2
    ydes2 = r * np.sin(m*t) + Yc2
    Xdes_d[:,n] = [xdes2[n], ydes2[n]]

    X_i[:,n] = [x1i[n], y1i[n]]
    X_d[:,n] = [x1d[n], y1d[n]]
    V_i[:,n] = [x2i[n], y2i[n]]
    X_wa_i[:,n] = [x_wa[n], y_wa[n]]
    
    ### Controller for intelligent robot to track ###
    e1 = xdes1[n] - x1i[n]
    alpha1 = -c*r * np.sin(c*t) + k * e1
    z1 = x2i[n] - alpha1
    alpha1_dot = -c**2 * r * np.cos(c*t) - k * z1 - k**2 * e1
    ux_i = e1 + alpha1_dot - k * z1

    e2 = ydes1[n] - y1i[n]
    alpha2 = c*r * np.cos(c*t) + k * e2
    z2 = y2i[n] - alpha2
    alpha2_dot = -c**2 * r * np.sin(c*t) - k * z2 - k**2 * e2
    uy_i = e2 + alpha2_dot - k * z2

    U_nom = np.array([ux_i, uy_i])

    ### Barrier function implementation ###
    vi = np.sqrt(x2i[n]**2 + y2i[n]**2)
    norm_rob_cs = np.sqrt((X_i[0, n] - X_cs[0, n])**2 + (X_i[1, n] - X_cs[1, n])**2)
    rho = (B_d / k_rho) * np.log(norm_rob_cs / d_charge)
    h2 = (-B_d / k_rho * norm_rob_cs ** 2) * ((X_i[0, n] - X_cs[0, n]) * V_i[0, n] + (X_i[1, n] - X_cs[1, n]) * V_i[1, n]) - B_d * abs(V_i[0, n] ** 2) + alpha * (E[n] - E_min - rho)
    partialh2_partialx = (-B_d / k_rho) * (norm_rob_cs ** 2 * V_i[0, n] - 2 * ((X_i[0, n] - X_cs[0, n]) * V_i[0, n] + (X_i[1, n] - X_cs[1, n]) * V_i[1, n]) * (X_i[0, n] - X_cs[0, n])) / norm_rob_cs ** 4 - alpha * (B_d / k_rho) * (X_i[0, n] - X_cs[0, n]) / norm_rob_cs ** 2
    partialh2_partialy = (-B_d / k_rho) * (norm_rob_cs ** 2 * V_i[1, n] - 2 * ((X_i[0, n] - X_cs[0, n]) * V_i[0, n] + (X_i[1, n] - X_cs[1 ,n]) * V_i[1, n]) * (X_i[1, n] - X_cs[1, n])) / norm_rob_cs ** 4 - alpha * (B_d / k_rho) * (X_i[1, n] - X_cs[1, n]) / norm_rob_cs ** 2
    Lfh2 = partialh2_partialx * V_i[0, n] + partialh2_partialy * V_i[1, n] + alpha * (-B_d * abs(vi ** 2))
    Lgh2 = (-B_d / k_rho * (norm_rob_cs ** 2)) * np.array([X_i[0, n] - X_cs[0, n] - 2 * B_d * V_i[0, n], X_i[1, n] - X_cs[1, n] - 2 * B_d * V_i[1, n]])
    H = np.eye(2)
    f = -U_nom.T
    A = -Lgh2
    b = alpha * h2 + Lfh2
    u = solve_qp(H, f[n], A, np.array([b]), None, None, None, None, solver="quadprog")
    
    U_track_i = np.array([u[0], u[1]]) # combined controller for tracking the desired trajectory and maintaining battery level
    
    ### controller to take the int. robot to the CS
    e_csx = x_cs[n] - x1i[n]
    sigma_cs1 = k_cs * e_csx
    z_csx = x2i[n] - sigma_cs1
    sigma_cs1dot = -k_cs * z_csx - k**2 * e_csx
    u_csx_i = e_csx + sigma_cs1dot - k * z_csx

    e_csy = y_cs[n] - y1i[n]
    sigma_cs2 = k_cs * e_csy
    z_csy = y2i[n] - sigma_cs2
    sigma_cs2dot = -k_cs * z_csy - k**2 * e_csy
    u_csy_i = e_csy + sigma_cs2dot - k * z_csy

    U_cs_i = np.array([u_csx_i, u_csy_i])

    ### controller to take the int. robot to the waiting area
    e_wa_x = x_wa[n] - x1i[n]
    sigma_wa1 = k_wa * e_wa_x
    z_wax = x2i[n] - sigma_wa1
    sigma_wa1dot = -k_wa * z_wax - k_wa**2 * e_wa_x
    u_wax = e_wa_x + sigma_wa1dot - k_wa * z_wax

    e_wa_y = y_wa[n] - y1i[n]
    sigma_wa2 = k_wa * e_wa_y
    z_way = y2i[n] - sigma_wa2
    sigma_wa2dot = -k_wa * z_way - k_wa**2 * e_wa_y
    u_way = e_wa_y + sigma_wa2dot - k_wa * z_way

    U_wa_i = np.array([u_wax, u_way])

    ### Controller for dumb to track the trajectory ###
    ex = x1d[n] - xdes2[n]
    ey = y1d[n] - ydes2[n]

    ux_d = -r * np.sin(t) - K_x * ex
    uy_d = r * np.cos(t) - K_y * ey

    U_track_dumb = np.array([ux_d[n], uy_d[n]])

    ### Controller for dumb to take to the charging station
    excs = x1d[n] - x_cs[n]
    eycs = y1d[n] - y_cs[n]

    ucs_x_d = -k_cs * excs
    ucs_y_d = -k_cs * eycs

    Ucs_d = np.array([ucs_x_d, ucs_y_d])

    p_d = np.sqrt((X_d[0, n] - X_cs[0, n])**2 + (X_d[1, n] - X_cs[1, n])**2)
    p_c = np.sqrt((X_i[0, n] - X_cs[0, n])**2 + (X_i[1, n] - X_cs[1, n])**2)
    p_wa = np.sqrt((X_i[0, n] - X_wa_i[0, n])**2 + (X_i[1, n] - X_wa_i[1, n])**2)

    ### dumb condition
    if Ed[n] <= E_lower:
        needs_charging_d = True
    elif Ed[n] >= E_charge:
        needs_charging_d = False

    if needs_charging_d:
        U_dumb = Ucs_d
    else:
        U_dumb = U_track_dumb

    if p_d > d_charge:
        cs_occupied_d = False
        battery_dumb = -B_d_dumb - 15
    elif p_d <= d_charge:
        cs_occupied_d = True
        battery_dumb = B_c_dumb
    	
    ### intelligent condition
    if E[n] <= E_lower:
        needs_charging_i = True
    elif E[n] >= E_charge:
        needs_charging_i = False

    if needs_charging_i and needs_charging_d:
        U = U_wa_i
        if p_wa < 0.02:
            vi = 0
    elif needs_charging_i:
        U = U_cs_i
    else:
        U = U_track_i

    if p_c > d_charge:
        B = -B_d*abs(vi)**2 - Kd
    else:
        B = B_c

    x1i[n+1] = x1i[n] + dt * x2i[n]
    x2i[n+1] = x2i[n] + dt * U[0]
    y1i[n+1] = y1i[n] + dt * y2i[n]
    y2i[n+1] = y2i[n] + dt * U[1]
    E[n + 1] = E[n] + dt * B

    x1d[n+1] = x1d[n] + dt * U_dumb[0]
    y1d[n+1] = y1d[n] + dt * U_dumb[1]
    Ed[n + 1] = Ed[n] + dt * battery_dumb


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(xdes1, ydes1, color = 'cyan')
ax1.plot(xdes2, ydes2, color = 'cyan')
ax1.scatter(x_cs[0], y_cs[0], facecolors="none", edgecolors = "green")
ax1.plot(barrier_x, barrier_y)
ax1.scatter(x_wa[0], y_wa[0], facecolors="none", edgecolors = "black")

# Initialize Line2D objects for x1i vs y1i and E vs Ed plots
line_x1i_y1i, = ax1.plot([], [])
line_x1d_y1d, = ax1.plot([], [])
line_E, = ax2.plot([], [], color='blue', label='E')
line_E_Ed_d, = ax2.plot([], [], color='magenta', label='Ed')
moving_point_i, = ax1.plot([], [], marker='o', color='blue', markersize=2)
moving_point_d, = ax1.plot([], [], marker='o', color='magenta', markersize=2)
ax2.legend()

ax2.plot(t, np.full_like(t, E_charge), '--', label='E_charge')
ax2.plot(t, np.full_like(t, E_min), '--', label='E_min')
ax2.plot(t, np.full_like(t, E_lower), '--', label='E_lower')

# Set axis limits for the subplots
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_xlabel('x position')
ax1.set_ylabel('y position')

ax2.set_xlim(0, t_span)
ax2.set_ylim(3500, 5500)
ax2.set_xlabel('Time') 
ax2.set_ylabel('Energy')
ax2.grid()

# Initialize text for displaying time in the plots
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
time_text2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

# Function to update the plot data in each frame
def update(frame):
    
    #line_x1i_y1i.set_data(x1i[:frame], y1i[:frame]) ## for tracing lines
    #line_x1d_y1d.set_data(x1d[:frame], y1d[:frame]) ## for tracing lines
    moving_point_i.set_data(x1i[frame], y1i[frame])
    moving_point_d.set_data(x1d[frame], y1d[frame])
    line_E.set_data(t[:frame], E[:frame])
    line_E_Ed_d.set_data(t[:frame], Ed[:frame])
    time_text.set_text('Time = {:.2f}'.format(frame * dt))
    time_text2.set_text('Time = {:.2f}'.format(frame * dt))
    return line_E_Ed_d, time_text, time_text2, line_x1d_y1d, line_E, moving_point_i, moving_point_d

# Create the animations
animation_x1i_y1i = FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
animation_x1d_y1d = FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
animation_E_Ed = FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
animation_E = FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)

# Show the animations
plt.show()



#plt.plot(t, Ed[:-1])
#plt.plot(t, np.full_like(t, E_charge), '--', label='E_charge')
#plt.plot(t, np.full_like(t, E_min), '--', label='E_min')
#plt.plot(t, np.full_like(t, E_lower), '--', label='E_lower')
#plt.xlabel('Time')
#plt.ylabel('Ed')
#plt.title('Energy of Dumb Robot over Time')
#plt.grid(True)
#plt.show()