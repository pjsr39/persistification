import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from matplotlib.animation import FuncAnimation
import os
import math
import matplotlib.animation as animation
from random import random

# Simulation parameters
t_span = 300
dt = 0.01
t = np.arange(0, t_span + dt, dt)
num_steps = len(t)

num_pairs = 5

e_min = 3600
e_charge = 5000
e_lower_dumb = 4200
e_lower_dumb_disp = [4200, 4200, 4200, 4200, 4200]
e_init_i = [5200, 5250, 5100, 5300, 5200]
e_init_d = [5200, 5250, 5100, 5300, 5200]
e_lower_int = [3616, 3622, 3616, 3616, 3623]

"""Initial conditions"""
# Initial intelligent state
x1 = [10, 2, -2, -4, -10]
y1 = [16, 17, 17, 17, 18]
x2 = [1, 1, 1, 1, 1]
y2 = [1, 1, 1, 1, 1]

# Initial dumb state
x1d = [10, 2, -2, -6, -8]
y1d = [6, 3, 9, 4, 7]
x2d = [1, 1, 1, 1, 1]
y2d = [1, 1, 1, 1, 1]

# Charging state position
x_cs = [8, 4, 0, -4, -8]
y_cs = [10.5, 10.5, 10.5, 10.5, 10.5]

# Energy parameters
E = e_init_i
Ed = e_init_d
E_charge = e_charge
E_lower_i = e_lower_int
E_lower_d = e_lower_dumb
E_min = e_min

# Track roundabout positions
Xc1 = [8, 4, 0, -4, -8]
Yc1 = [14, 14, 14, 14, 14]
Xc2 = [8, 4, 0, -4, -8]
Yc2 = [7, 7, 7, 7, 7]

# Control parameters
Kd = 0.1
k = 3
k_cs = 2
r = 1.5
B_d = 25
B_c = 28
B_d_dumb = 25
B_c_dumb = 28
d_charge = 0.2
alpha = 4

threshold_distance = 0.015

c = [1.5, 2.0, 1.0, 1.75, 1.8]
m = [2.0, 2.5, 2.2, 1.8, 2.25]

p_d = [0] * num_pairs # for dumb
p_c = [0] * num_pairs # for intelligent

# Flags
needs_charging_d = [False] * num_pairs
needs_charging_i = [False] * num_pairs
start_computing_distance = [False] * num_pairs

# block flags
vi_computed = [False] * num_pairs
vcs_computed = [False] * num_pairs

# Distance covered
arc_length = [0] * num_pairs

# parameters for circle outside the charging station
r_cs = d_charge

ed_plot = [[] for _ in range(num_pairs)]
e_plot = [[] for _ in range(num_pairs)]
e_lower_d_plot = np.full(num_steps, E_lower_d)
e_charge_plot = np.full(num_steps, E_charge)
e_min_plot = np.full(num_steps, E_min)


"Animation Lists"
x1_plot = [[] for _ in range(num_pairs)]
y1_plot = [[] for _ in range(num_pairs)]
xd_plot = [[] for _ in range(num_pairs)]
yd_plot = [[] for _ in range(num_pairs)]
nci_flag_plot = [[] for _ in range(num_pairs)]
ncd_flag_plot = [[] for _ in range(num_pairs)]
E_plot = [[] for _ in range(num_pairs)]
Ed_plot = [[] for _ in range(num_pairs)]
p_c_plot = [[] for _ in range(num_pairs)]
p_d_plot = [[] for _ in range(num_pairs)]
E_min_plot = []
E_charge_plot = []
E_lower_d_plot = []

def compute_vcs(x_curr, y_curr, x_cs, y_cs, B_d, K_d, E, E_min):
    d = math.sqrt((x_curr - x_cs)**2 + (y_curr - y_cs)**2)
    offset = 0 # it can't be any arbitrary value, it has to satisfy discriminant b^2 always > 4*a*c
    delta_E = E[i] - E_min - offset

    a = B_d
    b = - (delta_E / d)
    c = K_d

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        print("Not enough energy to reach CS at any speed!")
        return None

    sqrt_disc = math.sqrt(discriminant)
    v1 = (-b + sqrt_disc) / (2*a)
    v2 = (-b - sqrt_disc) / (2*a)

    # Return the *larger* real root (safer, more practical speed)
    vcs = max(v1, v2)
    return vcs

for i in range(num_pairs):  # Loop over each pair

    print(f'Pair: {i}')

    for n in range(len(t)-1):

        timestep = n * dt

        # Intelligent Desired Tracking Path
        xdes_i = r * math.cos(c[i]*timestep) + Xc1[i]
        ydes_i = r * math.sin(c[i]*timestep) + Yc1[i]

        # Dumb Desired Tracking Path
        xdes_d = r * math.cos(m[i]*timestep) + Xc2[i]
        ydes_d = r * math.sin(m[i]*timestep) + Yc2[i]

        if not needs_charging_d[i] and not needs_charging_i[i]:
            vi = math.sqrt(x2[i]**2 + y2[i]**2)

        ### Controller for intelligent robot to track ###
        e1 = xdes_i - x1[i]
        alpha1 = -c[i]*r * math.sin(c[i]*timestep) + k * e1
        z1 = x2[i] - alpha1
        alpha1_dot = -c[i]**2 * r * math.cos(c[i]*timestep) - k * z1 - k**2 * e1
        ux_i = e1 + alpha1_dot - k * z1
        
        e2 = ydes_i - y1[i]
        alpha2 = c[i]*r * math.cos(c[i]*timestep) + k * e2
        z2 = y2[i] - alpha2
        alpha2_dot = -c[i]**2 * r * math.sin(c[i]*timestep) - k * z2 - k**2 * e2
        uy_i = e2 + alpha2_dot - k * z2

        U_track_i = np.array([ux_i, uy_i])

        ### Controller for dumb to track the trajectory ###
        ex = xdes_d - x1d[i]
        alpha1d = -m[i] * r * math.sin(m[i]*timestep) + k * ex
        z1d = x2d[i] - alpha1d
        alpha1d_dot = -m[i]**2 * r * math.cos(m[i]*timestep) - k * z1d - k**2 * ex
        ux_d = ex + alpha1d_dot - k * z1d

        ey = ydes_d - y1d[i]
        alpha2d = m[i]*r * math.cos(m[i]*timestep) + k * ey
        z2d = y2d[i] - alpha2d
        alpha2d_dot = -m[i]**2 * r * math.sin(m[i]*timestep) - k * z2d - k**2 * ey
        uy_d = ey + alpha2d_dot - k * z2d

        U_track_dumb = np.array([ux_d, uy_d]) 

        ### controller to take the dumb robot to the CS
        e_csd_x = x_cs[i] - x1d[i]
        sigma_csd1 = k_cs * e_csd_x
        z_csd_x = x2d[i] - sigma_csd1
        sigma_csd1_dot = -k_cs * z_csd_x - k**2 * e_csd_x
        u_csd_x = e_csd_x + sigma_csd1_dot - k * z_csd_x

        e_csd_y = y_cs[i] - y1d[i]
        sigma_csd2 = k_cs * e_csd_y
        z_csd_y = y2d[i] - sigma_csd2
        sigma_csd2_dot = -k_cs * z_csd_y - k**2 * e_csd_y
        u_csd_y = e_csd_y + sigma_csd2_dot - k * z_csd_y

        Ucs_d = np.array([u_csd_x, u_csd_y])

        p_d[i] = math.sqrt((x1d[i] - x_cs[i])**2 + (y1d[i] - y_cs[i])**2)

        """needs charging flags part"""
        ### dumb condition
        if Ed[i] <= E_lower_d:
            needs_charging_d[i] = True
        elif Ed[i] >= E_charge:
            needs_charging_d[i] = False

        if needs_charging_d[i]:
            U_dumb = Ucs_d
        else:
            U_dumb = U_track_dumb

        if p_d[i] > d_charge:
            if n < 15000:
                battery_dumb = -B_d_dumb
            elif 15000 <= n <= 18000:
                battery_dumb = -B_d_dumb - 10
            else:
                battery_dumb = -B_d_dumb - 15
        elif p_d[i] <= d_charge:
            battery_dumb = B_c_dumb

        ### intelligent condition
        if E[i] <= E_lower_i[i]:
            needs_charging_i[i] = True
        elif E[i] >= E_charge:
            needs_charging_i[i] = False
            vi_computed[i] = False

        """ Instant cs """
        if needs_charging_i[i] and not needs_charging_d[i] and not vcs_computed[i]:
            vcs = compute_vcs(x1[i], y1[i], x_cs[i], y_cs[i], B_d, Kd, E, E_min)
            vi = vcs
            vcs_computed[i] = True 
            vec_to_cs = np.array([x_cs[i] - x1[i], y_cs[i] - y1[i]])
            dist_to_cs = np.linalg.norm(vec_to_cs)
            v_vec = vcs * ( vec_to_cs /  dist_to_cs)
            x2[i], y2[i] = v_vec[0], v_vec[1]
            # print(f'vcs computed: {vcs}')
            U_i = np.array([0.0, 0.0])

        if needs_charging_i[i] and needs_charging_d[i]: 
            T_CD = (E_charge - E_lower_d) / B_c_dumb
            offset = 0
            v_red = math.sqrt(((E[i] - E_min - offset - T_CD * Kd )) / (T_CD * B_d ))
            vi = v_red
            vi_computed[i] = True
            # print(f'vred_sqr:{vi}')
            vcs_computed[i] = False

        p_c[i] = math.sqrt((x1[i] - x_cs[i])**2 + (y1[i] - y_cs[i])**2)
        
        if needs_charging_i[i] and p_c[i] <= d_charge and vcs_computed[i]:
            x2[i], y2[i] = 0.0, 0.0  # Stop motion
            vi = math.sqrt(x2[i]**2 + y2[i]**2)

        if needs_charging_i[i] and needs_charging_d[i]:
            U_i = U_track_i
        elif needs_charging_i[i]:
            U_i = np.array([0.0, 0.0])
        else:
            U_i = U_track_i

        if p_c[i] > d_charge:
            B = -B_d*(vi)**2 - Kd
        else:
            B = B_c

        if abs(xdes_i - x1[i]) <= threshold_distance and abs(ydes_i - y1[i]) <= threshold_distance:
            if start_computing_distance[i] == False:
                start_computing_distance[i] = True
                alpha_prev = math.atan2(y1[i] - Yc1[i], x1[i] - Xc1[i])
            else:
                arc_length[i] += r * abs( abs( (math.atan2(y1[i] - Yc1[i], x1[i] - Xc1[i]) ) ) - abs(alpha_prev) )
                alpha_prev = math.atan2(y1[i] - Yc1[i], x1[i] - Xc1[i])
        else:
            start_computing_distance[i] = False

        # Intelligent's double integrator dynamics        
        x1[i] = x1[i] + dt * x2[i] 
        x2[i] = x2[i] + dt * U_i[0] 
        y1[i] = y1[i] + dt * y2[i] 
        y2[i] = y2[i] + dt * U_i[1] 
        E[i] =   E[i] + dt * B

        # Dumb's double integrator dynamics
        x1d[i] = x1d[i] + dt * x2d[i]
        x2d[i] = x2d[i] + dt * U_dumb[0] 
        y1d[i] = y1d[i] + dt * y2d[i] 
        y2d[i] = y2d[i] + dt * U_dumb[1] 
        Ed[i] =  Ed[i] + dt * battery_dumb

        if E[i] < E_min:
            print(f'Energy went below E_min: {E[i]}, Pair: {i}')
            exit()

        # print(f'Time : {n*dt}, Ed: {round(Ed[i], 3)}, E: {round(E[i],3)}, Int. battery rate: {round(B,3)}, vcs_c: {vcs_computed[i]},  vi_c : {vi_computed[i]}, vi: {round(vi,3)}, battery rate dumb: {round(battery_dumb,3)}')

        x1_plot[i].append(x1[i])
        y1_plot[i].append(y1[i])
        xd_plot[i].append(x1d[i])
        yd_plot[i].append(y1d[i])
        E_plot[i].append(E[i])
        Ed_plot[i].append(Ed[i])
        nci_flag_plot[i].append(needs_charging_i[i])
        ncd_flag_plot[i].append(needs_charging_d[i])
        p_c_plot[i].append(p_c[i])
        p_d_plot[i].append(p_d[i])


ax = plt.subplot()
plt.ylim(3500, 5350)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
agent_colors = [("red","darkred"),("green","darkgreen"),("blue","darkblue"),("violet","darkviolet"),("gold","goldenrod")]
for i in range(num_pairs):
    t_val = [x * dt for x in range(len(Ed_plot[i]))]
    plt.plot(t_val, Ed_plot[i], color = agent_colors[i][0])  
    plt.plot(t_val, E_plot[i], color = agent_colors[i][1])

t_val = [x * dt for x in range(len(t))]
plt.plot(t_val, e_lower_d_plot, linewidth = 2, color = 'c', linestyle="dashed")
plt.plot(t_val, e_charge_plot, linewidth = 2, color = 'g', linestyle="dashed")
plt.plot(t_val, e_min_plot, linewidth = 2, color = 'r', linestyle="dashed")
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend(["$E_{IA,1}$", "$E_{DA,1}$", "$E_{IA,2}$", "$E_{DA,2}$", "$E_{IA,3}$", "$E_{DA,3}$", "$E_{IA,4}$", "$E_{DA,4}$", "$E_{IA,5}$", "$E_{DA,5}$", "$E_{low,IA}$",  "$E_{max}$", "$E_{min}$"], loc="upper center", ncol=7, mode="expand", fontsize="small")
plt.savefig("without_learning_energy_plot_circle.pdf", format="pdf", bbox_inches='tight') 
plt.savefig("without_learning_energy_plot_circle.png", format="png", dpi=600, bbox_inches='tight')  # For high-res PNG
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(-18, 18)
ax.set_ylim(3, 18)
ax.set_xlabel('x position')
ax.set_ylabel('y position')

# --- Initialize plot elements ---
intelligent_robots = []
dumb_robots = []
tracking_circles_i = []
tracking_circles_d = []
charging_circles = []
energy_text_i = []
energy_text_d = []
charging_flag_text_i = []
charging_flag_text_d = []
pair_labels = []
e_lower_labels = []
e_lower_labels_dumb = []

for i in range(num_pairs):
    ir, = ax.plot([], [], 'ko', markersize=4)
    dr, = ax.plot([], [], 'bo', markersize=4)
    circ_i = plt.Circle((Xc1[i], Yc1[i]), r, color='black', fill=False, linestyle='--')
    circ_d = plt.Circle((Xc2[i], Yc2[i]), r, color='blue', fill=False, linestyle='--')
    circ_cs = plt.Circle((x_cs[i], y_cs[i]), radius=d_charge, color='green', fill=False)
    
    ax.add_artist(circ_i)
    ax.add_artist(circ_d)
    ax.add_artist(circ_cs)

    # Texts (position offset to avoid overlap)
    y_offset = 0.95 - i * 0.08
    ei_text = ax.text(0, 0, '', fontsize=8, color='black')   # above IA
    nci_text = ax.text(0, 0, '', fontsize=8, color='black')
    ed_text = ax.text(0, 0, '', fontsize=8, color='blue')  # below DA
    ncd_text = ax.text(0, 0, '', fontsize=8, color='blue')
    label = ax.text(Xc1[i], Yc1[i] + r + 1.5, f'Pair {num_pairs - i}', ha='center', fontsize=12, fontweight='bold')
    e_text = ax.text(Xc1[i], Yc1[i] + r + 1, f'$E_{{low}}$: {E_lower_i[i]}', ha='center', fontsize=10)
    e_text_dumb = ax.text(Xc1[i], Yc1[i] + r - 11.5, f'$E_{{low}}$: {e_lower_dumb_disp[i]}', ha='center', fontsize=10, color='blue')

    intelligent_robots.append(ir)
    dumb_robots.append(dr)
    tracking_circles_i.append(circ_i)
    tracking_circles_d.append(circ_d)
    charging_circles.append(circ_cs)
    energy_text_i.append(ei_text)
    energy_text_d.append(ed_text)
    charging_flag_text_i.append(nci_text)
    charging_flag_text_d.append(ncd_text)
    pair_labels.append(label)
    e_lower_labels.append(e_text)
    e_lower_labels_dumb.append(e_text_dumb)

time_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontweight='bold', fontsize=12)
e_charge_text = ax.text(0.05, 0.56, f'$E_{{max}}$ = {E_charge}', transform=ax.transAxes, fontsize=9, color = 'green', fontweight = 'bold')
e_min_text = ax.text(0.05, 0.52, f'$E_{{min}}$ = {E_min}', transform=ax.transAxes, fontsize=9, color = 'red', fontweight = 'bold')

# --- Init function ---
def init():
    artists = []
    for i in range(num_pairs):
        intelligent_robots[i].set_data([], [])
        dumb_robots[i].set_data([], [])
        energy_text_i[i].set_text('')
        charging_flag_text_i[i].set_text('')
        energy_text_d[i].set_text('')
        charging_flag_text_d[i].set_text('')
        artists += [intelligent_robots[i], dumb_robots[i],
                    energy_text_i[i], charging_flag_text_i[i],
                    energy_text_d[i], charging_flag_text_d[i],
                    pair_labels[i], e_lower_labels[i], e_lower_labels_dumb[i]]
    time_text.set_text('')
    e_charge_text.set_text(f'$E_{{max}}$ = {E_charge}')
    e_min_text.set_text(f'$E_{{min}}$ = {E_min}')
    return artists + [time_text, e_charge_text, e_min_text]

# --- Animate function ---
def animate(i):
    time_text.set_text('Time = {:.2f}'.format(i * dt))
    artists = []

    for j in range(num_pairs):
        xi = x1_plot[j][i]
        yi = y1_plot[j][i]
        xd = xd_plot[j][i]
        yd = yd_plot[j][i]

        if nci_flag_plot[j][i]:  # needs charging
            if p_c_plot[j][i] <= d_charge:  # at CS
                intelligent_robots[j].set_color('green')
            else:
                intelligent_robots[j].set_color('red')
        else:
            intelligent_robots[j].set_color('black')

        # Dumb agent marker color logic
        if ncd_flag_plot[j][i]:  # needs charging
            if p_d_plot[j][i] <= d_charge:  # at CS
                dumb_robots[j].set_color('green')
            else:
                dumb_robots[j].set_color('red')
        else:
            dumb_robots[j].set_color('blue')

        intelligent_robots[j].set_data(xi, yi)
        dumb_robots[j].set_data(xd, yd)

        # Position the texts relative to agents
        energy_text_i[j].set_position((xi, yi + 0.4))
        charging_flag_text_i[j].set_position((xi, yi + 0.15))
        energy_text_d[j].set_position((xd, yd - 0.35))
        charging_flag_text_d[j].set_position((xd, yd - 0.6))

        energy_text_i[j].set_text(f'$E$: {E_plot[j][i]:.1f}')
        charging_flag_text_i[j].set_text(f'$n_{{ch}}$: {nci_flag_plot[j][i]}')
        energy_text_d[j].set_text(f'$E$: {Ed_plot[j][i]:.1f}')
        charging_flag_text_d[j].set_text(f'$n_{{ch}}$: {ncd_flag_plot[j][i]}')

        artists += [intelligent_robots[j], dumb_robots[j], energy_text_i[j], charging_flag_text_i[j], energy_text_d[j], charging_flag_text_d[j]]
    
    return artists + [time_text, e_charge_text, e_min_text] + pair_labels + e_lower_labels + e_lower_labels_dumb


# --- Run Animation ---
ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                              interval=10, blit=True, repeat=True)

# ani.save('without_learning_circle.mp4', writer='ffmpeg', fps=30)
plt.show()
