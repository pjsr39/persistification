import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from matplotlib.animation import FuncAnimation
import os
import math
import argparse
import matplotlib.animation as animation
import argparse
from random import random
import time


"""Simulation length"""
t_span = 300
dt = 0.01
t = np.arange(0, t_span + dt, dt)
num_steps = len(t)

num_pairs = 5

e_min = 3600
e_lower_dumb = 4200
e_lower_dumb_disp = [4200, 4200, 4200, 4200, 4200]
e_charge = 5000
e_init_i = [5200, 5250, 5100, 5300, 5150]
e_init_d = [5200, 5250, 5100, 5300, 5150]
e_lower_int = [3616, 3637, 3616, 3617, 3617]

"""Initial conditions"""
# Initial intelligent state
x1 = [10.0, 1.50, -7.0, -13.5, -19.0]
y1 = [16.0, 17.0, 16.0, 15.5, 17.5]
x2 = [1, 1, 1, 1, 1]
y2 = [1, 1, 1, 1, 1]

# Initial dumb state
x1d = [10, 1.0, -7.0, -13.5, -17.5]
y1d = [6, 4, 5, 4, 5]
x2d = [1, 1, 1, 1, 1]
y2d = [1, 1, 1, 1, 1]

# Charging state position
x_cs = [6.5, 0.5, -5.5, -11.5, -17.5]
y_cs = [10.5, 10.5, 10.5, 10.5, 10.5]

# Energy parameters
E = e_init_i
Ed = e_init_d
E_charge = e_charge
E_lower_i = e_lower_int
E_lower_d = e_lower_dumb
E_min = e_min

# Track roundabout positions
Xc1 = [6.5, 0.5, -5.5, -11.5, -17.5]
Yc1 = [13, 13, 13, 13, 13]
Xc2 = [6.5, 0.5, -5.5, -11.5, -17.5]
Yc2 = [8, 8, 8, 8, 8]

# Control parameters
Kd = 0.1
k = 3
k_cs = 2
r = 2
B_d = 25
B_c = 28
B_d_dumb = 25
B_c_dumb = 28
d_charge = 0.2
alpha = 4

threshold_distance = 0.05

c = [1.5, 2, 1, 1.75, 0.8]
m = [1, 2.5, 2.2, 1.8, 1.5]

p_d = [0] * num_pairs
p_c = [0] * num_pairs

# Flags
needs_charging_d = [False] * num_pairs
needs_charging_i = [False] * num_pairs
start_computing_distancE = [False] * num_pairs

# block flags
vi_computed = [False] * num_pairs
v_cs_computed = [False] * num_pairs

# Distance covered
arc_length_1 = [0] * num_pairs

# parameters for circle outside the charging station
r_cs = d_charge

# auxiliar variables
ed_plot = [[] for _ in range(num_pairs)]
e_plot = [[] for _ in range(num_pairs)]
e_lower_d_plot = np.full(num_steps, E_lower_d)
e_charge_plot = np.full(num_steps, E_charge)
e_min_plot = np.full(num_steps, E_min)

"""ANIMATION LISTS"""
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
arc_length_plot_1 = []

# Empty lists to store trajectory
xdes1_plot = [[] for _ in range(num_pairs)]
ydes1_plot = [[] for _ in range(num_pairs)]
xdes2_plot = [[] for _ in range(num_pairs)]
ydes2_plot = [[] for _ in range(num_pairs)]

def compute_v_cs(x_curr, y_curr, x_cs, y_cs, B_d, K_d, E_lower, E_min):
    d = math.sqrt((x_curr - x_cs)**2 + (y_curr - y_cs)**2)
    offset = 0 # it can't be any arbitrary value, it has to satisfy discriminant b^2 always > 4*a*c
    delta_E = E_lower[i] - E_min - offset

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
    v_cs = max(v1, v2)
    return v_cs

for i in range(num_pairs):

    print(f'Pair: {i}')

    for n in range(len(t)-1):

        timestep = n * dt

        # Desired trajectory for Intelligent (DA) agent
        xdes1 = r * math.sin(c[i]*timestep) + Xc1[i]
        ydes1 = r * math.cos(c[i]*timestep) * math.sin(c[i]*timestep) + Yc1[i]

        # Desired trajectory for Dumb (IA) agent
        xdes2 = r * math.sin(m[i]*timestep) + Xc2[i]
        ydes2 = r * math.cos(m[i]*timestep) * math.sin(m[i]*timestep) + Yc2[i]

        xdes1_dot = r * c[i] * math.cos(c[i]*timestep)
        ydes1_dot = r * c[i] * math.cos(2*c[i]*timestep)
        xdes1_ddot = -r * c[i]**2 * math.sin(c[i]*timestep)
        ydes1_ddot = -2 * r * c[i]**2 * math.sin(2*c[i]*timestep)

        xdes2_dot = r * m[i] * math.cos(m[i]*timestep)
        ydes2_dot = r * m[i] * math.cos(2*m[i]*timestep)
        xdes2_ddot = -r * m[i]**2 * math.sin(m[i]*timestep)
        ydes2_ddot = -2 * r * m[i]**2 * math.sin(2*m[i]*timestep)

        if not needs_charging_d[i] and not needs_charging_i[i]:
            vi_1 = math.sqrt(x2[i]**2 + y2[i]**2)

        ### Controller for intelligent robot to track ###
        e1 = xdes1 - x1[i]
        alpha1 = xdes1_dot + k * e1
        z1 = x2[i] - alpha1
        alpha1_dot = xdes1_ddot - k * z1 - k**2 * e1
        ux_i = e1 + alpha1_dot - k * z1
        
        e2 = ydes1 - y1[i]
        alpha2 = ydes1_dot + k * e2
        z2 = y2[i] - alpha2
        alpha2_dot = ydes1_ddot - k * z2 - k**2 * e2
        uy = e2 + alpha2_dot - k * z2

        U_track = np.array([ux_i, uy])

        vd_1 = math.sqrt(x2d[i]**2 + y2d[i]**2)

        ### Controller for dumb to track the trajectory ###
        ex = xdes2 - x1d[i]
        alpha1_d = xdes2_dot + k * ex
        z1d = x2d[i] - alpha1_d
        alpha1d_dot = xdes2_ddot - k * z1d - k**2 * ex
        ux_d = ex + alpha1d_dot - k * z1d

        ey = ydes2 - y1d[i]
        alpha2d = ydes2_dot + k * ey
        z2d = y2d[i] - alpha2d
        alpha2d_dot = ydes2_ddot - k * z2d - k**2 * ey
        uy_d = ey + alpha2d_dot - k * z2d

        U_track_d = np.array([ux_d, uy_d]) 

        ### controller to take the dumb robot to the CS
        e_csd_x = x_cs[i] - x1d[i]
        sigma_csd1 = k_cs * e_csd_x
        z_csd_x = x2d[i] - sigma_csd1
        sigma_csd1_dot = -k_cs * z_csd_x - k**2 * e_csd_x
        u_csx_d = e_csd_x + sigma_csd1_dot - k * z_csd_x

        e_csd_y = y_cs[i] - y1d[i]
        sigma_csd2 = k_cs * e_csd_y
        z_csd_y = y2d[i] - sigma_csd2
        sigma_csd2_dot_1 = -k_cs * z_csd_y - k**2 * e_csd_y
        u_csy_d = e_csd_y + sigma_csd2_dot_1 - k * z_csd_y

        Ucs_d = np.array([u_csx_d, u_csy_d]) 

        p_d[i] = math.sqrt((x1d[i] - x_cs[i])**2 + (y1d[i] - y_cs[i])**2)
        # p_c[i] = math.sqrt((x1[i] - x_cs[i])**2 + (y1[i] - y_cs[i])**2)

        """needs charging flags part"""
        ### dumb condition
        if Ed[i] <= E_lower_d:
            needs_charging_d[i] = True
        elif Ed[i] >= E_charge:
            needs_charging_d[i] = False

        if needs_charging_d[i]:
            U_dumb = Ucs_d
        else:
            U_dumb = U_track_d

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
        if needs_charging_i[i] and not needs_charging_d[i] and not v_cs_computed[i]:
            v_cs = compute_v_cs(x1[i], y1[i], x_cs[i], y_cs[i], B_d, Kd, E, E_min)
            vi_1 = v_cs
            v_cs_computed[i] = True # block flag, so the computation of view is done just one time
            vec_to_cs = np.array([x_cs[i] - x1[i], y_cs[i] - y1[i]])
            dist_to_cs = np.linalg.norm(vec_to_cs)
            v_vec = v_cs * ( vec_to_cs /  dist_to_cs)
            x2[i], y2[i] = v_vec[0], v_vec[1]
            # print(f'vcs computed: {v_cs}')
            U = np.array([0.0, 0.0])

        """Instant v_red"""
        if needs_charging_i[i] and needs_charging_d[i]:
            T_CD = (E_charge - E_lower_d)  / B_c_dumb
            offset = 0
            v_red = math.sqrt(((E[i] - E_min - offset - T_CD * Kd )) / (T_CD * B_d ))
            vi_1 = v_red
            vi_computed[i] = True
            # print(f'v_red: {vi_1}')
            v_cs_computed[i] = False

        p_c[i] = math.sqrt((x1[i] - x_cs[i])**2 + (y1[i] - y_cs[i])**2)
        if needs_charging_i[i] and p_c[i] <= d_charge and v_cs_computed[i]:
            x2[i], y2[i] = 0.0, 0.0  # Stop motion
            vi_1 = math.sqrt(x2[i]**2 + y2[i]**2)

        if needs_charging_i[i] and needs_charging_d[i]:
            U = U_track
        elif needs_charging_i[i]:  
            U = np.array([0.0, 0.0])
        else: 
            U = U_track

        if p_c[i] > d_charge:
            B = -B_d*(vi_1)**2 - Kd
        else:
            B = B_c

        if abs(xdes1 - x1[i]) <= threshold_distance and abs(ydes1 - y1[i]) <= threshold_distance:
            if start_computing_distancE[i] == False:
                start_computing_distancE[i] = True
                alpha_prev_1 = math.atan2(y1[i] - Yc1[i], x1[i] - Xc1[i])
            else:
                arc_length_1[i] += r * abs( abs( (math.atan2(y1[i] - Yc1[i], x1[i] - Xc1[i]) ) ) - abs(alpha_prev_1) )
                alpha_prev_1 = math.atan2(y1[i] - Yc1[i], x1[i] - Xc1[i])
        else:
            start_computing_distancE[i] = False
        

        # Intelligent's double integrator dynamics
        x1[i] = x1[i] + dt * x2[i]
        x2[i] = x2[i] + dt * U[0] 
        y1[i] = y1[i] + dt * y2[i]
        y2[i] = y2[i] + dt * U[1] 
        E[i] = E[i] + dt * B

        # Dumb's double integrator dynamics
        x1d[i] = x1d[i] + dt * x2d[i] 
        x2d[i] = x2d[i] + dt * U_dumb[0] 
        y1d[i] = y1d[i] + dt * y2d[i] 
        y2d[i] = y2d[i] + dt * U_dumb[1] 
        Ed[i] = Ed[i] + dt * battery_dumb 

        if E[i] < E_min:
            print(f'Energy went below E_min: {E[i]}, Pair: {i}')
            exit()
        
        # print(f'Time : {n*dt}, Ed: {round(Ed, 3)}, E: {round(E,3)}, , vcs_c: {v_cs_computed},  vi_c : {vi_computed}, vi: {round(vi_1,3)}, battery_rate: {round(B,3)}, E_rem: {round(E - (E_min + e_worst_1),4)}')

        # print(f'Time : {round(n*dt,3)}, Ed: {round(Ed, 3)}, , E: {round(E,3)}, Total Distance: {round(arc_length_1,3)}')

        #if args.animation:
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
        xdes1_plot[i].append(xdes1)
        ydes1_plot[i].append(ydes1)
        xdes2_plot[i].append(xdes2)
        ydes2_plot[i].append(ydes2)
        arc_length_plot_1.append(arc_length_1)


ax = plt.subplot()
plt.ylim(3500, 5500)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
agent_colors = [("red","darkred"),("green","darkgreen"),("blue","darkblue"),("violet","darkviolet"),("gold","goldenrod")]

for i in range(num_pairs):
    t_val = [x * dt for x in range(len(Ed_plot[i]))]
    plt.plot(t_val, Ed_plot[i])  
    plt.plot(t_val, E_plot[i])    
    
t_val = [x * dt for x in range(len(t))]
plt.plot(t_val, e_lower_d_plot, linewidth = 2, color = 'c', linestyle="dashed")
plt.plot(t_val, e_charge_plot, linewidth = 2, color = 'g', linestyle="dashed")
plt.plot(t_val, e_min_plot, linewidth = 2, color = 'r', linestyle="dashed")
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend(["$E_{IA,1}$", "$E_{DA,1}$", "$E_{IA,2}$", "$E_{DA,2}$", "$E_{IA,3}$", "$E_{DA,3}$", "$E_{IA,4}$", "$E_{DA,4}$", "$E_{IA,5}$", "$E_{DA,5}$", "$E_{low,IA}$",  "$E_{max}$", "$E_{min}$"], loc="upper center", ncol=7, mode="expand", fontsize="small")
plt.savefig("without_learning_energy_plot_lemniscate.pdf", format="pdf", bbox_inches='tight') 
plt.savefig("without_learning_energy_plot_lemniscate.png", format="png", dpi=600, bbox_inches='tight')  # For high-res PNG
plt.show()



fig, ax = plt.subplots()
ax.set_xlim(-30, 15)
ax.set_ylim(2, 18)
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
    circle1_line, = ax.plot(xdes1_plot[i], ydes1_plot[i], color='red', linestyle = '--')
    circle2_line, = ax.plot(xdes2_plot[i], ydes2_plot[i], color='blue', linestyle = '--') 
    circ_cs = plt.Circle((x_cs[i], y_cs[i]), radius=d_charge, color='green', fill=False)

    ax.add_artist(circle1_line)
    ax.add_artist(circle2_line)
    ax.add_artist(circ_cs)

   # Texts (position offset to avoid overlap)
    y_offset = 0.95 - i * 0.08
    ei_text = ax.text(0, 0, '', fontsize=8, color='black')   # above IA
    nci_text = ax.text(0, 0, '', fontsize=8, color='black')
    ed_text = ax.text(0, 0, '', fontsize=8, color='blue')  # below DA
    ncd_text = ax.text(0, 0, '', fontsize=8, color='blue')
    label = ax.text(Xc1[i], Yc1[i] + r + 1.5, f'Pair {num_pairs - i}', ha='center', fontsize=12, fontweight='bold')
    e_text = ax.text(Xc1[i], Yc1[i] + r + 1, f'$E_{{low}}$: {E_lower_i[i]}', ha='center', fontsize=10)
    e_text_dumb = ax.text(Xc1[i], Yc1[i] + r - 9.5, f'$E_{{low}}$: {e_lower_dumb_disp[i]}', ha='center', fontsize=10, color='blue')

    intelligent_robots.append(ir)
    dumb_robots.append(dr)
    tracking_circles_i.append(circle1_line)
    tracking_circles_d.append(circle2_line)
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
plt.show()

