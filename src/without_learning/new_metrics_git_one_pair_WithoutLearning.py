import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from matplotlib.animation import FuncAnimation
import os
import math
import matplotlib.animation as animation
from random import random
import csv

"""Simulation length"""
t_span = 300
dt = 0.01
t = np.arange(0, t_span + dt, dt)
num_steps = len(t)

e_min = 3600
e_init_1 = 5200
e_charge = 5000
e_lower_dumb = 4200      # PV threshold
e_lower_int = 3616       # SV threshold

"""Initial conditions"""
# intelligent = SV
x1 = 10
y1 = 16
x2 = 1
y2 = 1

# dumb = PV
x1d_1 = 10
y1d_1 = 6
x2d_1 = 1
y2d_1 = 1

# Charging station position
x_cs_1 = 8
y_cs_1 = 10.5

# Energy parameters
E = e_init_1          # SV energy
Ed = e_init_1         # PV energy
E_charge = e_charge
E_lower_i = e_lower_int
E_lower_d = e_lower_dumb
E_min = e_min

# Service route centers
Xc1_1 = 8   # SV route center
Yc1_1 = 14
Xc2_1 = 8   # PV route center
Yc2_1 = 7

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

c_1 = 1.5   # SV angular frequency
m_1 = 2     # PV angular frequency

p_d_1 = 0
p_c_1 = 0

t_dis = 0

# Flags
needs_charging_d_1 = False   # PV charging flag
needs_charging_i_1 = False   # SV charging flag
start_computing_distancE = False

# Block flags
vi_computed = False
v_cs_computed = False

# Distance covered
arc_length_1 = 0

# Parameters for charging station region
r_cs = d_charge

# Auxiliary variables
ed_plot_1 = []
e_plot_1 = []
vi_plot_1 = []
e_lower_plot_i_1 = np.full(num_steps, E_lower_i)
e_lower_plot_d_1 = np.full(num_steps, E_lower_d)
e_min_plot_1 = np.full(num_steps, E_min)
e_charge_plot_1 = np.full(num_steps, E_charge)

PRINT_PERIOD = 1000
TIME_SHOW_DATA = 1

"""CODE FLAGS"""
SHOW_PLOTS = True
PLOT_SIMULATION_1 = True
PLOT_FOR_A_BIT = False

"""ANIMATION LISTS"""
x_d_1, y_d_1 = [], []   # PV positions
x_i_1, y_i_1 = [], []   # SV positions
nci_flag_1, ncd_flag_1 = [], []
e_i_1, e_d_1 = [], []
v_i_1 = []
v_d_1 = []
arc_length_plot_1 = []

# ===================== ITS METRICS =====================
sv_service_time = 0.0
cs_occupied_time = 0.0

charging_conflict_events = 0
previous_conflict_state = False

service_tol = 0.10

# ===================== EQ. (2) AND EQ. (5) TERMS =====================
# These are included to reflect the paper equations.
# Set their effective contribution to zero so the numerical results remain unchanged.

ds_energy = 0.1
phi_max = 1.0

sigma_SV = 1
sigma_PV = 1

USE_INTERACTION_IN_ENERGY = True
USE_NOISE_IN_ENERGY = True


def phi_interaction(d, ds=ds_energy, phi_max=phi_max):
    if d >= ds:
        return 0.0
    return phi_max * (1.0 - d / ds)


def effective_interaction(phi_value):
    if USE_INTERACTION_IN_ENERGY:
        return phi_value
    return 0.0


def effective_noise(sigma):
    if USE_NOISE_IN_ENERGY:
        return np.random.normal(0.0, sigma)
    return 0.0

def compute_v_cs(x_curr, y_curr, x_cs, y_cs, B_d, K_d, E_current, E_min):
    d = math.sqrt((x_curr - x_cs)**2 + (y_curr - y_cs)**2)

    if d <= 1e-8:
        return 0.0

    offset = 0
    delta_E = E - E_min - offset

    a = B_d
    b = -(delta_E / d)
    c = K_d

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        print("Not enough energy to reach CS at any speed!")
        return None

    sqrt_disc = math.sqrt(discriminant)
    v1 = (-b + sqrt_disc) / (2 * a)
    v2 = (-b - sqrt_disc) / (2 * a)

    return max(v1, v2)


for n in range(len(t) - 1):

    timestep = n * dt

    # Desired route for SV
    xdes1_1 = r * math.cos(c_1 * timestep) + Xc1_1
    ydes1_1 = r * math.sin(c_1 * timestep) + Yc1_1

    # Desired route for PV
    xdes2_1 = r * math.cos(m_1 * timestep) + Xc2_1
    ydes2_1 = r * math.sin(m_1 * timestep) + Yc2_1

    if not needs_charging_d_1 and not needs_charging_i_1:
        vi_1 = math.sqrt(x2**2 + y2**2)

    # ===================== SV TRACKING CONTROLLER =====================
    e1_1 = xdes1_1 - x1
    alpha1_1 = -c_1 * r * math.sin(c_1 * timestep) + k * e1_1
    z1_1 = x2 - alpha1_1
    alpha1_dot_1 = -c_1**2 * r * math.cos(c_1 * timestep) - k * z1_1 - k**2 * e1_1
    ux_i_1 = e1_1 + alpha1_dot_1 - k * z1_1

    e2_1 = ydes1_1 - y1
    alpha2_1 = c_1 * r * math.cos(c_1 * timestep) + k * e2_1
    z2_1 = y2 - alpha2_1
    alpha2_dot_1 = -c_1**2 * r * math.sin(c_1 * timestep) - k * z2_1 - k**2 * e2_1
    uy_i_1 = e2_1 + alpha2_dot_1 - k * z2_1

    U_track_i_1 = np.array([ux_i_1, uy_i_1])

    # ===================== PV TRACKING CONTROLLER =====================
    vd_1 = math.sqrt(x2d_1**2 + y2d_1**2)

    ex_1 = xdes2_1 - x1d_1
    alpha1d_1 = -m_1 * r * math.sin(m_1 * timestep) + k * ex_1
    z1d_1 = x2d_1 - alpha1d_1
    alpha1d_dot_1 = -m_1**2 * r * math.cos(m_1 * timestep) - k * z1d_1 - k**2 * ex_1
    ux_d_1 = ex_1 + alpha1d_dot_1 - k * z1d_1

    ey_1 = ydes2_1 - y1d_1
    alpha2d_1 = m_1 * r * math.cos(m_1 * timestep) + k * ey_1
    z2d_1 = y2d_1 - alpha2d_1
    alpha2d_dot_1 = -m_1**2 * r * math.sin(m_1 * timestep) - k * z2d_1 - k**2 * ey_1
    uy_d_1 = ey_1 + alpha2d_dot_1 - k * z2d_1

    U_track_dumb_1 = np.array([ux_d_1, uy_d_1])

    # ===================== PV CONTROLLER TO CS =====================
    e_csd_x_1 = x_cs_1 - x1d_1
    sigma_csd1_1 = k_cs * e_csd_x_1
    z_cs_x_1 = x2d_1 - sigma_csd1_1
    sigma_csd1_dot_1 = -k_cs * z_cs_x_1 - k**2 * e_csd_x_1
    u_csx_d_1 = e_csd_x_1 + sigma_csd1_dot_1 - k * z_cs_x_1

    e_csd_y_1 = y_cs_1 - y1d_1
    sigma_csd2_1 = k_cs * e_csd_y_1
    z_csd_y_1 = y2d_1 - sigma_csd2_1
    sigma_csd2_dot_1 = -k_cs * z_csd_y_1 - k**2 * e_csd_y_1
    u_csy_d_1 = e_csd_y_1 + sigma_csd2_dot_1 - k * z_csd_y_1

    Ucs_d_1 = np.array([u_csx_d_1, u_csy_d_1])

    # Distances to CS
    p_d_1 = math.sqrt((x1d_1 - x_cs_1)**2 + (y1d_1 - y_cs_1)**2)  # PV to CS
    p_c_1 = math.sqrt((x1 - x_cs_1)**2 + (y1 - y_cs_1)**2)        # SV to CS

    # ===================== CHARGING FLAGS =====================
    # PV condition
    if Ed <= E_lower_d:
        needs_charging_d_1 = True
    elif Ed >= E_charge:
        needs_charging_d_1 = False

    # SV condition
    if E <= E_lower_i:
        needs_charging_i_1 = True
    elif E >= E_charge:
        needs_charging_i_1 = False
        vi_computed = False

    # PV control selection
    if needs_charging_d_1:
        U_dumb_1 = Ucs_d_1
    else:
        U_dumb_1 = U_track_dumb_1

    # ===================== PV ENERGY DYNAMICS: Eq. (5) =====================
    # Eq. (5): Edot_PV = -rho_PV(t) - phi(dij) - w_PV(t)

    d_pv_sv = math.sqrt((x1d_1 - x1)**2 + (y1d_1 - y1)**2)
    phi_pv_raw = phi_interaction(d_pv_sv)
    phi_pv = effective_interaction(phi_pv_raw)
    w_pv = effective_noise(sigma_PV)

    if p_d_1 > d_charge:
        if n < 15000:
            rho_pv = B_d_dumb
        elif 15000 <= n <= 18000:
            rho_pv = B_d_dumb + 10
        else:
            rho_pv = B_d_dumb + 15

        battery_dumb = -rho_pv - phi_pv - w_pv
    else:
        battery_dumb = B_c_dumb

    # ===================== SV CHARGING / SERVICE LOGIC =====================
    # If SV needs charging and PV does not, SV goes to CS
    if needs_charging_i_1 and not needs_charging_d_1 and not v_cs_computed:
        v_cs = compute_v_cs(x1, y1, x_cs_1, y_cs_1, B_d, Kd, E, E_min)

        if v_cs is not None:
            vi_1 = v_cs
            v_cs_computed = True

            vec_to_cs = np.array([x_cs_1 - x1, y_cs_1 - y1])
            dist_to_cs = np.linalg.norm(vec_to_cs)

            if dist_to_cs > 1e-8:
                v_vec = v_cs * (vec_to_cs / dist_to_cs)
                x2, y2 = v_vec[0], v_vec[1]

            U_1 = np.array([0.0, 0.0])

    # If both need charging, PV has priority and SV returns to service route
    if needs_charging_i_1 and needs_charging_d_1:
        T_CD = (E_charge - E_lower_d) / B_c_dumb
        offset = 0

        val = (E - E_min - offset - T_CD * Kd) / (T_CD * B_d)

        if val > 0:
            v_red = math.sqrt(val)
        else:
            v_red = 0.0

        vi_1 = v_red
        vi_computed = True
        v_cs_computed = False

    # Stop SV if it reaches CS
    if needs_charging_i_1 and p_c_1 <= d_charge and v_cs_computed:
        x2, y2 = 0.0, 0.0
        vi_1 = math.sqrt(x2**2 + y2**2)

    # SV control selection
    if needs_charging_i_1 and needs_charging_d_1:
        U_1 = U_track_i_1
    elif needs_charging_i_1:
        U_1 = np.array([0.0, 0.0])
    else:
        U_1 = U_track_i_1

    # ===================== SV ENERGY DYNAMICS: Eq. (2) =====================
    # Eq. (2): Edot_SV = -c1|v_SV|^2 - c2 - phi(dij) - w_SV(t)

    d_sv_pv = math.sqrt((x1 - x1d_1)**2 + (y1 - y1d_1)**2)
    phi_sv_raw = phi_interaction(d_sv_pv)
    phi_sv = effective_interaction(phi_sv_raw)
    w_sv = effective_noise(sigma_SV)

    if p_c_1 > d_charge:
        c1_sv = B_d
        c2_sv = Kd

        B = -c1_sv * (vi_1)**2 - c2_sv - phi_sv - w_sv
    else:
        B = B_c

    # ===================== ORIGINAL DISTANCE / IDLE-TIME LOGIC =====================
    if abs(xdes1_1 - x1) <= threshold_distance and abs(ydes1_1 - y1) <= threshold_distance:
        if start_computing_distancE is False:
            start_computing_distancE = True
            alpha_prev_1 = math.atan2(y1 - Yc1_1, x1 - Xc1_1)
        else:
            alpha_now_1 = math.atan2(y1 - Yc1_1, x1 - Xc1_1)
            arc_length_1 += r * abs(abs(alpha_now_1) - abs(alpha_prev_1))
            alpha_prev_1 = alpha_now_1
            t_dis += dt
    else:
        start_computing_distancE = False

    # ===================== ITS METRIC COMPUTATION =====================
    pv_at_cs = p_d_1 <= d_charge
    sv_at_cs = p_c_1 <= d_charge

    # 1. SV Service Continuity
    sv_route_error = math.sqrt((x1 - xdes1_1)**2 + (y1 - ydes1_1)**2)
    sv_service_active = (sv_route_error <= service_tol) and (not sv_at_cs)

    if sv_service_active:
        sv_service_time += dt

    # 2. CS Utilization
    if pv_at_cs or sv_at_cs:
        cs_occupied_time += dt

    # 3. Charging Conflict Events
    # Conflict occurs when PV requires charging while SV still occupies the CS.

    current_conflict_state = needs_charging_d_1 and sv_at_cs

    # Count only rising edges
    if current_conflict_state and not previous_conflict_state:
        charging_conflict_events += 1

    previous_conflict_state = current_conflict_state

    # ===================== SV DYNAMICS =====================
    x1 = x1 + dt * x2
    x2 = x2 + dt * U_1[0]
    y1 = y1 + dt * y2
    y2 = y2 + dt * U_1[1]
    E = E + dt * B

    if E < E_min:
        print("SV energy went below E_min")
        exit()

    # ===================== PV DYNAMICS =====================
    x1d_1 = x1d_1 + dt * x2d_1
    x2d_1 = x2d_1 + dt * U_dumb_1[0]
    y1d_1 = y1d_1 + dt * y2d_1
    y2d_1 = y2d_1 + dt * U_dumb_1[1]
    Ed = Ed + dt * battery_dumb

    print(
        f"Time: {n * dt:.2f}, "
        f"Service Distance: {round(arc_length_1, 3)}, "
        f"Idle Time: {round(len(t) * dt - t_dis, 3)}"
    )

    # Save data for animation
    x_d_1.append(x1d_1)
    y_d_1.append(y1d_1)
    x_i_1.append(x1)
    y_i_1.append(y1)

    nci_flag_1.append(needs_charging_i_1)
    ncd_flag_1.append(needs_charging_d_1)

    e_i_1.append(E)
    e_d_1.append(Ed)

    v_d_1.append(vd_1)
    arc_length_plot_1.append(arc_length_1)

    if PLOT_SIMULATION_1:
        ed_plot_1.append(Ed)
        e_plot_1.append(E)


# ===================== FINAL ITS METRICS =====================
total_time = t_span

service_continuity = 100 * sv_service_time / total_time
service_interruption_time = total_time - sv_service_time
cs_utilization = 100 * cs_occupied_time / total_time

print("\n================ ITS METRICS WITHOUT PREDICTION ================")
print(f"SV Service Continuity (%): {service_continuity:.2f}")
print(f"SV Service Interruption Time (s): {service_interruption_time:.2f}")
print(f"CS Utilization (%): {cs_utilization:.2f}")
print(f"Charging Conflict Events: {charging_conflict_events}")
print("=================================================================\n")


"""Show plots and animation"""
if SHOW_PLOTS:

    if PLOT_SIMULATION_1:
        ax = plt.subplot()

        plt.ylim(3500, 5250)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        ax.set_yticks([3614, 3800, 4200, 4400, 4600, 4800, 5000, 5200])

        plt.plot([x * dt for x in range(len(ed_plot_1))], ed_plot_1)
        plt.plot([x * dt for x in range(len(e_plot_1))], e_plot_1)
        plt.plot([x * dt for x in range(len(t))], e_lower_plot_i_1, linestyle='--', linewidth=2, color='m')
        plt.plot([x * dt for x in range(len(t))], e_lower_plot_d_1, linestyle='--', linewidth=2, color='c')
        plt.plot([x * dt for x in range(len(t))], e_min_plot_1, color='r', linewidth=2)
        plt.plot([x * dt for x in range(len(t))], e_charge_plot_1, color='g', linewidth=2)

        plt.xlabel("Time (s)", fontsize=13)
        plt.ylabel("Energy", fontsize=13)
        plt.grid()
        plt.savefig('Energy_Plot_Without_Learning.png')

        if PLOT_FOR_A_BIT:
            plt.show(block=False)
            plt.pause(TIME_SHOW_DATA)
        else:
            plt.show()

        plt.close()

    x = [x for x in range(len(t) - 1)]

    fig, ax = plt.subplots()

    pv_robot_1, = ax.plot([], [], 'bo', markersize=4)
    sv_robot_1, = ax.plot([], [], 'ro', markersize=4)

    circle_sv = plt.Circle((Xc1_1, Yc1_1), r, color='red', fill=False, linestyle='--')
    circle_pv = plt.Circle((Xc2_1, Yc2_1), r, color='blue', fill=False, linestyle='--')
    circle_cs = plt.Circle((x_cs_1, y_cs_1), radius=0.1, color='g', fill=False)

    time_text = ax.text(0.05, 0.90, '', transform=ax.transAxes)
    e_sv_text = ax.text(0.65, 0.65, '', transform=ax.transAxes)
    nch_sv_text = ax.text(0.65, 0.55, '', transform=ax.transAxes)
    e_pv_text = ax.text(0.65, 0.45, '', transform=ax.transAxes)
    nch_pv_text = ax.text(0.65, 0.35, '', transform=ax.transAxes)
    arc_text_1 = ax.text(0.05, 0.80, '', transform=ax.transAxes)

    def init():
        ax.set_xlim(0, 18)
        ax.set_ylim(5, 18)

        ax.add_artist(circle_sv)
        ax.add_artist(circle_pv)
        ax.add_artist(circle_cs)

        handles = [pv_robot_1, sv_robot_1, circle_pv, circle_sv, circle_cs]
        labels = ['$PV_1$', '$SV_1$', 'PV service route', 'SV service route', 'CS']

        ax.legend(handles, labels, loc="upper right", ncol=3, fontsize=8)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')

        return pv_robot_1, sv_robot_1, time_text, circle_sv, circle_pv, circle_cs, arc_text_1

    def animate(i):
        pv_robot_1.set_data([x_d_1[i]], [y_d_1[i]])
        sv_robot_1.set_data([x_i_1[i]], [y_i_1[i]])

        time_text.set_text('Time = {:.2f}'.format(i * dt))

        e_sv_text.set_text('$E_{{SV_1}}$: {:.3f}'.format(e_i_1[i]))
        nch_sv_text.set_text('$n_{{ch,SV_1}}$: {}'.format(nci_flag_1[i]))

        e_pv_text.set_text('$E_{{PV_1}}$: {:.3f}'.format(e_d_1[i]))
        nch_pv_text.set_text('$n_{{ch,PV_1}}$: {}'.format(ncd_flag_1[i]))

        arc_text_1.set_text('SV Service Distance = {:.2f}'.format(arc_length_plot_1[i]))

        return (
            pv_robot_1, sv_robot_1, time_text,
            circle_sv, circle_pv, circle_cs,
            e_sv_text, nch_sv_text,
            e_pv_text, nch_pv_text,
            arc_text_1
        )

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(x),
        init_func=init,
        interval=10,
        blit=True,
        repeat=True
    )

    plt.show()