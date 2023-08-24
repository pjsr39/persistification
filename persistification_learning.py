import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from matplotlib.animation import FuncAnimation

t_span = 50
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

x1i[0] = 10
y1i[0] = 15
x2i[0] = 1
y2i[0] = 1
E[0] = 5500

x1d[0] = 10
y1d[0] = 6
Ed[0] = 5200

Xc1 = 6.5
Yc1 = 14

Xc2 = 6.5
Yc2 = 7

Kd = 20
k = 3
k_cs = 2
kv = 1.5
K_x = 1
K_y = 1
r = 1
B_d = 18
B_c = 100
B_d_dumb = -50
B_c_dumb = 50
k_rho = 6
d_charge = 0.5
alpha = 4
E_charge = 5000
E_lower = 4100
E_min = 3700

p_d = np.zeros(num_steps + 1)
p_c = np.zeros(num_steps + 1)
needs_charging_d = False
needs_charging_i = False
X_cs = np.zeros((2, num_steps))
X_d = np.zeros((2, num_steps))
X_i = np.zeros((2, num_steps))
V_i = np.zeros((2, num_steps))
Xdes_i = np.zeros((2, num_steps))
Xdes_d = np.zeros((2, num_steps))

# parameters for circle outside the charging station
r_cs = d_charge

"""
Learning
"""

# Define sequence parameters
from keras.preprocessing.sequence import TimeseriesGenerator
LOOK_BACK  = 20 # How much past samples it sees
train_generator = []
test_generator = []

# Create neural network and define train parameters
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
NUM_FEATURES = 1
NUM_NEURONS  = 4
LEARNING_RATE = 1e-3
model = Sequential()
model.add(LSTM(NUM_NEURONS, input_shape=(LOOK_BACK, NUM_FEATURES)))
model.add(Dense(1, activation=None))
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mse')
model.summary()
PATIENCE = 100
callback = EarlyStopping(monitor='loss', patience=PATIENCE)
EPOCHS = 300
BATCH_SIZE = 4

# Data parameters
NUM_POINTS = 3000
SPLIT_TRAIN_TEST = 0.8
TRAIN_LENGTH = int(NUM_POINTS * SPLIT_TRAIN_TEST)
GAIN = 0.01
data = []
X_train = []
X_test = []
y_train = []
y_test = []

# Buffer where y values (Ed values) will be stored for learning
Ed_buffer = []

for n in range(len(t)-1):

    print("Current time =", n * dt)
    # Circle around the charging station
    barrier_x = r_cs * np.cos(t) + x_cs
    barrier_y = r_cs * np.sin(t) + y_cs

    X_cs[:,n] = [x_cs[0], y_cs[0]]

    xdes1 = r * np.cos(t) + Xc1
    ydes1 = r * np.sin(t) + Yc1
    Xdes_i[:,n] = [xdes1[n], ydes1[n]]

    xdes2 = r * np.cos(t) + Xc2
    ydes2 = r * np.sin(t) + Yc2
    Xdes_d[:,n] = [xdes2[n], ydes2[n]]

    X_i[:,n] = [x1i[n], y1i[n]]
    X_d[:,n] = [x1d[n], y1d[n]]
    V_i[:,n] = [x2i[n], y2i[n]]
    
    ### Controller for intelligent robot to track ###
    e1 = xdes1[n] - x1i[n]
    alpha1 = -r * np.sin(t) + k * e1
    z1 = x2i[n] - alpha1
    alpha1_dot = -r * np.cos(t) - k * z1 - k**2 * e1
    ux_i = e1 + alpha1_dot - k * z1

    e2 = ydes1[n] - y1i[n]
    alpha2 = r * np.cos(t) + k * e2
    z2 = y2i[n] - alpha2
    alpha2_dot = -r * np.sin(t) - k * z2 - k**2 * e2
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

    if Ed[n] <= E_lower:
        needs_charging_d = True
    elif Ed[n] >= E_charge:
        needs_charging_d = False

    if needs_charging_d:
        U_dumb = Ucs_d
    else:
        U_dumb = U_track_dumb

    if p_d > d_charge:
        battery_dumb = B_d_dumb
    elif p_d <= d_charge:
        battery_dumb = B_c_dumb

    if E[n] <= E_lower:
        needs_charging_i = True
    elif E[n] >= E_charge:
        needs_charging_i = False

    if needs_charging_i:
        U = U_cs_i
    else:
        U = U_track_i

    if p_c > d_charge:
        B = -B_d*abs(vi) - Kd
    else:
        B = B_c

    if Ed[n] < E_charge:
        ## Predicting dumb energy
        if len(Ed_buffer) < NUM_POINTS:
            # collect
            Ed_buffer.append(Ed[n])
            data.append(n*dt)
            print('Collecting Ed:',Ed[n])
        else:
            if len(Ed_buffer) == NUM_POINTS:
                # Define data
                data = np.array(data)
                X_train = data[:TRAIN_LENGTH]
                X_test = data[TRAIN_LENGTH:]
                y_train = np.array(Ed_buffer[:TRAIN_LENGTH])
                y_test = np.array(Ed_buffer[TRAIN_LENGTH:])
                # Normalize
                Y_MIN_TRAIN, Y_MAX_TRAIN = min(y_train), max(y_train)
                Y_MIN_TEST, Y_MAX_TEST = min(y_test), max(y_test)
                y_train = (y_train - Y_MIN_TRAIN) / (Y_MAX_TRAIN - Y_MIN_TRAIN)
                y_test = (y_test - Y_MIN_TEST) / (Y_MAX_TEST - Y_MIN_TEST)
                # Prepare sequence generators
                train_series = y_train.reshape((len(y_train), NUM_FEATURES))
                test_series  = y_test.reshape((len(y_test), NUM_FEATURES))
                train_generator = TimeseriesGenerator(train_series, train_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)
                test_generator = TimeseriesGenerator(test_series, test_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)
                # Train
                model.fit(train_generator,epochs=EPOCHS, verbose=1, callbacks=[callback])
                # Evaluate model
                NUM_EVALUATIONS = 20
                NUM_EVAL_POINTS = LOOK_BACK
                for i in range(NUM_EVALUATIONS):    
                    print(f'EVALUATION [{i}]')
                    x_eval = X_test[i:NUM_EVAL_POINTS+i].reshape(1,NUM_EVAL_POINTS,1)
                    y_eval = y_test[i:NUM_EVAL_POINTS+i].reshape(1,NUM_EVAL_POINTS,1)
                    pred = model.predict(y_eval)[0][0]
                    print('x_eval: ', x_eval.reshape(1,-1))
                    print('y_eval: ', y_eval.reshape(1,-1))
                    print('prediction: ', pred)
                    print('ground_truth: ', y_test[NUM_EVAL_POINTS+i],'\n')

                test_predictions  = model.predict(test_generator)
                XMAX, XMIN = max(data), min(data)
                RESOLUTION = dt
                print('data: ', data)
                print('max, min:', XMAX, XMIN)
                x = np.arange(XMAX - (XMAX - XMIN) * (1 - SPLIT_TRAIN_TEST) + LOOK_BACK * RESOLUTION, XMAX, 1)
                print('x: ', x, x.shape)
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.plot(X_train,y_train/GAIN, lw=2, label='train data')
                ax.plot(X_test,y_test/GAIN, lw=3, c='y', label='test data')
                ax.plot(x,test_predictions/GAIN, lw=3, c='r',linestyle = ':', label='predictions')
                ax.legend(loc="lower left")
                plt.show()
                exit()
            # predict
            print('Will predict')

    x1i[n+1] = x1i[n] + dt * x2i[n]
    x2i[n+1] = x2i[n] + dt * U[0]
    y1i[n+1] = y1i[n] + dt * y2i[n]
    y2i[n+1] = y2i[n] + dt * U[1]
    E[n + 1] = E[n] + dt * B

    x1d[n+1] = x1d[n] + dt * U_dumb[0]
    y1d[n+1] = y1d[n] + dt * U_dumb[1]
    Ed[n + 1] = Ed[n] + dt * battery_dumb

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(xdes1, ydes1)
ax1.plot(xdes2, ydes2)
ax1.plot(x_cs[0], y_cs[0], marker='o')
ax1.plot(barrier_x, barrier_y)

# Initialize Line2D objects for x1i vs y1i and E vs Ed plots
line_x1i_y1i, = ax1.plot([], [], color='blue')
line_x1d_y1d, = ax1.plot([], [], color='magenta')
line_E, = ax2.plot([], [], color='blue', label='E')
line_E_Ed_d, = ax2.plot([], [], color='magenta', label='Ed')
moving_point_i, = ax1.plot([], [], marker='o', color='red', markersize=2)
moving_point_d, = ax1.plot([], [], marker='o', color='magenta', markersize=2)
ax2.legend()

ax2.plot(t, np.full_like(t, E_charge), '--', label='E_charge')
ax2.plot(t, np.full_like(t, E_min), '--', label='E_min')
ax2.plot(t, np.full_like(t, E_lower), '--', label='E_lower')

# Set axis limits for the subplots
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_xlabel('x1i')
ax1.set_ylabel('y1i')

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
    #line_x1i_y1i.set_data(x1i[:frame], y1i[:frame])
    #line_x1d_y1d.set_data(x1d[:frame], y1d[:frame])
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