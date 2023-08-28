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

"""
Learning
"""

# Define sequence parameters
from keras.preprocessing.sequence import TimeseriesGenerator
LOOK_BACK  = 5 # How much past samples it sees
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
PATIENCE = 50
callback = EarlyStopping(monitor='loss', patience=PATIENCE)
EPOCHS = 50
BATCH_SIZE = 2

# Data parameters
NUM_POINTS = 7300
SPLIT_TRAIN_TEST = 0.5
TRAIN_LENGTH = int(NUM_POINTS * SPLIT_TRAIN_TEST)
GAIN = 0.01
data = []
X_train = []
X_test = []
y_train = []
y_test = []

# Buffers
from collections import deque
Ed_buffer = [] # Buffer where y values (Ed values) will be stored for learning
predictions = [] # Buffer of predicted energy value through time
Ed_past = deque(maxlen=LOOK_BACK)

# flags
learned = False

# Forecast parameters
N = 5 # number of predicted values

# Normalization bounds
Y_MAX, Y_MIN = E_charge + 50, E_min + 10
NORMALIZE_LOCAL = True # flag to normalize with local dataset distribution bounds

# Display
TIME_SHOW_DATA = 3

"""
Algorithms and functions
"""
def unnormalize(x):
    """ Auxiliar function that unnormalizes values """
    return x * (Y_MAX - Y_MIN) + Y_MIN
def normalize(x):
    """ Auxiliar function that normalizes values """
    return (x - Y_MIN) / (Y_MAX - Y_MIN)
def unnormalize_local(x, min_, max_):
    """ Auxiliar function that unnormalizes values """
    return x * (max_ - min_) + min_
def normalize_local(x, min_, max_):
    """ Auxiliar function that normalizes values """
    return (x - min_) / (max_ - min_)

for n in range(len(t)-1):

    print("Current time =", n * dt)
    time = n * dt

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

    # Start learning as soon as E[n] < E_charge
    if learned == False and Ed[n] < E_charge:
        # Collect data to learn
        if len(Ed_buffer) < NUM_POINTS:
            Ed_buffer.append(Ed[n])
            data.append(time)
            print('Collecting Ed:',Ed_buffer[-1])
        # As soon as data is collected start learning
        else:
            if len(Ed_buffer) == NUM_POINTS:
                # Define data
                data = np.array(data)
                X_train = data[:TRAIN_LENGTH]
                X_test = data[TRAIN_LENGTH:]
                y_train = np.array(Ed_buffer[:TRAIN_LENGTH])
                y_test = np.array(Ed_buffer[TRAIN_LENGTH:])
                # Get bounds
                Y_MIN_TRAIN, Y_MAX_TRAIN, Y_MIN_TEST, Y_MAX_TEST = min(y_train), max(y_train), min(y_test),max(y_test)
                print(f'# Bounds # \n Y_MIN_TRAIN:{Y_MIN_TRAIN}, Y_MAX_TRAIN:{Y_MAX_TRAIN}, Y_MIN_TEST:{Y_MIN_TEST}, Y_MAX_TEST:{Y_MAX_TEST}')
                # Show data
                fig, ax = plt.subplots(1, 1, figsize=(15, 4))
                ax.plot(X_train,y_train, lw=3, label='train data')
                ax.plot(X_test, y_test,  lw=3, label='test data')
                ax.legend(loc="lower left")
                plt.show(block=False)
                plt.pause(TIME_SHOW_DATA)
                plt.close()
                # Normalize
                if NORMALIZE_LOCAL:
                    y_train = normalize_local(y_train, Y_MIN_TRAIN, Y_MAX_TRAIN)
                    y_test = normalize_local(y_test, Y_MIN_TEST, Y_MAX_TEST)
                else:
                    y_train = normalize(y_train)
                    y_test = normalize(y_test)
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
                    pred_eval = model.predict(y_eval)[0][0]
                    print('x_eval: ', x_eval.reshape(1,-1))
                    print('y_eval: ', y_eval.reshape(1,-1))
                    print('prediction: ', pred_eval)
                    print('ground_truth: ', y_test[NUM_EVAL_POINTS+i])
                    if NORMALIZE_LOCAL:
                        print('#predicted_un: ', unnormalize_local(pred_eval, Y_MIN_TEST, Y_MAX_TEST))
                        print('#ground_truth_un: ', unnormalize_local(y_test[NUM_EVAL_POINTS+i], Y_MIN_TEST, Y_MAX_TEST))
                    else:
                        print('#predicted_un: ', unnormalize(pred_eval))
                        print('#ground_truth_un: ', unnormalize(y_test[NUM_EVAL_POINTS+i]))
                """
                test_predictions  = model.predict(test_generator)
                XMAX, XMIN = max(data), min(data)
                RESOLUTION = dt
                x = np.arange(XMAX - (XMAX - XMIN) * (1 - SPLIT_TRAIN_TEST) + LOOK_BACK * RESOLUTION, XMAX, 1)
                print('x: ', x, x.shape)
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.plot(X_train,y_train/GAIN, lw=2, label='train data')
                ax.plot(X_test,y_test/GAIN, lw=3, c='y', label='test data')
                ax.plot(x,test_predictions/GAIN, lw=3, c='r',linestyle = ':', label='predictions')
                ax.legend(loc="lower left")
                plt.show()
                exit()
                """
                learned = True
                val = input("Press any key to start to predict")
                print(val)
    # After n > NUM_POINTS    
    Ed_past.append(Ed[n]) # Save past Ed for predictions
    Ed_past_normalized = deque(list(map(normalize, Ed_past)),maxlen=LOOK_BACK)
    predictions.clear()
    """ Prediction part """
    if learned:
        #print('\nPredicting')
        # Save the past values normalized in y_past
        y_past = np.array([Ed_past_normalized]).reshape(1,LOOK_BACK,1)
        #print('y_past: ', y_past.reshape(1,LOOK_BACK))

        # Filling the prediction vector
        Ed_past_temp = Ed_past_normalized.copy() # Saving normalized values in Ed_past_temp
        # (n+1),(n+2),...,(n+k+N)
        for k in range(N):
            #if k == 0: # Predicting n+1
            predictions.append(model.predict(y_past, verbose=0)[0][0]) # predictions gets Ed'[n+1]
            Ed_past_temp.append(predictions[-1])
            y_past = np.array([Ed_past_temp]).reshape(1,LOOK_BACK,1)
            #if k > 0:
            #    predictions.append(model.predict(y_past)[0][0]) # predictions -> [Ed'[n+k+1]]
            # print(f'k [{k}] y_past_temp: {y_past.reshape(1,LOOK_BACK)}')
            # print(f'k [{k}] prediction: {predictions[-1]}')
        # print('predictions_actual: ', predictions)
        print('predictions_unnormalized: ', list(map(unnormalize, predictions)))
    
        # If any prediction <= E_lower 
        # then needs_charging_d = True
        if any(unnormalize(pred) <= E_lower for pred in predictions):
            print(f'Leaving the charging station because some {predictions} <= {E_lower}')
            needs_charging_d = True
    
        # If any prediction >= E_charge 
        # then needs_charging_d = False
        if any(unnormalize(pred) >= E_charge for pred in predictions):
            print(f'Going to charging station because some {predictions} >= {E_charge}')
            needs_charging_d = False
    
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