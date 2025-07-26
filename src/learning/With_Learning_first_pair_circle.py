import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from matplotlib.animation import FuncAnimation
import os
import math
import argparse
from copy import deepcopy
import calendar
import time
import os
import pandas as pd
import csv

"""Args"""
parser = argparse.ArgumentParser()
parser.add_argument('--e_min', type=float, required=False, default=3600)
parser.add_argument('--e_init', type=float, required=False, default=5200)
parser.add_argument('--e_lower_int', type=float, required=False, default=3616) # the only mandatory parameter set by user
parser.add_argument('--e_lower_dumb', type=float, required=False, default=4200)
parser.add_argument('--e_charge', type=float, required=False, default=5000)
parser.add_argument('--out_file', type=str, required=False, default='')
parser.add_argument('--animation', action="store_true")
args = parser.parse_args()

""" CSV Headers """
csv_header = [
    'timestep', 'e_d', 'ncd_flag', 'dwnc_flag', 'e_i', 'nci_flag', 
    'v_i', 'arc_length', 'xi', 'yi', 'xd', 'yd'
]
csv_path = os.path.join(os.getcwd(),'results_with_learning_first_pair.csv')
with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)  # Write header

"""Simulation length"""
t_span = 300
dt = 0.01
t = np.arange(0, t_span + dt, dt)
num_steps = len(t)

""" Initial intelligent state """
x1i = 10
y1i = 16
x2i = 1
y2i = 1

""" Initial dumb state """
x1d = 10
y1d = 6
x2d = 1
y2d = 1

""" Charging state position """
x_cs = 8
y_cs = 10.5

""" Energy parameters """
E = args.e_init
Ed = args.e_init
E_charge = args.e_charge
E_lower_i = args.e_lower_int
E_lower_d = args.e_lower_dumb
E_min = args.e_min

""" Track roundabout positions """
Xc1 = 8
Yc1 = 14
Xc2 = 8
Yc2 = 7

""" Control parameters """
Kd = 0.1
k = 3
k_cs = 2
r = 1.5
B_d = 25
B_c = 28 # default: 20
B_d_dumb = 25
B_c_dumb = 28 # default: 23
d_charge = 0.2
alpha = 4

threshold_distance = 0.015 #default = 0.015

c = 1.5
m = 2

p_d = 0
p_c = 0

std_dev = 0
mean = 0

"""
Flags
"""
needs_charging_d = False
needs_charging_i = False
start_computing_distance = False

""" block flags """
dumb_reached_e_lower = False
int_reached_e_lower = False
vi_computed = False
vcs_computed = False

""" Auxiliar variables """
arc_length = 0

# parameters for circle outside the charging station
r_cs = d_charge

# Time taken by dumb to charge
td = 0

# auxiliar variables
ed_plot = []
e_plot = []
vi_plot = []
e_lower_plot_i = np.full(num_steps, E_lower_i)
e_lower_plot_d = np.full(num_steps, E_lower_d)
e_min_plot = np.full(num_steps, E_min)
e_charge_plot = np.full(num_steps, E_charge)
went_below_emin = False

t_dis = 0 # time it covered distance


def compute_v_cs(x_curr, y_curr, x_cs, y_cs, B_d, K_d, E, E_min):
    d = math.sqrt((x_curr - x_cs)**2 + (y_curr - y_cs)**2)
    offset = 0 # it can't be any arbitrary value, it has to satisfy discriminant b^2 always > 4*a*c
    delta_E = E - E_min - offset

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


"""Paths"""
wd = os.path.join(os.getcwd())

"""Learning"""
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Disable warning messages of tensorflow
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.backend import set_learning_phase
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, AdamW, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error
from collections import deque
import tensorflow as tf

# Tensor board
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(wd, "tb_logs"))

# Set seed
seed_obj = np.random.RandomState(9999)

# Define sequence parameters
LOOK_BACK  = 8 # How much past samples it sees
train_generator = []
test_generator = []

# Create neural network and define train parameters
NUM_FEATURES = 1
NUM_NEURONS  = 8 # default: 4
LEARNING_RATE = 1e-3 # default: 1e-3

"""
Network
"""
""" Model """
model = Sequential()
model.add(GRU(NUM_NEURONS))
model.add(Dense(1, activation=None))

""" Optimizer """
optimizer = Adam(learning_rate=LEARNING_RATE)

""" Compile """
model.compile(optimizer=optimizer, loss=mean_squared_error)

""" Early Stopping """
PATIENCE = 20
early_stopping_callback = EarlyStopping(monitor='loss', patience=PATIENCE) # should be monitor='val_loss'. for that it needs validation data

""" Batch Size """
BATCH_SIZE = 32 # the bigger the batch size, the more epochs we need
EPOCH_UNIT = 200 # default 400
EPOCHS = EPOCH_UNIT * BATCH_SIZE # some empirical criteria to define batch size
EPOCHS_ONLINE_TRAINING = 5

""" Epoch counter """
EPOCH_COUNTER = 0 # it counts epochs during throughout offline and online learning

""" Tensorboard writer """
tb_writer = tf.summary.create_file_writer(os.path.join(wd, "tb_logs"))

""" Model checkpoint (to save the best model during learning) """
checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

""" Loss callback on epoch end """
class LossCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        global EPOCH_COUNTER
        with tb_writer.as_default(step=EPOCH_COUNTER):
            tf.summary.scalar(name='train_loss', data=logs["loss"], step=EPOCH_COUNTER, description="offline and online training loss")
        EPOCH_COUNTER += 1


loss_callback = LossCallback()

# Data parameters
NUM_POINTS = 8250
INITIAL_PART_DURATION = 500
SPLIT_TRAIN_TEST = 0.7
GAIN = 0.01
data = []
X_train = []
X_test = []
y_train = []
y_test = []

# Buffers
Ed_buffer = [] # Buffer where y values (Ed values) will be stored for learning # TODO still important?
predictions = [] # Buffer of predicted energy value through time
Ed_past = deque(maxlen=LOOK_BACK)

# flags 
"""
To active learning:
learned = False
predict = True
LEARN = True
"""
learned = False # already learned should be True
predict = True
LEARN = True

# Forecast parameters
N = 3  # number of predicted values

# Normalization bounds
Y_MAX, Y_MIN = E_charge + 5, E_lower_d - 40
NORMALIZE_LOCAL = False # flag to normalize with local dataset distribution bounds

# Display
TIME_SHOW_DATA = 200

"""
Auxiliar functions
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

"""
Logging
"""
current_GMT = time.gmtime()
ts = str(calendar.timegm(current_GMT)) # Timestamp for the logging folder
log_path = os.path.join(wd, "log")
if not os.path.isdir(log_path):
    os.mkdir(os.path.join(log_path))
folder_name = 'log_' + ts
os.mkdir(os.path.join(log_path, folder_name))
folder_dir = os.path.join(log_path, folder_name)
logging.basicConfig(filename=os.path.join(folder_dir,"log.txt"),
                    level=logging.DEBUG,
                    format="%(asctime)s %(message)s",
                    )

PRINT_PERIOD = 100 # each PRINT_PERIOD steps, print whatever
PRINT_PREDICTIONS_INFO = False

vred_filename = "vred"

"""CODE FLAGS"""
PLOT_SIMULATION = True
PLOT_FOR_A_BIT = False
case_1_flag = False
case_2_flag = False
MUST_COUNT = True

"""AUX VARIABLES"""
count = 0
proof = ""

"""COLLECTION PARAMETERS"""
COLLECTION_PERIOD = 150

"""ANIMATION LISTS"""
x_d, y_d = [], []
x_i, y_i = [], []
nci_flag, ncd_flag = [], []
dwnc_flag, dwrel_flag = [], []
e_i, e_d = [], []
v_i = []
v_d = []

xdes_i = []
ydes_i = []
xdes_d = []
ydes_d = []
arc_length_plot_1 = []

"""
(A) Dumb simulation cycle (to collect data OFFLINE)
"""
for n in range(NUM_POINTS + INITIAL_PART_DURATION):
    if n % PRINT_PERIOD == 0:
        print(f"(Dumb simulation. Collecting samples) | Current time: {n * dt}")
    timestep = n * dt

    vd = math.sqrt(x2d**2 + y2d**2)

    xdes2 = r * math.cos(m*timestep) + Xc2
    ydes2 = r * math.sin(m*timestep) + Yc2

    xdes2_dot = - r * m * math.sin(m*timestep)
    ydes2_dot =   r * m * math.cos(m*timestep)
    xdes2_ddot = - r * m**2 * math.cos(m*timestep)
    ydes2_ddot = - r * m**2 * math.sin(m*timestep)

    ### Controller for dumb to track the trajectory ###
    ex = xdes2 - x1d
    alpha1d = xdes2_dot + k * ex
    z1d = x2d - alpha1d
    alpha1d_dot = xdes2_ddot - k * z1d - k**2 *ex
    ux_d = ex + alpha1d_dot - k * z1d

    ey = ydes2 - y1d
    alpha2d = ydes2_dot + k * ey
    z2d = y2d - alpha2d
    alpha2d_dot = ydes2_ddot - k * z2d - k**2 *ey
    uy_d = ey + alpha2d_dot - k * z2d

    U_track_dumb = np.array([ux_d, uy_d]) # TODO could be python list

    ### Controller for dumb to take to the charging station
    e_csd_x = x_cs - x1d
    sigma_csd1 = k_cs * e_csd_x
    z_csd_x = x2d - sigma_csd1
    sigma_csd1_dot = -k_cs * z_csd_x - k**2 * e_csd_x
    u_csx_d = e_csd_x + sigma_csd1_dot - k * z_csd_x

    e_csd_y = y_cs - y1d
    sigma_csd2 = k_cs * e_csd_y
    z_csd_y = y2d - sigma_csd2
    sigma_csd2_dot = -k_cs * z_csd_y - k**2 * e_csd_y
    u_csy_d = e_csd_y + sigma_csd2_dot - k * z_csd_y

    Ucs_d = np.array([u_csx_d, u_csy_d]) # TODO could be python list

    p_d = math.sqrt((x1d - x_cs)**2 + (y1d - y_cs)**2)

    """needs charging flags part"""
    ### dumb condition
    if Ed <= E_lower_d:
        needs_charging_d = True
    elif Ed >= E_charge:
        needs_charging_d = False

    if needs_charging_d:
        U_dumb = Ucs_d
    else:
        U_dumb = U_track_dumb

    if p_d > d_charge:
        cs_occupied_d = False
        if n < 15000:
            battery_dumb = -B_d_dumb
        elif 15000 <= n <= 18000:
            battery_dumb = -B_d_dumb - 10
        else:
            battery_dumb = -B_d_dumb - 15
    elif p_d <= d_charge:
        cs_occupied_d = True
        battery_dumb = B_c_dumb

    """ Dynamic part """
    x1d = x1d + dt * x2d 
    x2d = x2d + dt * U_dumb[0]
    y1d = y1d + dt * y2d
    y2d = y2d + dt * U_dumb[1]

    Ed = Ed + dt * battery_dumb

    INITIAL_PART_PASSED = Ed < E_charge # @TODO temporary thing
    """ Collect if INITIAL PART PASSED """
    if INITIAL_PART_PASSED:
        if len(Ed_buffer) < NUM_POINTS:
            if n % COLLECTION_PERIOD == 0:
                Ed_buffer.append(Ed) # Here i save the energies
                data.append(timestep) # Here I save the timesteps


SHOW_DATASET = False
if SHOW_DATASET:
    print(len(Ed_buffer))
    print('data: ', data, 'len: ', len(data))
    print('eds: ', Ed_buffer, 'len: ', len(Ed_buffer))
    plt.figure('Data to learn')
    plt.plot([x for x in range(len(Ed_buffer))], list(Ed_buffer))
    plt.pause(2)
    plt.show(block=False)
    plt.close()
    plt.clf()
    plt.cla()
DATASET_LEN = len(data) # dataset should not vary according to collection period. TODO improve this

"""
(B) Learn OFFLINE
If collected all data start learning OFFLINE
"""
"""
(B1) Learn canonical wave section. This is just a debug section.
"""
from scipy import signal
LEARN_DUMMY = False # If LEARN_DUMMY = True we enter (B1) section and continue after that
LEARN_SINE = False
NOISE = False
if LEARN_DUMMY:
    points = 800
    periods = points / 200
    period = points/periods
    x_triangle = np.array([xt for xt in range(points)])
    #x_triangle = x_triangle / max(x_triangle) # normalize

    y_triangle = np.sin(2*np.pi*x_triangle) if LEARN_SINE else signal.sawtooth(2*np.pi*x_triangle / period, width=0.5) # width=0.5 is a triangle wave

    print('x_triangle:',x_triangle)
    noise = np.random.normal(0,1,len(y_triangle)) * 1e-2
    print('y_triangle:',y_triangle)
    if NOISE: y_triangle = y_triangle + noise

    plt.plot(x_triangle, y_triangle)
    plt.pause(3)
    plt.show(block=False)
    plt.close()
    plt.clf()
    plt.cla()

    X_triangle_train = x_triangle.copy()
    X_triangle_test = x_triangle.copy()

    y_triangle_train = y_triangle.copy()
    y_triangle_test = y_triangle.copy()

    train_series = y_triangle_train.reshape((len(y_triangle_train), NUM_FEATURES))
    test_series  = y_triangle_test.reshape((len(y_triangle_test), NUM_FEATURES))
    print(f'train_series:{train_series}')
    print(f'test_series:{test_series}')

    train_generator = TimeseriesGenerator(train_series, train_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)
    model.fit(train_generator,epochs=EPOCHS, verbose=1, callbacks=[early_stopping_callback, model_checkpoint_callback, loss_callback])

    model.load_weights(checkpoint_filepath) # Load the best model

    # Evaluate model
    NUM_EVAL_POINTS = LOOK_BACK
    NUM_EVALUATIONS = len(x_triangle) - LOOK_BACK - 1 # 100 is the limit = period
    abs_error = []
    y_gt = []
    preds_wave_eval = []
    for i in range(NUM_EVALUATIONS):
        x_eval = X_triangle_test[i:NUM_EVAL_POINTS+i].reshape(1,NUM_EVAL_POINTS,1)
        y_eval = y_triangle_test[i:NUM_EVAL_POINTS+i].reshape(1,NUM_EVAL_POINTS,1)

        pred_eval = keras.backend.get_value(model(y_eval, training = False)[0][0])

        gt = y_triangle_test[NUM_EVAL_POINTS+i]
        next_val = keras.backend.get_value(pred_eval)
        error_abs = np.abs(next_val - gt)

        gt_norm = gt * (Y_MAX-Y_MIN)+Y_MIN
        next_val_norm = next_val * (Y_MAX-Y_MIN)+Y_MIN
        error_abs_norm = np.abs(next_val_norm - gt_norm)

        print(f'({i}) | {next_val} | {gt} | {error_abs}')
        print(f'({i}) | {next_val_norm} | {gt_norm} | {error_abs_norm}')
        print()

        y_gt.append(gt)
        abs_error.append(abs(next_val - gt))
        preds_wave_eval.append(next_val)
    x_triangle_eval = x_triangle[:NUM_EVALUATIONS]

    plt.title('Error per x')
    plt.scatter(x_triangle_eval[:int(len(x_triangle_eval)*0.5)], abs_error[:int(len(x_triangle_eval)*0.5)], color='blue') # Upwards
    plt.scatter(x_triangle_eval[int(len(x_triangle_eval)*0.5):], abs_error[int(len(x_triangle_eval)*0.5):], color='red') # Dowards
    plt.show()
    plt.title('Learned wave')
    plt.scatter(x_triangle_eval, preds_wave_eval)
    plt.plot(x_triangle_eval, y_triangle[:NUM_EVALUATIONS])
    plt.show()

if LEARN:
    """ Define data """
    data = np.array(data)
    data = data - min(data) # offset it to start at zero

    X_train = data.copy() # data[:int(len(data) * SPLIT_TRAIN_TEST)]
    X_test = data.copy() # data[int(len(data) * SPLIT_TRAIN_TEST):]

    y_train = np.array(deepcopy(Ed_buffer)) # np.array(Ed_buffer[:int(len(data) * SPLIT_TRAIN_TEST)])
    y_test = np.array(deepcopy(Ed_buffer)) # np.array(Ed_buffer[int(len(data) * SPLIT_TRAIN_TEST):])

    """ Get bounds """
    Y_MIN_TRAIN, Y_MAX_TRAIN, Y_MIN_TEST, Y_MAX_TEST = min(y_train), max(y_train), min(y_test),max(y_test)
    print(f'# Bounds # \n Y_MIN_TRAIN:{Y_MIN_TRAIN}, Y_MAX_TRAIN:{Y_MAX_TRAIN}, Y_MIN_TEST:{Y_MIN_TEST}, Y_MAX_TEST:{Y_MAX_TEST}')

    """ Show data """
    if SHOW_DATASET:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(X_train,y_train, lw=3, label='train data')
        ax.plot(X_test, y_test,  lw=3, label='test data')
        ax.legend(loc="lower left")
        plt.show()
        plt.pause()
        plt.close()

    """ Normalize data """
    if NORMALIZE_LOCAL:
        y_train = normalize_local(y_train, Y_MIN_TRAIN, Y_MAX_TRAIN)
        y_test = normalize_local(y_test, Y_MIN_TEST, Y_MAX_TEST)
    else:
        y_train = normalize(y_train)
        y_test = normalize(y_test)

    """ Prepare sequence generators """
    train_series = y_train.reshape((len(y_train), NUM_FEATURES)) # e.g.,: [E[0], E[1], E[2], ...]
    test_series  = y_test.reshape((len(y_test), NUM_FEATURES))
    print(f'train_series:{train_series}')
    print(f'test_series:{test_series}')
    train_generator = TimeseriesGenerator(train_series, train_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)
    test_generator = TimeseriesGenerator(test_series, test_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)

    """ Train """
    print('>> Learning part started')

    model.fit(train_generator,epochs=EPOCHS, verbose=1, callbacks=[early_stopping_callback, model_checkpoint_callback, loss_callback])

    print('Learned')

    model.load_weights(checkpoint_filepath) # Load the best model

    learned = True # Setting this to true to declare that it finished learning

"""
(B2) Evaluate
"""
if LEARN and learned: # if finished learning, then evaluate
    NUM_EVAL_POINTS = LOOK_BACK # How many points I look back to predict
    NUM_EVALUATIONS = len(data) - LOOK_BACK - 1 # 100 is the limit = period
    abs_errors = [] # absolute errors list
    abs_errors_unnorm = []
    gts = []
    gts_unnorm = []
    pred_vals = []
    pred_vals_unnorm = []
    for i in range(NUM_EVALUATIONS):
        x_eval = X_test[i:NUM_EVAL_POINTS+i].reshape(1,NUM_EVAL_POINTS,1) # x for evaluation
        y_eval = y_test[i:NUM_EVAL_POINTS+i].reshape(1,NUM_EVAL_POINTS,1) # y for evaluation

        """ Predict t+N value """
        pred_eval = keras.backend.get_value(model(y_eval, training = False)[0][0]) # predict t+N
        pred_val = keras.backend.get_value(pred_eval) # prediction
        pred_val_unnorm = pred_val * (Y_MAX-Y_MIN) + Y_MIN

        """ Save t+N ground truth to compare """
        gt = y_test[NUM_EVAL_POINTS+i] # ground truth
        gt_unnorm = gt * (Y_MAX-Y_MIN) + Y_MIN

        error_abs = np.abs(pred_val - gt) # abs error between prediction and grouth truth
        error_abs_unnorm = np.abs(pred_val_unnorm - gt_unnorm)

        print(f'({i}) | {pred_val} | {gt} | {error_abs}')
        print(f'({i}) | {pred_val_unnorm} | {gt_unnorm} | {error_abs_unnorm}')
        print()

        gts.append(gt)
        gts_unnorm.append(gt_unnorm)
        abs_errors.append(error_abs)
        abs_errors_unnorm.append(error_abs_unnorm)
        pred_vals.append(pred_val)
        pred_vals_unnorm.append(pred_val_unnorm)
        #abs_error_and_energy.append((gts[-1], abs_error[-1]))
    x_evals = X_test[NUM_EVAL_POINTS:NUM_EVALUATIONS+NUM_EVAL_POINTS]

    """ Plot error of evaluations """
    """ Normalized """
    plt.close(); plt.clf(); plt.cla()

    plt.title('Error per x and learned wave normalized')
    plt.plot(x_evals, abs_errors, c='red') # Error
    plt.plot(x_evals, pred_vals, c='black') # Predictions
    plt.plot(x_evals, gts, c='blue') # ground truths
    
    plt.legend(["error", "predictions", "ground truth"], loc ="upper right")
    plt.grid()
    plt.show()
    
    """ Raw """
    plt.close(); plt.clf(); plt.cla()
    plt.title('Error per x and learned wave raw')
    plt.plot(x_evals, abs_errors_unnorm, c='red') # Error
    plt.plot(x_evals, pred_vals_unnorm, c='black') # Predictions
    plt.plot(x_evals, gts_unnorm, c='blue') # ground truths 
    plt.legend(["error", "predictions", "ground truth"], loc ="upper right")
    plt.grid()
    plt.show()

"""
(C) Reset and initialize some new variables
"""
n = 0
x1d = 10
y1d = 6
x2d = 1
y2d = 1
vd = 0
needs_charging_d = False
dumb_reached_e_lower = False
Ed = args.e_init
INITIAL_PART_PASSED = False # maybe not important
learned = True # TODO it should predict from the start (True) or predict after relearning starts (False)?

""" also reset learning dataset buffers """
data = deque(maxlen=DATASET_LEN) # timesteps
X_train = []
X_test = []
y_train = []
y_test = []
Ed_buffer = deque(maxlen=DATASET_LEN) # Buffer where y values (Ed values) will be stored for learning
predictions = [] # Buffer of predicted energy value through time

"""(D) Main loop"""
wncd = False # will need charge dumb flag
dwrel = False # dumb will reach e_lower
Ed_value_down = 0
dumb_instant_energy = False
vi_1 = 0 # vi starts at zero

DATASET_READY = False

PRINT_PERIOD = 10 # put print period more frequent because main loop takes more time
logging.debug("The program started")
logging.debug(f'Running for E_lower_i: {E_lower_i}, E_min: {E_min}, E_charge: {E_charge}, E_init: {args.e_init}, t_span: {t_span}')
print(f'Running for E_lower_i: {E_lower_i}, E_min: {E_min}, E_charge: {E_charge}, E_init: {args.e_init}, t_span: {t_span}')
for n in range(len(t)-1):
    if n % PRINT_PERIOD == 0:
        print(f"Time: {round(n * dt, 1)}, Dist: {round(arc_length, 3)}, Ed: {round(Ed,3)}, ncd: {needs_charging_d}, wncd: {wncd}, E: {round(E, 3)}, nci: {needs_charging_i}, battery_dumb: {battery_dumb}, vi: {vi_1}, vcs_c: {vcs_computed}, vi_c:{vi_computed}")
        #print(f"Time: {round(n * dt, 1)}")
    
    timestep = n * dt
    logging.debug(f"iteration: {n}, time: {timestep}, E: {E}, Ed: {Ed}, arc_length: {arc_length}, vi: {vi_1}")

    # Circle around the charging station
    barrier_x = r_cs * math.cos(timestep) + x_cs
    barrier_y = r_cs * math.sin(timestep) + y_cs

    xdes1 = r * math.cos(c*timestep) + Xc1
    ydes1 = r * math.sin(c*timestep)+ Yc1

    xdes2 = r * math.cos(m*timestep) + Xc2
    ydes2 = r * math.sin(m*timestep) + Yc2

    xdes1_dot = - r * c * math.sin(c*timestep)
    ydes1_dot =   r * c * math.cos(c*timestep)
    xdes1_ddot = - r * c**2 * math.cos(c*timestep)
    ydes1_ddot = - r * c**2 * math.sin(c*timestep)

    xdes2_dot = - r * m * math.sin(m*timestep)
    ydes2_dot =   r * m * math.cos(m*timestep)
    xdes2_ddot = - r * m**2 * math.cos(m*timestep)
    ydes2_ddot = - r * m**2 * math.sin(m*timestep)

    # xdes_i.append(xdes1)
    # ydes_i.append(ydes1)
    # xdes_d.append(xdes2)
    # ydes_d.append(ydes2)

    if not needs_charging_d and not needs_charging_i:
        vi_1 = math.sqrt(x2i**2 + y2i**2)

    vd = math.sqrt(x2d**2 + y2d**2)

    ### Controller for intelligent robot to track ###
    e1 = xdes1 - x1i
    alpha1 = xdes1_dot + k * e1
    z1 = x2i - alpha1
    alpha1_dot = xdes1_ddot - k * z1 - k**2 * e1
    ux_i = e1 + alpha1_dot - k * z1

    e2 = ydes1 - y1i
    alpha2 = ydes1_dot + k * e2
    z2 = y2i - alpha2
    alpha2_dot = ydes1_ddot - k * z2 - k**2 * e2
    uy_i = e2 + alpha2_dot - k * z2

    U_track_i = np.array([ux_i, uy_i]) # @TODO this could be python list or simply using ux_i, uy_i

    ### controller to take the int. robot to the CS
    # e_csx = x_cs - x1i
    # sigma_cs1 = k_cs * e_csx
    # z_csx = x2i - sigma_cs1
    # sigma_cs1dot = -k_cs * z_csx - k**2 * e_csx
    # u_csx_i = e_csx + sigma_cs1dot - k * z_csx

    # e_csy = y_cs - y1i
    # sigma_cs2 = k_cs * e_csy
    # z_csy = y2i - sigma_cs2
    # sigma_cs2dot = -k_cs * z_csy - k**2 * e_csy
    # u_csy_i = e_csy + sigma_cs2dot - k * z_csy

    # U_cs_i = np.array([u_csx_i, u_csy_i]) # @TODO this could be python list as in U_track_i

    ### Controller for dumb to track the trajectory ###
    ex = xdes2 - x1d
    alpha1d = xdes2_dot + k * ex
    z1d = x2d - alpha1d
    alpha1d_dot = xdes2_ddot - k * z1d - k**2 *ex
    ux_d = ex + alpha1d_dot - k * z1d

    ey = ydes2 - y1d
    alpha2d = ydes2_dot + k * ey
    z2d = y2d - alpha2d
    alpha2d_dot = ydes2_ddot - k * z2d - k**2 *ey
    uy_d = ey + alpha2d_dot - k * z2d

    U_track_dumb = np.array([ux_d, uy_d]) # TODO could be python list

    ### Controller for dumb to take to the charging station
    e_csd_x = x_cs - x1d
    sigma_csd1 = k_cs * e_csd_x
    z_csd_x = x2d - sigma_csd1
    sigma_csd1_dot = -k_cs * z_csd_x - k**2 * e_csd_x
    u_csx_d = e_csd_x + sigma_csd1_dot - k * z_csd_x

    e_csd_y = y_cs - y1d
    sigma_csd2 = k_cs * e_csd_y
    z_csd_y = y2d - sigma_csd2
    sigma_csd2_dot = -k_cs * z_csd_y - k**2 * e_csd_y
    u_csy_d = e_csd_y + sigma_csd2_dot - k * z_csd_y

    Ucs_d = np.array([u_csx_d, u_csy_d]) # TODO could be python list

    p_d = math.sqrt((x1d - x_cs)**2 + (y1d - y_cs)**2)
    p_c = math.sqrt((x1i - x_cs)**2 + (y1i - y_cs)**2)

    """needs charging flags part"""
    ### dumb condition
    if Ed <= E_lower_d:
        needs_charging_d = True
    elif Ed >= E_charge:
        needs_charging_d = False
        vcs_computed = False
        vi_computed = False
        wncd = False

    if needs_charging_d:
        U_dumb = Ucs_d
    else:
        U_dumb = U_track_dumb

    if p_d > d_charge:
        # cs_occupied_d = False
        if n < 15000:
            battery_dumb = -B_d_dumb
        elif 15000 <= n <= 18000:
            battery_dumb = -B_d_dumb - 10
        else:
            battery_dumb = -B_d_dumb - 15
    elif p_d <= d_charge:
        battery_dumb = B_c_dumb

    ### intelligent condition
    if E <= E_lower_i:
        needs_charging_i = True
    elif E >= E_charge:
        needs_charging_i = False
        vi_computed = False

    """ Save Eds """
    Ed_past.append(Ed) # Save past Ed for predictions. It is a deque with MAXLEN = LOOK_BACK

    """ Reset prediction list """
    predictions.clear() # clear buffer of predictions to start new prediction phase

    """ Write Ed to tensorboard """
    with tb_writer.as_default(step=n):
        tf.summary.scalar(name='Ed', data=Ed, step=n, description="Energy of dumb robot")
        tf.summary.scalar(name='Ei', data=E, step=n, description="Energy of int robot")
        tf.summary.scalar(name='B_d_dumb', data=B_d_dumb, step=n, description="Rate of discharge of dumb")

    """(D0) Online learning part"""
    INITIAL_PART_PASSED = Ed < E_charge
    """ Collect if INITIAL PART PASSED """
    if INITIAL_PART_PASSED:
        if n % COLLECTION_PERIOD == 0: # each COLLECTION_PERIOD save data in dataset buffers
            Ed_buffer.append(Ed) # Here i save the energies
            data.append(timestep) # Here I save the timesteps
            print(f'Collected new sample ({timestep},{Ed})  {len(data)}/{DATASET_LEN}')

    if not DATASET_READY and len(data) == DATASET_LEN:
        print('Dataset is ready')
        DATASET_READY = True

    if DATASET_READY:
        """ Collecting new sample """
        Ed_buffer.append(Ed) # Here i save the energies
        data.append(timestep) # Here I save the timesteps
        #print(f'Collected new sample ({timestep},{Ed})  {len(data)}/{DATASET_LEN}')

        """ Define data """
        data = np.array(data) # transform from deque to numpy array to be able to subtract it from min
        data = data - min(data) # offset it to start at zero

        X_train = data.copy() # data[:int(len(data) * SPLIT_TRAIN_TEST)]
        X_test = data.copy() # data[int(len(data) * SPLIT_TRAIN_TEST):]

        y_train = np.array(deepcopy(Ed_buffer)) # np.array(Ed_buffer[:int(len(data) * SPLIT_TRAIN_TEST)])
        y_test = np.array(deepcopy(Ed_buffer)) # np.array(Ed_buffer[int(len(data) * SPLIT_TRAIN_TEST):])

        """ Get bounds """
        Y_MIN_TRAIN, Y_MAX_TRAIN, Y_MIN_TEST, Y_MAX_TEST = min(y_train), max(y_train), min(y_test),max(y_test)
        #print(f'# Bounds # \n Y_MIN_TRAIN:{Y_MIN_TRAIN}, Y_MAX_TRAIN:{Y_MAX_TRAIN}, Y_MIN_TEST:{Y_MIN_TEST}, Y_MAX_TEST:{Y_MAX_TEST}')

        """ Normalize data """
        if NORMALIZE_LOCAL:
            y_train = normalize_local(y_train, Y_MIN_TRAIN, Y_MAX_TRAIN)
            y_test = normalize_local(y_test, Y_MIN_TEST, Y_MAX_TEST)
        else:
            y_train = normalize(y_train)
            y_test = normalize(y_test)

        """ Prepare sequence generators """
        train_series = y_train.reshape((len(y_train), NUM_FEATURES)) # e.g.,: [E[0], E[1], E[2], ...]
        test_series  = y_test.reshape((len(y_test), NUM_FEATURES))
        train_generator = TimeseriesGenerator(train_series, train_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)
        test_generator = TimeseriesGenerator(test_series, test_series, length = LOOK_BACK, sampling_rate = 1, stride = 1, batch_size = BATCH_SIZE)

        """ Train online """
        model.fit(train_generator,epochs=EPOCHS_ONLINE_TRAINING, verbose=0, callbacks=[model_checkpoint_callback, loss_callback])

        learned = True

        # print(f'Trained {EPOCHS_ONLINE_TRAINING} more epochs online\n')

        data = deque(data, maxlen=DATASET_LEN) # put data back into deque to append new values

    """
    (D1) Prediction part
    """
    if learned and predict and len(Ed_past) == LOOK_BACK: # if already learned and Ed_past deque is full, start predicting

        """ Normalize Ed samples to [0,1] """
        Ed_past_normalized = deque(list(map(normalize, Ed_past)),maxlen=LOOK_BACK) # Normalize between 0 and 1

        """ Convert to numpy and reshape Ed_past normalized to be fed to the model """
        y_past = np.array([Ed_past_normalized]).reshape(1,LOOK_BACK,1) # Save the past values normalized in y_past

        Ed_past_temp = Ed_past_normalized.copy() # Saving normalized values in Ed_past_temp e.g., [Ed[n - LOOK_BACK + 1],..., Ed[n]]

        """ Predict N times """
        for pred_i in range(N):
            predictions.append((model(y_past, training = False)[0][0]).numpy()) # append Ed'[n+1] to predictions list
            Ed_past_temp.append(predictions[-1]) # Save last prediction into Ed_past_temp, e.g.,: [Ed[n - LOOK_BACK + 1 + i], ... Ed[n + i]]
            y_past = np.array([Ed_past_temp]).reshape(1,LOOK_BACK,1) # y_past is Ed_past_temp transformed into a numpy array
            with tb_writer.as_default(step=n):
                tf.summary.scalar(name=f'pred n + {pred_i}', data=predictions[-1] * (Y_MAX-Y_MIN) + Y_MIN, step=n+pred_i, description=f"Ed predicted for n + {pred_i}")

        """ Unormalize predictions to log and show """
        unnormalized_predictions = list(map(unnormalize, predictions)) # unnormalize
        logging.debug(f"predictions: {unnormalized_predictions}") # log
        if PRINT_PREDICTIONS_INFO: print(f"predictions: {unnormalized_predictions}") # show

        """(D2) Inflection evaluation part"""
        """ New prediction aray containing unnormalized predictions """
        preds = np.array([unnormalize(pred) for pred in predictions])

        """ Define CLOSE_TO_INFLECTION distance """
        CLOSE_TO_INFLECTION_THRESHOLD = 40 # TODO this should be = (E_charge - energy_in_which_error_starts_to_be_significant)

        """ Evaluate if any prediction is inside INFLECTION REGION """
        if any(preds - E_lower_d < CLOSE_TO_INFLECTION_THRESHOLD): # Predict if Ed will <= E_lower_d
            #needs_charging_d = True REMOVE
            wncd = True # will need charge dumb
            dwrel = True # dumb will reach elower

            if wncd and not dumb_instant_energy:
                #Ed_value_down.append(Ed)
                Ed_value_down = Ed
                dumb_instant_energy = True

            logging.debug(f'Leaving the charging station because some {unnormalized_predictions} <= {E_lower_d}') # log
            if PRINT_PREDICTIONS_INFO:
                print(f'Leaving the charging station because some {unnormalized_predictions} <= {E_lower_d}') # log

        """ Predict if Ed will >= E_charge """
        # if any(E_charge - preds < CLOSE_TO_INFLECTION_THRESHOLD):
        #     logging.debug(f'Going to the charging station because some {unnormalized_predictions} >= {E_charge}') # log
        #     if PRINT_PREDICTIONS_INFO: print(f'Going to the charging station because some {unnormalized_predictions} >= {E_charge}') # show
        #     #needs_charging_d = False REMOVE
        #     wncd = False # will not need charge dumb
        #     dwrel = False # dumb will not reach elower
        #     vi_computed = False
        #     # vcs_computed = False

    """(E) Cases part"""

    """ Intelligent behaviour """

    if E < E_min:
        print(f'n: {n}. it went below 0')
        # Plot energies
        plt.title(str(E_lower_i))
        plt.plot([x for x in range(len(ed_plot))], ed_plot)
        plt.plot([x for x in range(len(e_plot))], e_plot)
        plt.plot([x for x in range(len(e_lower_plot_i))], e_lower_plot_i)
        plt.plot([x for x in range(len(e_lower_plot_d))], e_lower_plot_d)
        plt.plot([x for x in range(len(e_min_plot))], e_min_plot)
        plt.plot([x for x in range(len(e_charge_plot))], e_charge_plot)
        plt.plot([x for x in range(len(e_min_plot))], e_min_plot)
        plt.legend(["Agent 1", "Agent 2", "e_lower_int", "e_lower_dumb", "e_min", "e_charge", "e_min + e_w"], loc ="lower right")
        plt.grid()
        plt.savefig('Learning_case_noise' + ts + '.png')
        exit()

    """Instant cs"""
    if needs_charging_i and not needs_charging_d and not vcs_computed:
        v_cs = compute_v_cs(x1i, y1i, x_cs, y_cs, B_d, Kd, E, E_min)
        vi_1 = v_cs
        vcs_computed = True 
        vec_to_cs = np.array([x_cs - x1i, y_cs - y1i])
        dist_to_cs = np.linalg.norm(vec_to_cs)
        v_vec = v_cs * ( vec_to_cs /  dist_to_cs)
        x2i, y2i = v_vec[0], v_vec[1]
        print(f'vcs computed: {v_cs}')
        U = np.array([0.0, 0.0])

    # if vcs_computed:
    #     print(f'vcs: {round(v_cs,3)}')


    """
    Instant v_red
    """
    if needs_charging_i and wncd:
        T_CD = (E_charge - E_lower_d) / B_c_dumb
        offset = 0
        v_red = math.sqrt(((E - E_min - offset - T_CD * Kd )) / (T_CD * B_d ))
        vi_1 = v_red
        vi_computed = True
        # print(f'vred_sqr:{vi_1}')
        # vcs_computed = False

    p_c_1 = math.sqrt((x1i - x_cs)**2 + (y1i - y_cs)**2)
    if needs_charging_i and p_c_1 <= d_charge and not wncd:
        x2i, y2i = 0.0, 0.0  # Stop motion
        vi_1 = math.sqrt(x2i**2 + y2i**2)
    
    """Change of controllers for intelligent"""
    if needs_charging_i and wncd:
        U = U_track_i
    elif needs_charging_i:
        U = np.array([0.0, 0.0])
    else:
        U = U_track_i

    if p_c > d_charge:
        B = -B_d*(vi_1)**2 - Kd
    else:
        B = B_c

    """Compute distance covered"""
    if abs(xdes1 - x1i) <= threshold_distance and abs(ydes1 - y1i) <= threshold_distance:
        if start_computing_distance == False:
            start_computing_distance = True
            alpha_prev = math.atan2(y1i - Yc1, x1i - Xc1)
        else:
            arc_length += r * abs( abs( (math.atan2(y1i - Yc1, x1i - Xc1) ) ) - abs(alpha_prev) )
            alpha_prev = math.atan2(y1i - Yc1, x1i - Xc1)
            t_dis += dt
    else:
        start_computing_distance = False

    """
    Intelligent dynamics
    """
    #if learned:
    x1i = x1i + dt * (x2i + mean) + std_dev * np.random.randn()
    x2i = x2i + dt * (U[0] + mean) + std_dev * np.random.randn()
    y1i = y1i + dt * (y2i + mean) + std_dev * np.random.randn()
    y2i = y2i + dt * (U[1] + mean) + std_dev * np.random.randn()
    E = E + dt * B

    """
    Dumb dynamics
    """
    x1d = x1d + dt * x2d
    x2d = x2d + dt * U_dumb[0]
    y1d = y1d + dt * y2d
    y2d = y2d + dt * U_dumb[1]
    Ed = Ed + dt * battery_dumb
    #print(f'n: {n} | E: {E}| Ed: {Ed} | vi: {vi} | E_rem: {E - (E_min + e_worst)}')

    if E < E_min:
        print(f'Eenergy went below E_min')
        exit()


    """ Save data for animation """
    if args.animation:
        x_d.append(x1d)
        y_d.append(y1d)
        x_i.append(x1i)
        y_i.append(y1i)
        nci_flag.append(needs_charging_i)
        ncd_flag.append(needs_charging_d)
        dwnc_flag.append(wncd)
        dwrel_flag.append(dwrel)
        e_i.append(E)
        e_d.append(Ed)
        v_i.append(vi_1)
        arc_length_plot_1.append(arc_length)
    
    """ Save in .csv """
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write new row in the csv
        writer.writerow([
            timestep, Ed, needs_charging_d, wncd,
            E, needs_charging_i, vi_1, arc_length, x1i, y1i, x1d, y1d
        ])

    """ Show energies if necessary """
    if PLOT_SIMULATION:
        ed_plot.append(Ed)
        e_plot.append(E)
        vi_plot.append(vi_1)

print(f'Idle time {len(t) * dt - t_dis}')

"""
PLOT
"""
if PLOT_SIMULATION:
    # Plot energies
    plt.title(str(E_lower_i))
    plt.plot([x for x in range(len(ed_plot))], ed_plot)
    plt.plot([x for x in range(len(e_plot))], e_plot)
    plt.plot([x for x in range(len(e_lower_plot_i))], e_lower_plot_i, linestyle = '--', linewidth = 2, color = 'm')
    plt.plot([x for x in range(len(e_lower_plot_d))], e_lower_plot_d, linestyle = '--', linewidth = 2, color = 'c')
    plt.plot([x for x in range(len(e_min_plot))], e_min_plot, color = 'r')
    plt.plot([x for x in range(len(e_charge_plot))], e_charge_plot, color = 'g')
    # plt.plot([x for x in range(len(e_min_plot))], e_min_plot + e_worst)
    plt.legend(["Agent 1", "Agent 2", "e_lower_int", "e_lower_dumb", "e_min", "e_charge"], loc ="lower right")
    plt.grid()
    plt.savefig('Learning_case_' + ts + '.png')
    if PLOT_FOR_A_BIT:
        plt.show(block=False)
        plt.pause(TIME_SHOW_DATA)
    else:
        plt.show()
    plt.close()

    # # Plot velocity
    # plt.plot([x for x in range(len(vi_plot))], vi_plot)
    # plt.grid()
    # if PLOT_FOR_A_BIT:
    #     plt.show(block=False)
    #     plt.pause(TIME_SHOW_DATA)
    # else:
    #     plt.show()
    # plt.close()

print(f'Distance covered: {str(arc_length)}') # show distance covered

"""
(F) Animation part
"""
import matplotlib.animation as animation
if args.animation:
    x = [x for x in range(len(t)-1)]

    """ Plot animation """
    fig, ax = plt.subplots()

    plt.plot([],[])

    dumb_robot, = ax.plot([], [], 'bo', markersize=4)
    intelligent_robot, = ax.plot([], [], 'ro', markersize=4)
    circle1_line, = ax.plot(xdes_i, ydes_i, color='red', linestyle = '--')
    circle2_line, = ax.plot(xdes_d, ydes_d, color='blue', linestyle = '--')
    circle3 = plt.Circle((x_cs, y_cs), radius=0.1, color='g', label = 'CS', fill=False) # charging station

    """
    @TODO add flags and velocity and energy as text
    """

    time_text = ax.text(0.05, 0.90, '', transform=ax.transAxes)
    e_i_text = ax.text(0.65, 0.65, '', transform=ax.transAxes)
    nci_text = ax.text(0.65, 0.55, '', transform=ax.transAxes)
    #v_i_text = ax.text(0.8, 0.8, '', transform=ax.transAxes)
    e_d_text = ax.text(0.65, 0.45, '', transform=ax.transAxes)
    ncd_text = ax.text(0.65, 0.35, '', transform=ax.transAxes)
    arc_text_1 = ax.text(0.05, 0.80, '', transform=ax.transAxes)
    dwnc_text = ax.text(0.6, 0.4, '', transform=ax.transAxes)
    #dwrel_text = ax.text(0.6, 0.3, '', transform=ax.transAxes)


    def init():
        ax.set_xlim(0, 18)
        ax.set_ylim(5, 18)
        # ax.add_artist(circle1)
        # ax.add_artist(circle2)
        ax.add_artist(circle3)
        handles = [dumb_robot, intelligent_robot, circle1_line, circle2_line, circle3]
        labels = ['$IA_1$', '$DA_1$', 'Tracking Path', 'Tracking Path', 'CS']
        ax.legend(handles, labels, loc="upper right", ncol=3, fontsize=8)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        return dumb_robot, time_text, intelligent_robot, circle1_line, circle2_line, circle3, arc_text_1, dwnc_text

    def animate(i):
        dumb_robot.set_data(x_d[i], y_d[i])
        intelligent_robot.set_data(x_i[i], y_i[i])
        time_text.set_text('Time = {:.2f}'.format(i * dt))

        # Display needs charging values as text
        e_i_text.set_text('$E_{{DA_1}}$: {:.3f}'.format(e_i[i]))
        nci_text.set_text('$n_{{ch,DA_1}}$: {}'.format(nci_flag[i]))
        #v_i_text.set_text('vi: {:.3f}'.format(v_i[i]))
        e_d_text.set_text('$E_{{IA_1}}$: {:.3f}'.format(e_d[i]))
        ncd_text.set_text('$n_{{ch,IA_1}}$: {}'.format(ncd_flag[i]))
        arc_text_1.set_text('Arc Length = {:.2f}'.format(arc_length_plot_1[i]))
        dwnc_text.set_text('dwnc: {}'.format(dwnc_flag[i]))
        #dwrel_text.set_text('dwrel: {}'.format(dwrel_flag[i]))

        return dumb_robot, time_text, intelligent_robot, circle1_line, circle2_line, circle3, e_i_text, nci_text, e_d_text, ncd_text, arc_text_1, dwnc_text

    ani = animation.FuncAnimation(fig, animate, frames=len(x),
                                init_func=init, interval=10, blit=True, repeat=True)
    plt.show()

"""
Write to file the distance covered
"""
# print('FINAL: ', proof)
# Save arc length and model
num_runs = 0
if args.out_file:
    with open(args.out_file, "r") as file:
        num_runs = file.readlines()
    with open(args.out_file, "a") as file:
        if num_runs == 0:
            if E <= E_min:
                file.write(str(E_lower_i) + " | " + "Energy went below E_min")
            else:
                file.write(str(E_lower_i) + " | " + str(arc_length))
        else:
            if E <= E_min:
                file.write('\n' + str(E_lower_i) + " | " + "Energy went below E_min")
            else:
                file.write('\n' + str(E_lower_i) + " | " + str(arc_length))
