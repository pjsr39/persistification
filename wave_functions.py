import numpy as np

def square_wave(x, num_harmonics):
    sum_ = 0
    for i in range(num_harmonics):
        sum_ += np.sin((2*i+1) * x) / (2*i+1)
    return sum_
def triangle_wave(x, num_harmonics):
    sum_ = 0
    for i in range(num_harmonics):
        sum_ += (-1)**(i+1)*np.sin((2*i+1) * x) / (2*i+1)**2
    return sum_
def saw_wave(x, num_harmonics):
    sum_ = 0
    for i in range(num_harmonics):
        sum_ += np.sin((i+1) * x)/(i+1)
    return sum_