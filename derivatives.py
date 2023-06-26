import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

p2_delta = 0.0001

x = np.arange(0,5, 0.001)
y = f(x)

#print(len(x), x)
#print(y)

def derivative(f, x1):
    x2 = x1 + p2_delta
    y1 = f(x1)
    y2 = f(x2)
    result = ((y2-y1) / (x2-x1))
    return result

def draw_tangent(x):
    approximate_derivative = derivative(f, x)
    print('Approximate derivative for f(x)',
f'where x = {x} is {approximate_derivative}')
    b = f(x) - approximate_derivative*x

    def tangent_line(x):
        return approximate_derivative*x + b

    to_plot = [x-0.9, x, x+0.9]
    plt.plot(to_plot, [tangent_line(x) for x in to_plot])
plt.plot(x, y)
[draw_tangent(x1) for x1 in np.arange(1,5,1)]
plt.show()