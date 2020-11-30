import numpy as np

''' setup benchmark functions '''
def target_function(function_name='ackley', dim=10):

    # Ackley function
    if function_name == 'ackley':
        fun = lambda x: -20 * np.exp(-.2 * np.sqrt(np.sum(x**2) / dim))\
            - np.exp(np.sum(np.cos(2*np.pi*x)) / dim) + 20 + np.exp(1)
        x_dom = [[-32.768, 32.768]] * dim
        x_min = [0] * dim

    # Griewank function
    elif function_name == 'griewank':
        fun = lambda x: np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(dim)+1))) + 1
        x_dom = [[-600, 600]] * dim
        x_min = [0] * dim

    # Levy function
    elif function_name == 'levy':
        w = lambda x: (x + 3) / 4
        fun = lambda x: np.sin(np.pi*w(x)[0])**2 \
            + np.sum((w(x)[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w(x)[:-1] + 1)**2)) \
            + (w(x)[-1] - 1)**2 * (1 + np.sin(2*np.pi*w(x)[-1])**2)
        x_dom = [[-10, 10]] * dim
        x_min = [1] * dim

    # Rastrigin function
    elif function_name == 'rastrigin':
        fun = lambda x: 10*dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim

    # sphere function
    elif function_name == 'sphere':
        fun = lambda x: np.sum(x**2)
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim

    # Branin function
    elif function_name == 'branin':
        fun = lambda x: 10 + 10*(1 - 1/(8*np.pi)) * np.cos(x[0]) \
            + (x[1] - 5.1/(4*np.pi**2) * x[0]**2 + 5/np.pi * x[0] - 6)**2
        x_dom = [[-5, 10], [0, 15]]
        x_min = [np.pi, 2.275]

    # cross-in-tray function
    elif function_name == 'cross-in-tray':
        fun = lambda x: -.0001*(np.abs(np.sin(x[0]) * np.sin(x[1]) \
            * np.exp(np.abs(100 - np.sqrt(np.sum(x**2))/np.pi))) + 1)**.1
        x_dom = [[-10, 10], [-10, 10]]
        x_min = [1.3491, 1.3491]

    # dropwave function
    elif function_name == 'dropwave':
        fun = lambda x: 1 - (1 + np.cos(12*np.sqrt(np.sum(x**2)))) / (.5*np.sum(x**2) + 2)
        x_dom = [[-5.12, 5.12], [-5.12, 5.12]]
        x_min = [0, 0]

    # other
    else:
        raise SystemExit('function {:s} is not defined...'.format(function_name))

    return fun, np.array(x_min), np.array(x_dom).T


# randomly sample initial guess
def initial_guess(x_dom):
    dim = x_dom.shape[-1]
    x0 = (x_dom[1] - x_dom[0]) * np.random.random(dim) + x_dom[0]
    return x0


