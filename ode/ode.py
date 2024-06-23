"""Give three methods to resolve ODE's.

This module allows the user to resolve ODE's with three methods.

Examples:
    >>> import ode
    >>> import numpy as np
    >>> def f(x,t):
    ...     return x +  t
    >>> ode.RK4(f, 10, 0.0, 0.0, 10.0)
    array([0.00000000e+00, 7.08333333e-01, 4.52488426e+00, 1.67595245e+01,
           5.17931566e+01, 1.48574058e+02, 4.12587149e+02, 1.12952075e+03,
           3.07311407e+03, 8.33891079e+03])

The module contains the following functions:
- 'RK4(f, N, xi, ti, tf)' - Returns an array with the results of the ODE.
- 'RK2(f, N, xi, ti, tf)' - Returns an array with the results of the ODE.
- 'Euler(f, N, xi, ti, tf)' - Returns an array with the results of the ODE.
"""


import numpy as np

def RK4(f,N, xi,ti,tf):
    """Compute and  Return an array with N values of x that are the results of an ODE within the Runge Kutta 4 method.

    Examples:
        >>> import numpy as np
        >>> def f(x,t):
        ...     return x +  t
        >>> RK4(f, 10, 0.0, 0.0, 10.0)
        array([0.00000000e+00, 7.08333333e-01, 4.52488426e+00, 1.67595245e+01,
               5.17931566e+01, 1.48574058e+02, 4.12587149e+02, 1.12952075e+03,
               3.07311407e+03, 8.33891079e+03])

    Args:
        f (function): function that returns the ordinary diferential equation (ODE)
        N (int): Steps number
        xi (float): initial condition for space (x)
        ti (float): initial condtion for time (t)
        tf (float): final contion for time (t)

    Returns:
        ndarray: Returns the array of space results x of the ODE
    """
    h = (tf-ti)  /N
    t = np.linspace(ti, tf, N)
    x = np.zeros(len(t))
    x[0] = xi
    for i in range(len(t)-1):
        k1 = h * f(x[i],t[i])
        k2 = h * f(x[i] + k1/2, t[i] + h/2)
        k3 = h * f(x[i] + k2/2, t[i] + h/2)
        k4 = h * f(x[i]+k3,t[i]+h)
        x[i+1] = x[i] + (1/6) * (k1+2*k2+2*k3+k4)
    return x

def RK2(f,N, xi,ti,tf):
    """Compute and  Return an array with N values of x that are the results of an ODE within the Runge Kutta 2 method.

    Examples:
        >>> import numpy as np
        >>> def f(x,t):
        ...     return x + t
        >>> RK2(f, 10, 0.0, 0.0, 10.0)
        array([0.00000000e+00, 5.00000000e-01, 3.41666667e+00, 1.23750000e+01,
               3.64375000e+01, 9.82604167e+01, 2.54484375e+02, 6.46710937e+02,
               1.62894401e+03, 4.08619336e+03])

    Args:
        f (function): function that returns the ordinary diferential equation (ODE)
        N (int): Steps number
        xi (float): initial condition for space (x)
        ti (float): initial condtion for time (t)
        tf (float): final contion for time (t)

    Returns:
        ndarray: Returns the array of space results x of the ODE
    """
    h = (tf-ti)/N
    t = np.linspace(ti, tf, N)
    x = np.zeros(len(t))
    x[0] = xi
    for i in range(len(t)-1):
        k1 = h * f(x[i],t[i])
        k2 = h * f(x[i] + k1/2, t[i] + h/2)
        x[i+1] = x[i] + k2
    return x


def Euler(f, N, xi, ti, tf):
    """Compute and  RetEuler(f, 10, 0, 0, 10)urn an array with N values of x that are the results of an ODE within the Euler method.

    Examples:
        >>> import numpy as np
        >>> def f(x,t):
        ...     return x + t
        >>> Euler(f, 10, 0.0, 0.0, 10.0)
        array([  0.        ,   0.        ,   1.11111111,   4.44444444,
                12.22222222,  28.88888889,  63.33333333, 133.33333333,
               274.44444444, 557.77777778])

    Args:
        f (function): function that returns the ordinary diferential equation (ODE)
        N (int): Steps number
        xi (float): initial condition for space (x)
        ti (float): initial condtion for time (t)
        tf (float): final contion for time (t)

    Returns:
        ndarray: Returns the array of space results x of the ODE
    """
    h = (tf - ti) / N
    t = np.linspace(ti, tf, N)
    x = np.zeros(len(t))
    x[0] = xi
    for i in range(len(t)-1):
        x[i+1] = x[i] + h * f(x[i], t[i])
    return x
