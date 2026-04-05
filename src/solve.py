import math
import scipy.special


def compute_area(exponent: float) -> float:
    """
    Computes the area of a superellipse.

    :param exponent: The exponent for the superellipse
    :type exponent: float

    :returns: The area of the superellispe
    :rtype: float

    """
    if exponent <= 0:
        return 0
    
    return 4 * (math.gamma(1 + 1 / exponent))**2 / (math.gamma(1 + 2/exponent))


def area_derivative(x: float) -> float:
    """
    Computes the derivative of the superellipse area, as a function of the exponent.

    :param x: Which value of the exponent to compute the area derivative at
    :type x: float

    :returns: The derivative
    :rtype: float

    """
    numerator = math.gamma(1 + 1/x)**2
    numerator_prime = -2 * math.gamma(1 + 1/x) * math.gamma(1 + 1/x) * scipy.special.digamma(1 + 1/x) / x**2
    denominator =  math.gamma(1 + 2/x)
    denominator_prime = -2 * math.gamma(1 + 2/x) * scipy.special.digamma(1 + 2/x) / x**2

    return 4 * (numerator_prime*denominator - numerator*denominator_prime) / (denominator**2)


def transform(x: float, low: float = 0, high: float = 1) -> float:
    """
    Transforms x from the domain [low, high] -> [0, inf]

    :param x: The original value
    :type x: float
    :param low: The lower endpoint of the new range
    :type low: float
    :param high: The higher endpoint of the new range
    :type high: float

    :raises ValueError: If `x` is outside the original range

    :returns: The area of the superellispe
    :rtype: float

    """
    if x == 1:
        return math.inf
    
    if x < low or x > high:
        raise ValueError(f"Argument 'x' outside of range [{low}, {high}]")
    
    return 1 / (1-x) - 1 / (high - low)


def solve_bisect(func, low: float = 0, high: float = 1, tolerance: float = 1e-8, max_attempts: int = 100) -> float:
    """
    Uses the bisection method to find the roots of function `func`.
    Will return the first root found.
    
    :param func: The function to find the roots of
    :type func: function
    :param low: Lower endpoint of searching range
    :type low: float
    :param high: Higher endpoint of searching range
    :type high: float
    :param tolerance: Error tolerance
    :type tolerance: float
    :param max_attempts: The maximum number of bisections to perform
    :type max_attempts: int

    :raises ValueError: If the method does not converge

    :returns: The first root found
    :rtype: float
    """
    midpoint = (high + low) / 2
    for _ in range(max_attempts):
        y = func(midpoint)
        if abs(y) <= tolerance / 2:
            return midpoint
        
        elif y >  0:
            high = midpoint
        
        else:
            low = midpoint

        midpoint = (high + low) / 2

    raise ValueError("Bisection method did not converge")


def solve_newton(func, derivative_func, initial_guess: float, tolerance: float = 1e-13, max_attempts: int = 100) -> float:
    """
    Uses the Netwon-Raphson method to find the roots of function `func`.
    Will return the first root found.
    
    :param func: The function to find the roots of
    :type func: function
    :param func: Function representing the derivative of `func`
    :type func: function
    :param initial_guess: The initial guess
    :type initial_guess: float
    :param tolerance: Error tolerance
    :type tolerance: float
    :param max_attempts: The maximum number of bisections to perform
    :type max_attempts: int

    :raises ValueError: If the method does not converge

    :returns: The first root found
    :rtype: float
    """
    x_prev = initial_guess
    for _ in range(max_attempts):
        y = func(x_prev)
        y_prime = derivative_func(x_prev)
        if abs(y) < tolerance / 2:
            return x_prev
        
        x = x_prev - y / y_prime
        x_prev = x

    raise ValueError("Newton-Raphson method did not converge")
