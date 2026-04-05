from solve import solve_newton, compute_area, area_derivative
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def get_shape_points(exponent, N=10_000):
    x = np.linspace(0, 1, N)
    exponent = max(exponent, 1e-8)
    y = (1 - x**exponent) ** (1/exponent)
    return np.hstack((-x[::-1], x, x[::-1], -x)), np.hstack((y[::-1], y, -y[::-1], -y))


def plot():
    def plot_super_ellipse(area):
        target_function = lambda x: compute_area((x)) - area
        # The initial guess is important. Could use bisection method, but 
        # 0.5 seems to work fine.
        exponent = solve_newton(target_function, area_derivative, 0.5) 
        X, Y = get_shape_points(exponent)
        line.set_xdata(X)
        line.set_ydata(Y)
        fig.canvas.draw_idle()

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    slider_ax = fig.add_axes([0.2, 0.025, 0.6, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="Area",
        valmin=0,
        valmax=4,
        valinit=0,
    )

    x_init, y_init = get_shape_points(slider.valinit)
    line = ax.plot(x_init, y_init)[0]

    slider.on_changed(plot_super_ellipse)
    fig.canvas.manager.set_window_title("Area-based superellipse interpolation")
    plt.show()


if __name__ == "__main__":
    plot()
