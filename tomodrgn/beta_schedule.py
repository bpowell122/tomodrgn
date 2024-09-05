"""
Classes and functions to facilitate sampling beta values per minibatch when training a VAE.
"""
from typing import Literal, Any

import numpy as np


class ConstantSchedule:
    """
    Return a constant value for beta, independent of iteration.
    """
    def __init__(self,
                 value: float):
        """
        Instantiate a ConstantSchedule.
        :param value: the value of beta to always return
        """
        self.value = value

    def __call__(self, iteration: Any) -> float:
        return self.value


class LinearSchedule:
    """
    Return a linearly scaling beta value as a function of iteration.
    """
    def __init__(self,
                 start_y: float,
                 end_y: float,
                 start_x: float,
                 end_x: float):
        """
        Instantiate a LinearSchedule.

        :param start_y: the value of beta to start at iteration 0
        :param end_y: the value of beta to end at end_x iterations. Calling after end_x iterations will return a value clipped by min_y and max_y.
        :param start_x: the iteration associated with start_y for defining the linear fit
        :param end_x: the iteration associated with end_y for defining the linear fit
        """
        self.min_y = min(start_y, end_y)
        self.max_y = max(start_y, end_y)
        self.start_x = start_x
        self.start_y = start_y
        self.coef = (end_y - start_y) / (end_x - start_x)

    def __call__(self, iteration: int) -> float:
        return np.clip((iteration - self.start_x) * self.coef + self.start_y,
                       self.min_y,
                       self.max_y).item(0)


class CyclicalSchedule:
    """
    Return a cyclically scaling beta value as a function of iteration.
    Implementation of https://doi.org/10.48550/arXiv.1903.10145, https://github.com/haofuml/cyclical_annealing
    """
    def __init__(self,
                 n_iterations: int,
                 start_value: int = 0,
                 end_value: int = 1,
                 n_cycles: int = 4,
                 fraction_cycle_increase: float = 0.5,
                 increase_function: Literal['linear'] = 'linear'):
        """
        Instantiate a CyclicalSchedule.

        :param n_iterations: the total number of iterations over which the CyclicalSchedule is defined.
        :param start_value: the initial value of beta to start at the beginning of each cycle
        :param end_value: the final value of beta at the end of each cycle
        :param n_cycles: the number of cycles to perform
        :param fraction_cycle_increase: the fraction of each cycle's iterations to spend changing beta, versus keeping beta constant at end_value
        :param increase_function: the type of function as which to define the change in beta
        """

        betas = end_value * np.ones(n_iterations, dtype=np.float32)
        period = int(n_iterations / n_cycles)
        step = (end_value - start_value) / (period * fraction_cycle_increase)
        for i in range(n_cycles):
            for j in range(round(period * fraction_cycle_increase)):
                if increase_function == 'linear':
                    betas[i * period + j] = j * step
                else:
                    raise NotImplementedError

        self.betas = betas

    def __call__(self, iteration):
        return self.betas[iteration]


def get_beta_schedule(schedule: float | Literal['a', 'b', 'c', 'd', 'e'],
                      n_iterations: int | None = None) -> ConstantSchedule | LinearSchedule | CyclicalSchedule:
    """
    Return one of several types of beta schedules: Constant, Linear, or Cyclical.
    Each schedule, when called, returns the beta value from a constant function, linear function, or cyclical function.

    :param schedule: if type float, the value of the beta to always return via a Constant Schedule; otherwise, return a pre-configured Linear or Cyclical schedule based on the supplied letter.
    :param n_iterations: If using a CyclicalSchedule, define the total number of batches during training (used to set the period of cycles together with n_cycles).
    :return: a ConstantSchedule, LinearSchedule, or CyclicalSchedule object which may be called to return the next beta value according to the type of schedule.
    """
    try:
        schedule = float(schedule)
        return ConstantSchedule(schedule)
    except ValueError:
        if schedule == 'a':
            return LinearSchedule(0.001, 15, 0, 1000000)
        elif schedule == 'b':
            return LinearSchedule(5, 15, 200000, 800000)
        elif schedule == 'c':
            return LinearSchedule(5, 18, 200000, 800000)
        elif schedule == 'd':
            return LinearSchedule(5, 18, 1000000, 5000000)
        elif schedule == 'e':
            return CyclicalSchedule(n_iterations)
        else:
            raise RuntimeError(f'Unrecognized beta schedule: {schedule=}')
