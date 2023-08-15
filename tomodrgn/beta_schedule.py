import numpy as np

def get_beta_schedule(schedule, n_iterations = None):
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
            raise RuntimeError('Wrong beta schedule. Schedule={}'
                               .format(schedule))

class ConstantSchedule:
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return self.value


class LinearSchedule:
    def __init__(self, start_y, end_y, start_x, end_x):
        self.min_y = min(start_y, end_y)
        self.max_y = max(start_y, end_y)
        self.start_x = start_x
        self.start_y = start_y
        self.coef = (end_y - start_y) / (end_x - start_x)

    def __call__(self, x):
        return np.clip((x - self.start_x) * self.coef + self.start_y,
                       self.min_y, self.max_y).item(0)

class CyclicalSchedule:
    # implementation of https://doi.org/10.48550/arXiv.1903.10145, https://github.com/haofuml/cyclical_annealing
    def __init__(self,
                 n_iterations,
                 start_value = 0,
                 end_value = 1,
                 n_cycles = 4,
                 fraction_cycle_increase = 0.5,
                 increase_function = 'linear'):

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


