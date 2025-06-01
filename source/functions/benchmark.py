import numpy as np


def sphere(x):
    return np.sum(np.square(x))


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ellipsoid(x):
    return np.sum([(i + 1) * xi**2 for i, xi in enumerate(x)])


def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)


def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def griewank(x):
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1


def zakharov(x):
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4


def michalewicz(x, m=10):
    return -np.sum(
        np.sin(x) * (np.sin((np.arange(1, len(x) + 1) * x**2) / np.pi)) ** (2 * m)
    )


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


benchmark_functions = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ellipsoid": ellipsoid,
    "ackley": ackley,
    "schwefel": schwefel,
    "griewank": griewank,
    "zakharov": zakharov,
    "michalewicz": michalewicz,
    "booth": booth,
}
