"""
File: sphere.py
By Peter Caven, peter@sparseinference.com

Description:

An example of gradient-free and gradient assisted optimization
of a simple test function.

See: https://en.wikipedia.org/wiki/Test_functions_for_optimization


"""

import torch
from brkga import BRKGA


def Sphere(x):
    return x.dot(x)


def box(fun, lower, upper):
    width = upper - lower
    def inner(x):
        return fun((x * width) + lower)
    return inner


def Optimize(keyShape, elites=2, mutants=2):
    """
    Return the Random-Key that minimizes the Sphere function
    in the interval (-50,+50) over 'keyShape[1]' dimensions.
    """
    trial = 0
    lower = -50.0
    upper =  50.0
    f = box(Sphere, lower, upper)
    pop = BRKGA(keyShape, elites=elites, mutants=mutants)
    results = pop(f)
    bestResult,best = pop.orderBy(results)
    print(f"[{trial:6d}] {bestResult:.8f}")
    print(best.data)
    try:
        while bestResult > 1.0e-7:
            trial += 1
            pop.evolve()
            results = pop(f)
            bestResult,best = pop.orderBy(results)
            if trial % 100 == 0:
                print(f"[{trial:6d}] {bestResult:.8f}")
                print(best.data)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[{trial:6d}] {bestResult:.8f}")
        print(best.data)
    return bestResult,best


def OptimizeSGD(keyShape, elites=2, mutants=2):
    """
    Return the Random-Key that minimizes the Sphere function
    in the interval (-50,+50) over 'keyShape[1]' dimensions.
    This function uses gradient information to improve the evolved solutions.
    """
    trial = 0
    lr = 0.001
    lower = -50.0
    upper =  50.0
    f = box(Sphere, lower, upper)
    pop = BRKGA(keyShape, elites=elites, mutants=mutants)
    results = pop(f)
    bestResult,best = pop.orderBy(results)
    print(f"[{trial:6d}] {bestResult:.8f}")
    print(best.data)
    try:
        while bestResult > 1.0e-7:
            trial += 1
            pop.evolve()
            results = pop(f)
            results.mean().backward()
            pop.sgd(lr)
            results = pop(f)
            bestResult,best = pop.orderBy(results)
            if trial % 100 == 0:
                print(f"[{trial:6d}] {bestResult:.8f}")
                print(best.data)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[{trial:6d}] {bestResult:.8f}")
        print(best.data)
    return bestResult,best

        


# Optimize((15,100), elites=2, mutants=2)
OptimizeSGD((15,100), elites=2, mutants=2)



"""
# OptimizeSGD((15,100), ef=0.2, mf=0.3)
python3 sphere.py

[     0] 70442.96093750
tensor([0.7439, 0.1317, 0.1853, 0.3816, 0.3252, 0.3035, 0.9699, 0.2796, 0.9739,
        0.1261, 0.7658, 0.0667, 0.4930, 0.9989, 0.0891, 0.5779, 0.0907, 0.9734,
        0.1215, 0.3560, 0.4309, 0.4962, 0.7528, 0.8664, 0.8815, 0.3663, 0.8897,
        0.5409, 0.2426, 0.4976, 0.1265, 0.3967, 0.2100, 0.2796, 0.0251, 0.4091,
        0.2170, 0.6902, 0.4105, 0.5198, 0.1993, 0.0472, 0.9579, 0.2749, 0.1516,
        0.5758, 0.5707, 0.8070, 0.7476, 0.4327, 0.3002, 0.6590, 0.4263, 0.3852,
        0.4772, 0.2518, 0.4091, 0.3740, 0.4593, 0.6154, 0.3423, 0.9706, 0.8662,
        0.5238, 0.7196, 0.0884, 0.5240, 0.6563, 0.5333, 0.4050, 0.3927, 0.2232,
        0.2837, 0.1841, 0.2157, 0.4025, 0.9163, 0.5749, 0.4428, 0.1850, 0.3898,
        0.5002, 0.1881, 0.7279, 0.8905, 0.3230, 0.6522, 0.0822, 0.7456, 0.5023,
        0.8895, 0.1389, 0.9558, 0.4227, 0.5898, 0.2685, 0.4348, 0.6708, 0.3736,
        0.2090])
[    13] 0.00000002
tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000])


"""