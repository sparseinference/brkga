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
    """
    1 dimension per population member.
    Keyshape: (population size, parameter keys)
    Parameter keys only.
    """
    width = upper - lower
    #----
    def bounds(keys):
        return lower,upper
    #----
    def decode(keys):
        return (keys * width) + lower
    #----
    def evaluate(keys):
        return fun(decode(keys))
    #----
    return bounds,decode,evaluate


def Optimize(keyShape, elites=2, mutants=2):
    """
    Minimize the Sphere function in the interval (-50,+50).
    """
    trial = 0
    bounds,decode,f = box(Sphere, -50, 50)
    pop = BRKGA(keyShape, elites=elites, mutants=mutants)
    results = pop.map(f)
    bestResult,best = pop.orderBy(results)
    print(f"[{trial:6d}] {bestResult:.8f}")
    print(decode(best.data))
    try:
        while bestResult > 1.0e-7:
            trial += 1
            pop.evolve()
            results = pop.map(f)
            bestResult,best = pop.orderBy(results)
            if trial % 100 == 0:
                print(f"[{trial:6d}] {bestResult:.8f}")
                print(decode(best.data))
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[{trial:6d}] {bestResult:.8f}")
        print(f"BRKGA keys=\n{best.data}")
        print(f"Sphere parameters=\n{decode(best.data)}")
    return bestResult,best


def OptimizeSGD(keyShape, elites=2, mutants=2):
    """
    Minimize the Sphere function in the interval (-50,+50).
    This function uses gradient information to improve the evolved solutions.
    """
    trial = 0
    bounds,decode,f = box(Sphere, -50, 50)
    pop = BRKGA(keyShape, elites=elites, mutants=mutants)
    bestResult,best = pop.orderBy(pop.map(f))
    print(f"[{trial:6d}] {bestResult:.8f}")
    print(decode(best.data))
    try:
        while bestResult > 1.0e-7:
            trial += 1
            pop.evolve()
            pop.map(f).mean().backward()
            pop.optimize()  # uses default SGD optimizer with lr=1.0e-3
            bestResult,best = pop.orderBy(pop.map(f))
            if trial % 100 == 0:
                print(f"[{trial:6d}] {bestResult:.8f}")
                print(decode(best.data))
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[{trial:6d}] {bestResult:.8f}")
        print(f"BRKGA keys=\n{best.data}")
        print(f"Sphere parameters=\n{decode(best.data)}")
    return bestResult,best

        

##===================================================

if __name__ == '__main__':
    # Optimize((15,100), elites=2, mutants=2)
    OptimizeSGD((15,100), elites=2, mutants=2)



"""


#Optimize((15,100), elites=2, mutants=2)
python3 sphere.py
...
[500626] 0.63734626
BRKGA keys=
tensor([0.4999, 0.5000, 0.5008, 0.5013, 0.4994, 0.5003, 0.5000, 0.4970, 0.4996,
        0.5015, 0.4998, 0.5001, 0.4982, 0.5012, 0.5012, 0.4999, 0.5005, 0.5000,
        0.5001, 0.5026, 0.4997, 0.5006, 0.5001, 0.4998, 0.5009, 0.5004, 0.5002,
        0.5024, 0.4999, 0.5000, 0.4998, 0.5004, 0.5005, 0.4996, 0.5004, 0.5003,
        0.5002, 0.5000, 0.5003, 0.5004, 0.5000, 0.4996, 0.4990, 0.4995, 0.5004,
        0.5002, 0.4972, 0.5000, 0.4990, 0.5002, 0.5005, 0.5011, 0.4997, 0.5002,
        0.4998, 0.5001, 0.5004, 0.5000, 0.4993, 0.5003, 0.5001, 0.4996, 0.5006,
        0.4996, 0.4998, 0.5010, 0.5001, 0.5002, 0.5012, 0.4999, 0.5006, 0.4991,
        0.5004, 0.4990, 0.4999, 0.5006, 0.5000, 0.4989, 0.4997, 0.5009, 0.4997,
        0.4997, 0.4983, 0.5000, 0.4994, 0.4997, 0.5000, 0.4999, 0.4995, 0.5013,
        0.4994, 0.4999, 0.5003, 0.5008, 0.5007, 0.4996, 0.5005, 0.5001, 0.5000,
        0.5002], dtype=torch.float64)
Sphere parameters=
tensor([-0.0059, -0.0044,  0.0847,  0.1293, -0.0625,  0.0256,  0.0012, -0.3001,
        -0.0371,  0.1548, -0.0162,  0.0090, -0.1762,  0.1151,  0.1203, -0.0096,
         0.0517, -0.0047,  0.0069,  0.2566, -0.0302,  0.0620,  0.0095, -0.0160,
         0.0890,  0.0369,  0.0240,  0.2361, -0.0148, -0.0035, -0.0154,  0.0374,
         0.0505, -0.0436,  0.0401,  0.0310,  0.0181, -0.0036,  0.0264,  0.0420,
         0.0033, -0.0368, -0.1048, -0.0460,  0.0394,  0.0212, -0.2777, -0.0046,
        -0.1039,  0.0189,  0.0451,  0.1073, -0.0343,  0.0216, -0.0202,  0.0103,
         0.0429,  0.0031, -0.0731,  0.0306,  0.0145, -0.0406,  0.0635, -0.0448,
        -0.0159,  0.0999,  0.0090,  0.0181,  0.1172, -0.0136,  0.0573, -0.0875,
         0.0370, -0.0954, -0.0097,  0.0640,  0.0046, -0.1115, -0.0297,  0.0902,
        -0.0312, -0.0332, -0.1704, -0.0025, -0.0636, -0.0282, -0.0012, -0.0082,
        -0.0502,  0.1277, -0.0561, -0.0094,  0.0255,  0.0790,  0.0698, -0.0448,
         0.0522,  0.0067, -0.0008,  0.0236], dtype=torch.float64)


#OptimizeSGD((15,100), elites=2, mutants=2)
python3 sphere.py
...
[    13] 0.00000002
BRKGA keys=
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
        0.5000], dtype=torch.float64)
Sphere parameters=
tensor([-1.4926e-07,  2.0242e-05, -1.0749e-05, -1.3968e-06, -4.5698e-06,
        -6.9870e-06, -1.1794e-05, -9.1310e-06,  5.0731e-06,  2.7197e-06,
         8.5585e-06,  9.9705e-06, -2.9009e-05, -2.1043e-05,  5.7087e-06,
        -1.5246e-05, -2.6238e-05,  3.1020e-06, -7.7293e-06,  2.0093e-07,
        -2.2200e-05,  9.3724e-06,  1.0739e-06,  2.0286e-06, -2.1888e-05,
        -1.1448e-05,  2.1088e-05, -2.9327e-06, -2.4273e-06,  8.6209e-06,
        -7.8139e-07, -1.5582e-05,  3.9361e-06,  2.6587e-05,  3.6385e-06,
         3.7748e-06,  1.5920e-06, -9.0481e-06,  5.2714e-06,  5.4956e-06,
        -6.6350e-06,  8.0748e-06,  1.7238e-05, -1.2331e-05,  8.5674e-06,
        -1.3933e-05,  4.9173e-08,  1.4189e-05,  1.6775e-05, -7.6865e-06,
         2.6996e-06,  8.6106e-06, -4.8269e-06,  2.0764e-06, -1.5731e-05,
        -2.0949e-05, -2.4599e-06, -1.5428e-07,  7.1664e-06,  4.0362e-07,
        -1.2956e-05,  2.4096e-07,  6.5539e-06,  2.3172e-05,  9.2163e-06,
         1.6930e-05, -2.2781e-05,  2.1748e-05, -1.0453e-05,  1.8253e-05,
         7.1217e-06,  6.3617e-06,  1.0136e-05,  2.7739e-06, -4.7644e-06,
         1.9112e-05,  6.1150e-06,  3.6009e-06, -1.7924e-05, -3.8630e-06,
         9.8341e-06, -3.9837e-07, -1.9135e-06,  9.8562e-07, -1.4955e-06,
         1.3269e-05, -9.9196e-06, -2.2945e-05, -1.9122e-05, -1.3693e-05,
        -1.7941e-05,  1.2339e-05, -1.2651e-05,  2.5163e-06, -7.6857e-06,
         1.0542e-05,  5.3130e-07, -1.9567e-05, -1.5703e-05, -1.5390e-05],
       dtype=torch.float64)

"""