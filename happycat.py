"""
File: happycat.py
By Peter Caven, peter@sparseinference.com

Description:

HappyCat test function for the "Biased Random-Key Genetic Algorithm".

See: "HappyCat – A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail", 
        by Hans-Georg Beyer and Steffen Finck, 2012

"""

import torch
from brkga import BRKGA,optAdam


def HappyCat(x, alpha=1/8):
    X = x.dot(x)
    N = len(x)
    return (X - N).pow(2.0).pow(alpha) + (X.div(2.0) + x.sum())/N + 0.5


def box0(fun, lower, upper):
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


def box1(fun, initialLower, initialUpper):
    """
    1 dimension per population member.
    For keyshape: (population_size, parameter keys + two bounds keys)
    where the second dimension has lower and upper bounds keys
    prepended to the parameters keys.
    """
    initialWidth = initialUpper - initialLower
    #----
    def bounds(keys):
        lowerKey = keys[0]
        upperKey = keys[1]
        if lowerKey > upperKey:
            lowerKey,upperKey = upperKey,lowerKey
        lower = (lowerKey * initialWidth) + initialLower
        upper = (upperKey * initialWidth) + initialLower
        return lower,upper
    #----
    def decode(keys):
        lower,upper = bounds(keys)
        width = upper - lower
        #---
        return (keys[2:] * width) + lower
    #----
    def evaluate(keys):
        return fun(decode(keys))
    #----
    return bounds,decode,evaluate


def box2(fun, initialLower, initialUpper):
    """
    2 dimensions per population member.
    For keyshape: (population size, 3, parameter keys)
    where the first parameter dimension is two rows of lower and upper bounds keys,
    and one row of parameters keys.
    The bounds are learned, two per parameter.
    """
    #----
    def bounds(keys):
        lowers = initialLower + (3.0 * keys[0])
        uppers = initialUpper - (3.0 * keys[1])
        return lowers,uppers
    #----
    def decode(keys):
        lowers,uppers = bounds(keys)
        widths = uppers - lowers
        return (x[2] * widths) + lowers
    #----
    def evaluate(keys):
        return fun(decode(keys))
    #----
    return bounds,decode,evaluate


def box3(fun, initialLower, initialUpper):
    """
    2 dimensions per population member.
    For keyshape: (population size, 3, parameter keys)
    where the first parameter dimension is two rows of lower and upper bounds keys,
    and one row of parameters keys.
    The bounds are learned, two per parameter.
    """
    initialWidth = initialUpper - initialLower
    #----
    def bounds(keys):
        v,_ = keys[:2].sort(-2)
        lowerKeys = v[0]
        upperKeys = v[1]
        lowers = (lowerKeys * initialWidth) + initialLower
        uppers = (upperKeys * initialWidth) + initialLower
        return lowers,uppers
    #----
    def decode(keys):
        lowers,uppers = bounds(keys)
        widths = uppers - lowers
        valueKeys = keys[2]
        return (valueKeys * widths) + lowers
    #----
    def evaluate(keys):
        return fun(decode(keys))
    #----
    return bounds,decode,evaluate


def Optimize(keyShape, elites=2, mutants=2):
    """
    Minimize the HappyCat function in the interval (-2,+2).
    """
    trial = 0
    bounds,decode,f = box1(HappyCat, -2.0, 2.0)
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
        hcLower,hcUpper = bounds(best.data)
        print(f"HappyCat Bounds: lower:{hcLower} upper:{hcUpper}")
        print(f"HappyCat parameters=\n{decode(best.data)}")
    return bestResult,best




def OptimizeGrad(keyShape, elites=2, mutants=2, lr=0.001):
    """
    Minimize the HappyCat function in the interval (-2,+2).
    This function uses gradient information to improve the evolved solutions.
    """
    trial = 0
    bounds,decode,f = box1(HappyCat, -2.0, 2.0)
    pop = BRKGA(keyShape, elites=elites, mutants=mutants, optimizer=optAdam(lr=lr))
    try:
        results = pop.map(f)
        bestResult,best = pop.orderBy(results)
        while bestResult > 1.0e-7:
            if trial % 100 == 0:
                print(f"[{trial:6d}] {bestResult:.8f}")
                print(decode(best.data))
            trial += 1
            pop.evolve()
            results = pop.map(f)
            results.mean().backward()
            pop.optimize()
            results = pop.map(f)
            bestResult,best = pop.orderBy(results)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[{trial:6d}] {bestResult:.8f}")
        print(f"BRKGA keys=\n{best.data}")
        hcLower,hcUpper = bounds(best.data)
        print(f"HappyCat Bounds: lower:{hcLower} upper:{hcUpper}")
        print(f"HappyCat parameters=\n{decode(best.data)}")
    return bestResult,best





##===================================================

if __name__ == '__main__':
    
    ## Optimize only the parameter keys:
    # Optimize((20,10), elites=3, mutants=3)
    # OptimizeGrad((20,10), elites=3, mutants=3, lr=1.0e-8)

    ## Optimize the bounding box by prepending lower and upper bounds to all parameter keys:
    # Optimize((20,10+2), elites=3, mutants=3)
    OptimizeGrad((20,10+2), elites=3, mutants=3, lr=1.0e-10)

    ## Optimize lower and upper bounds for every parameter key:
    # Optimize((20,3,10), elites=3, mutants=3)
    # OptimizeGrad((20,3,10), elites=3, mutants=3, lr=1.0e-8)


"""
# OptimizeGrad((10,10+2), elites=3, mutants=3, lr=1.0e-10)

[ 33751] 0.02897739
BRKGA keys=
tensor([0.3464, 0.1415, 0.7916, 0.8312, 0.5307, 0.3156, 0.2378, 0.9082, 0.9056,
        0.6130, 0.2166, 0.2535], dtype=torch.float64)
HappyCat Bounds: lower:-1.4338566499343643 upper:-0.614287517658779
HappyCat parameters=
tensor([-0.7851, -0.7526, -0.9989, -1.1752, -1.2390, -0.6895, -0.6917, -0.9315,
        -1.2563, -1.2261], dtype=torch.float64)


"""