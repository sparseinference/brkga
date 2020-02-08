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
    Use for keyshape: (population_size, parameter_keys + two_bounds_keys)
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
    Use for keyshape: (population_size, 3, parameter_keys)
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
        return (keys[2] * widths) + lowers
    #----
    def evaluate(keys):
        return fun(decode(keys))
    #----
    return bounds,decode,evaluate


def box3(fun, initialLower, initialUpper):
    """
    Use for keyshape: (population_size, 3, parameter_keys)
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


def Optimize(box, keyShape, elites=2, mutants=2):
    """
    Minimize the HappyCat function in the interval (-2,+2).
    """
    trial = 0
    bounds,decode,f = box  #box3(HappyCat, -2.0, 2.0)
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




def OptimizeGrad(box, keyShape, elites=2, mutants=2, lr=0.001):
    """
    Minimize the HappyCat function in the interval (-2,+2).
    This function uses gradient information to improve the evolved solutions.
    """
    trial = 0
    bounds,decode,f = box
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
    # Optimize(box0(HappyCat, -2.0, 2.0), (50,10), elites=3, mutants=3)
    # OptimizeGrad(box0(HappyCat, -2.0, 2.0), (20,10), elites=3, mutants=3, lr=1.0e-8)

    ## Optimize the bounding box by prepending lower and upper bounds to all parameter keys:
    # Optimize(box1(HappyCat, -2.0, 2.0), (10,10+2), elites=3, mutants=3)
    # OptimizeGrad(box1(HappyCat, -2.0, 2.0), (10,10+2), elites=3, mutants=3, lr=1.0e-10)

    ## Optimize lower and upper bounds for every parameter key:
    Optimize(box2(HappyCat, -2.0, 2.0), (20,3,10), elites=3, mutants=3)
    # OptimizeGrad(box2(HappyCat, -2.0, 2.0), (20,3,10), elites=3, mutants=3, lr=1.0e-8)

    ## Optimize lower and upper bounds for every parameter key:
    # Optimize(box3(HappyCat, -2.0, 2.0), (20,3,10), elites=3, mutants=3)
    # OptimizeGrad(box3(HappyCat, -2.0, 2.0), (20,3,10), elites=3, mutants=3, lr=1.0e-8)

"""
# OptimizeGrad(box1(HappyCat, -2.0, 2.0), (10,10+2), elites=3, mutants=3, lr=1.0e-10)

[ 20003] 0.03961472
BRKGA keys=
tensor([0.1348, 0.3538, 0.2620, 0.4432, 0.9986, 0.6629, 0.2453, 0.7208, 0.7745,
        0.9454, 0.5650, 0.0409], dtype=torch.float64)
HappyCat Bounds: lower:-1.4609493686795931 upper:-0.5846058774871854
HappyCat parameters=
tensor([-1.2314, -1.0726, -0.5859, -0.8800, -1.2460, -0.8293, -0.7822, -0.6325,
        -0.9658, -1.4251], dtype=torch.float64)


"""