"""
File:  brkga.py
Created: 2019-06-16
Copyright 2019 Peter Caven, peter@sparseinference.com

Description:

An implementation of the "Biased Random-Key Genetic Algorithm".
[1] Random-key genetic algorithms, by José Fernando Gonçalves, Mauricio G. C. Resende, 
    August 18, 2014, http://mauricio.resende.info/


"""

import torch

#===============================================================================
# Wrapped Optimizers
#===============================================================================

def optSGD(lr=1.0e-3, momentum=0.0, dampening=0, weight_decay=0, nesterov=False):
    def opt(params):
        return torch.optim.SGD(params, 
                                lr=lr, 
                                momentum=momentum, 
                                dampening=dampening, 
                                weight_decay=weight_decay, 
                                nesterov=nesterov)
    return opt


def optAdam(lr=1.0e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    def opt(params):
        return torch.optim.Adam(params, 
                                lr=lr, 
                                betas=betas, 
                                eps=eps, 
                                weight_decay=weight_decay, 
                                amsgrad=amsgrad)
    return opt


def optAdagrad(lr=1.0e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
    def opt(params):
        return torch.optim.Adagrad(params, 
                                lr=lr, 
                                lr_decay=lr_decay, 
                                weight_decay=weight_decay, 
                                initial_accumulator_value=initial_accumulator_value)
    return opt


def optAdadelta(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
    def opt(params):
        return torch.optim.Adadelta(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
    return opt


def optRMSprop(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
    def opt(params):
        return torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
    return opt



#===============================================================================




class BRKGA():
    def __init__(self, populationShape, elites=1, mutants=1, optimizer=optSGD(), dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.keys = torch.rand(*populationShape, requires_grad=True, dtype=dtype)
        self.indexes = None
        self.eliteCount = max(1, elites)
        self.mutantCount = max(0, mutants)
        self.nonMutantCount = len(self.keys) - self.mutantCount
        self.optimizer = optimizer([self.keys])
    #------------------------------------------------------
    def orderBy(self, results, descending=False):
        """
        Sort the population of keys by the 'results' of mapping an objective function to each random key.
        The best result and its random key are returned.
        results: a tensor of objective function values (1D tensor), one value per row in 'self.keys'.
        """
        values,self.indexes = results.sort(descending=descending)
        return values[0],self.keys[self.indexes[0]]
    #------------------------------------------------------
    @property
    def elites(self):
        """
        Return a tensor of the best random keys in the population.
        PRECONDITION: The indexes of the random keys are sorted
                      by the most recent call to 'self.orderBy'.
        """
        return self.keys[self.indexes[:self.eliteCount]]
    #------------------------------------------------------
    @property
    def nonelites(self):
        """
        Return a tensor of the non-elite random keys in the population.
        PRECONDITION: The indexes of the random keys are sorted
                      by the most recent call to 'self.orderBy'.
        """
        return self.keys[self.indexes[self.eliteCount:]]
    #------------------------------------------------------
    def optimize(self):
        """
        Use the optimizer passed to the constructor to adjust the random keys
        using the gradients computed by a call to 'results.backward()'.
        Note: the elites may not be the best random keys after this call,
        but the entire population of random keys will (stochastically) 
        move toward the optimum.
        """
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.keys.data.clamp_(min=0, max=1)
    #------------------------------------------------------
    def map(self, func):
        """
        Compute a results tensor by applying 'func' to 
        each random key in the population.
        """
        return torch.stack([func(x) for x in self.keys])
    #------------------------------------------------------
    def evolve(self):
        """
        One iteration of the "Biased Random-Key Genetic Algorithm".
        Replaces the current population of random keys with a new population.
        PRECONDITION: The indexes of the random keys are already sorted
                      by a call to 'self.orderBy'.
        POSTCONDITION: The shape of the population of random keys is unchanged.
                       The first 'self.eliteCount' random keys are still the best keys from
                       the most recent call to 'self.orderBy'.
        """
        with torch.no_grad():
            keyShape = self.keys.shape[1:]
            #----
            if self.mutantCount > 0:
                mutants = torch.rand(self.mutantCount, *keyShape, dtype=self.dtype)
                nonElites = torch.cat([self.keys[self.indexes[self.eliteCount : self.nonMutantCount]], mutants], 0)
            else:
                nonElites = self.keys[self.indexes[self.eliteCount : self.nonMutantCount]]
            nonEliteCount = len(nonElites)
            #----
            elites = self.elites
            eliteSelectors = torch.multinomial(torch.full((self.eliteCount,), 0.5), nonEliteCount, replacement=True)
            selectedElites = elites[eliteSelectors]
            #----
            nonEliteSelectors = torch.multinomial(torch.full((nonEliteCount,), 0.5), nonEliteCount, replacement=True)
            selectedNonElites = nonElites[nonEliteSelectors]
            #----
            offspringSelectors = torch.full((nonEliteCount, *keyShape), 0.5, dtype=self.dtype).bernoulli()
            offspring = (offspringSelectors * selectedElites) + ((1.0 - offspringSelectors) * selectedNonElites)
            #----
            torch.cat([elites, offspring], 0, out=self.keys)
    #------------------------------------------------------





##===================================================

if __name__ == '__main__':
    pass

    