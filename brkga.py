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


class BRKGA(torch.nn.Module):
    def __init__(self, populationShape, elites=1, mutants=1, optimizer=None):
        super().__init__()
        self.keys = torch.nn.Parameter(torch.rand(*populationShape))
        count = len(self.keys)
        self.eliteCount = max(1, elites)
        self.mutantCount = max(1, mutants)
        self.nonMutantCount = count - self.mutantCount
        self.optimizer = optimizer(self.parameters()) if optimizer is not None else None
    #------------------------------------------------------
    def orderBy(self, results, descending=False):
        """
        Sort the population of keys by the objective function 'results'.
        results: a tensor of objective function values (1D tensor), one value per row in 'self.keys'.
        """
        values,indexes = results.sort(descending=descending)
        self.keys = torch.nn.Parameter(self.keys[indexes])
        return values[0],self.keys[0]
    #------------------------------------------------------
    @property
    def elites(self):
        """
        Return a tensor of the best random keys in the population.
        PRECONDITION: The population of random keys are sorted in ascending order 
                      by the most recent call to 'self.orderBy'.
        """
        return self.keys[:self.eliteCount]
    #------------------------------------------------------
    def sgd(self, lr):
        """
        Adjust the population of random keys 
        in the direction of their negative gradient.
        """
        with torch.no_grad():
            self.keys.data -= lr * self.keys.grad 
            self.keys.data.clamp_(min=0, max=1)
            self.keys.grad.zero_()
    #------------------------------------------------------
    def optimize(self):
        """
        Use an application-specific optimizer (with a PyTorch API).
        """
        with torch.no_grad():
            self.optimizer.step()
            self.keys.data.clamp_(min=0, max=1)
            self.zero_grad()
    #------------------------------------------------------
    def forward(self, closure):
        """
        Compute a results tensor by applying the 'closure' to 
        each random key in the population.
        """
        return torch.stack([closure(x) for x in self.keys])
    #------------------------------------------------------
    def evolve(self):
        """
        One iteration of the "Biased Random-Key Genetic Algorithm".
        Replaces the current population of random keys with a new population.
        PRECONDITION: The population of random keys are sorted in ascending order 
                      by the most recent call to 'self.orderBy'.
        POSTCONDITION: The shape of the population of random keys is unchanged.
                       The first 'self.eliteCount' random keys are still the best keys from
                       the most recent call to 'self.orderBy'.
                       The remaining keys in the population are constructed by biased crossover.
        """
        with torch.no_grad():
            keyShape = self.keys.shape[1:]
            #----
            elites = self.keys[:self.eliteCount]
            mutants = torch.rand(self.mutantCount, *keyShape)
            nonElites = torch.cat([self.keys[self.eliteCount : self.nonMutantCount], mutants], 0)
            nonEliteCount = len(nonElites)
            #----
            eliteSelectors = torch.multinomial(torch.full((self.eliteCount,), 0.5), nonEliteCount, replacement=True)
            selectedElites = elites[eliteSelectors]
            #----
            nonEliteSelectors = torch.multinomial(torch.full((nonEliteCount,), 0.5), nonEliteCount, replacement=True)
            selectedNonElites = nonElites[nonEliteSelectors]
            #----
            # offspringSelectors = torch.rand(nonEliteCount, *keyShape).bernoulli()
            offspringSelectors = torch.full((nonEliteCount, *keyShape), 0.5).bernoulli()
            offspring = (offspringSelectors * selectedElites) + ((1 - offspringSelectors) * selectedNonElites)
            #----
            self.keys = torch.nn.Parameter(torch.cat([elites, offspring], 0))
    #------------------------------------------------------





##===================================================

if __name__ == '__main__':
    pass

    