import random
import numpy as np
import numpy.random as rd


# defining problems

class ZDT1():  

    def __init__(self):
      self.lower = 0
      self.upper = 1
      self.size = 30
      self.pop = 80
      self.var = [rd.rand(self.size) for i in range(self.pop)] # 20 * 30 array

    def zdt1_f1(self):
      return [i[0] for i in self.var] 
    
    def zdt1_f2(self):
      f1 = self.zdt1_f1() 
      g = [1 + (9/(self.size-1)) * sum(i[1:]) for i in self.var]
      h = [1 - np.sqrt(i/j) for i,j in zip(f1, g)]
      f2 = [i*j for i,j in zip(g,h)]
      return f2 # list

class ZDT2():
    
    def __init__(self):
      self.lower = 0
      self.upper = 1
      self.size = 30
      self.pop = 50
      self.var = [rd.rand(self.size) for i in range(self.pop)]

    def zdt2_f1(self):
      return [i[0] for i in self.var] 
    
    def zdt2_f2(self):
      f1 = self.zdt2_f1() # list
      g = [1 + (9/(self.size-1)) * sum(i[1:]) for i in self.var]
      h = [1 - (i/j)**2 for i,j in zip(f1, g)]
      f2 = [i*j for i,j in zip(g,h)]
      return f2 # list

class Origin():

    def __init__(self):
        self.lower = -20
        self.upper = 20
        self.size = 50
        self.var = [self.lower + (self.upper - self.lower) * random.random() for i in range(self.size)]

    def func_1(self):
        value = [-x**2 for x in self.var]
        return value

    def func_2(self):
        value = [-(x-2)**2 for x in self.var]
        return value

class SCH():

    def __init__(self):
        self.lower = -100
        self.upper = 100
        self.size = 50
        self.var = [self.lower + (self.upper - self.lower)*random.random() for i in range(self.size)]
    
    def func_1(self):
        value = [i**2 for i in self.var]
        return value
    
    def func_2(self):
        value = [(i-2)**2 for i in self.var]
        return value