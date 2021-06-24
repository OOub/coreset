import coreset
import numpy as np

data = np.random.rand(500, 5)
Np = 100

coreset, weights = coreset.generate(data, Np)
