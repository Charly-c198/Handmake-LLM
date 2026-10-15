import torch
import numpy as np
from numpy.ma.core import size


def dm01():
    t1 = torch.tensor(10)
    print(f't1:{t1}')

    t2 = torch.tensor(5)
    print(t2)
    print(f't2:{t2}')

    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.tensor(data)
    print(f't3:{t3}')



if __name__ == '__main__':
    dm01()

