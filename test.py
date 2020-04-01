from driver import *
from multiprocessing import Process, Queue

geometry = """
0 2
   C       -4.67547        1.47099       -0.00000
   H       -4.67547        0.45998       -0.38699
   H       -4.67547        2.31164       -0.68207
   H       -4.67547        1.64136        1.06906
symmetry c1
"""

geometry2 = """
0 1
  Cl       -4.14193        1.38332        0.00000
  Cl       -2.22337        0.77732        0.00000
symmetry c1
"""

geometry3 = """
   C       -3.29000        0.34682       -0.00000
  Cl       -1.52303        0.34682        0.00000
   H       -3.64703        1.09071       -0.71558
   H       -3.64703        0.59459        1.00202
   H       -3.64703       -0.64483       -0.28644
symmetry c1
"""

geometry4 = """
0 2
Cl 0 0 0
symmetry c1
"""
queue = Queue()
p = Process(target = molecule, args = (geometry, 'cc-pvdz', 'uhf'), kwargs = {'optimize': True})
p.start()
p.join()

p = Process(target = molecule, args = (geometry2, 'cc-pvdz', 'rhf'), kwargs = {'optimize': True})
p.start()
p.join()

p = Process(target = molecule, args = (geometry3, 'cc-pvdz', 'rhf'), kwargs = {'optimize': True})
p.start()
p.join()

p = Process(target = molecule, args = (geometry4, 'cc-pvdz', 'uhf'), kwargs = {'optimize': False})
p.start()
p.join()


#ch3 = molecule(geometry, 'cc-pvdz', 'uhf', optimize = True)

#cl2 = molecule(geometry2, 'cc-pvdz', 'rhf', optimize = True)

#ch3cl = molecule(geometry3, 'cc-pvdz', 'rhf', optimize = True)

#cl = molecule(geometry4, 'cc-pvdz', 'uhf', optimize = False)


#print(he2.cepa() + h.cepa())
