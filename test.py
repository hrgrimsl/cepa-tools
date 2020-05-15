from driver import *
from multiprocessing import Process, Queue

geometry = """
0 1
   O        8.46572       12.31468        9.99183
   H        9.43361       12.35705       10.01083
   H        8.18687       13.16146       10.37154
symmetry c1
"""

geometry2 = """
0 1
   H 0 0 0 
   Cl 0 0 1
symmetry c1
"""

geometry3 = """
0 1
   H 0 0 0
   Cl 0 0 1

   O        100008.46572       100012.31468        100009.99183
   H        100009.43361       100012.35705       100010.01083
   H        100008.18687       100013.16146       100010.37154
symmetry c1
"""


queue = Queue()
p = Process(target = molecule, args = (geometry, 'cc-pvdz', 'rhf'), kwargs = {'optimize': False, 'cepa1': True, 'psi_acpf': True, 'acpf': False, 'psi_aqcc': True, 'aqcc': False, 'cepa0': False, 'ccsd': True, 'psi_cepa0': True, 'ucc3': True, 'shucc': True, 'scf': True, 'fci': False})
p.start()
p.join()

queue = Queue()
p = Process(target = molecule, args = (geometry2, 'cc-pvdz', 'rhf'), kwargs = {'optimize': False, 'cepa1': True, 'psi_acpf': True, 'acpf': False, 'psi_aqcc': True, 'aqcc': False, 'cepa0': False, 'ccsd': True, 'psi_cepa0': True, 'ucc3': True, 'shucc': True, 'scf': True, 'fci': False})
p.start()
p.join()

queue = Queue()
p = Process(target = molecule, args = (geometry3, 'cc-pvdz', 'rhf'), kwargs = {'optimize': False, 'cepa1': True, 'psi_acpf': True, 'acpf': False, 'psi_aqcc': True, 'aqcc': False, 'cepa0': False, 'ccsd': True, 'psi_cepa0': True, 'ucc3': True, 'shucc': True, 'scf': True, 'fci': False})
p.start()
p.join()
