cis_ch3ono = """
  0 1
    N                0.681170462229160    -0.189661062624943     0.660365006130894
    O                1.058947057078260    -0.717681452294326    -0.325238707719497
    O               -0.542817053890419     0.481810142533855     0.531068006427392
    C               -1.152032002868286     0.428626324995215    -0.772128132624601
    H               -2.072593658773097     0.997198233963335    -0.666806290397648
    H               -1.365148220691959    -0.603573612750921    -1.058489907840954
    H               -0.500989115810255     0.881461247614257    -1.523109442009479

symmetry c1
"""

trans_ch3ono = """
   0 1
    N                0.583593987538609    -0.412745501977718     0.227330962948888
    O                1.078150484943748    -0.083674463889344     1.232506658128671
    O               -0.517049451596286     0.440740036825867    -0.110537346161943
    C               -1.111042916935913     0.024795251341848    -1.344662449500125
    H               -2.158232501754135    -0.223176004011431    -1.162835252567883
    H               -0.582993887815015    -0.850163481317989    -1.731210335768984
    H               -1.043515795809969     0.846047492986682    -2.060372188098328

symmetry c1
"""

h2o = """
0 1
   O       -3.91287        2.15171       -0.01523
   H       -2.94498        2.18175        0.02018
   H       -4.19172        2.75204        0.69245
symmetry c1
"""


from driver import *
from multiprocessing import Process, Queue

queue = Queue()




p = Process(target = molecule, args = (h2o, 'cc-pvdz', 'rhf'), kwargs = {'log_file': 'new_data.log', 'optimize': False, 'mem': '60GB', 'scf': True, 'ucc3': False, 'ccsd': True, 'psi_acpf': False, 'psi_aqcc': False, 'shucc': True, 'cepa0': True, 'lccd': False, 'ccsdpt': False, 'cepa1': False, 'mp2': False, 'sys_name': 'h2o', 'ocepa': False, 'run_svd': False, 'tik_shucc': True, 'tik_cepa': True, 'omega': 0})
p.start()
p.join()

'''
p = Process(target = molecule, args = (trans_ch3ono, 'cc-pvtz', 'rhf'), kwargs = {'log_file': 'new_data.log', 'optimize': False, 'mem': '60GB', 'scf': True, 'ucc3': False, 'ccsd': True, 'psi_acpf': False, 'psi_aqcc': False, 'shucc': False, 'psi_cepa0': False, 'lccd': False, 'ccsdpt': False, 'cepa1': False, 'mp2': False, 'sys_name': 'trans_ch3ono', 'ocepa': False, 'run_svd': False, 'tik_shucc': True, 'tik_cepa': True, 'omega': .1})
p.start()
p.join()

p = Process(target = molecule, args = (cis_ch3ono, 'cc-pvtz', 'rhf'), kwargs = {'log_file': 'new_data.log', 'optimize': False, 'mem': '60GB', 'scf': True, 'ucc3': False, 'ccsd': True, 'psi_acpf': False, 'psi_aqcc': False, 'shucc': False, 'psi_cepa0': False, 'lccd': False, 'ccsdpt': False, 'cepa1': False, 'mp2': False, 'sys_name': 'trans_ch3ono', 'ocepa': False, 'run_svd': False, 'tik_shucc': True, 'tik_cepa': True, 'omega': .1})
p.start()
p.join()
'''
