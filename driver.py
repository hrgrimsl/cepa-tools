import numpy as np
from opt_einsum import contract
import copy
import scipy
import os
import algebratools as at
import time

class molecule:
    def __init__(self, geometry, basis, reference, **kwargs):
        import psi4

        #Miscellaneous
        self.sys_name = "unnamed system"
        self.log_file = 'out.dat'
        self.verbose = False
        self.mem = '24GB'

        #Numerical Parameters
        self.d_convergence = 1e-10
        self.tol = 1e-6
        self.tik_omega = 0
        self.nik_omega = 0

        #Psi4 Methods
        self.optimize = False
        self.scf = False
        self.mp2 = False
        self.cisd = False
        self.ccsd = False
        self.lccd = False
        self.cepa1 = False
        self.ccsdpt = False
        self.psi_cepa0 = False
        self.psi_acpf = False
        self.psi_aqcc = False
        self.ocepa = False
        self.fci = False

        #My Methods
        self.cepa0 = False
        self.shucc = False
        self.acpf = False
        self.aqcc = False
        self.uacpf = False
        self.uaqcc = False
        self.ucc3 = False
        self.tik_cepa = False
        self.tik_shucc = False
        self.nik_cepa = False
        self.nik_shucc = False
        
        if kwargs['combo'] == 'debug' and reference == 'rhf':
            self.scf = True
            self.mp2 = True
            self.cisd = True
            self.ccsd = True
            self.lccd = True
            self.cepa1 = True
            self.ccsdpt = True
            self.fci = True
            self.psi_cepa0 = True
            self.psi_acpf = True
            self.psi_aqcc = True
            self.ocepa = True
            self.cepa0 = True
            self.shucc = True
            self.acpf = True
            self.aqcc = True
            self.uacpf = True
            self.uaqcc = True
            self.ucc3 = True
            self.tik_cepa = True
            self.tik_shucc = True
            self.nik_cepa = True
            self.nik_shucc = True

        if kwargs['combo'] == 'general' and reference == 'uhf':
            self.scf = True
            self.mp2 = True
            self.ccsd = True
            self.lccd = True
            self.ccsdpt = True
            self.ocepa = True
            self.cepa0 = True
            self.shucc = True
            self.acpf = True
            self.aqcc = True
            self.uacpf = True
            self.uaqcc = True
            self.ucc3 = True
            self.tik_cepa = True
            self.tik_shucc = True
            self.nik_cepa = True
            self.nik_shucc = True

        if kwargs['combo'] == 'general' and reference == 'rhf':
            self.scf = True
            self.mp2 = True
            self.cisd = True
            self.ccsd = True
            self.lccd = True
            self.cepa1 = True
            self.ccsdpt = True
            self.psi_cepa0 = True
            self.psi_acpf = True
            self.psi_aqcc = True
            self.ocepa = True
            self.shucc = True
            self.uacpf = True
            self.uaqcc = True
            self.ucc3 = True
            self.tik_cepa = True
            self.tik_shucc = True
            self.nik_cepa = True
            self.nik_shucc = True
        

        

        for key, value in kwargs.items():
            setattr(self, key, value)
        try:
            assert reference == 'rhf' or reference == 'uhf'
        except:
            print("Only \"rhf\" or \"uhf\" allowed.")
            exit()
        self.reference = reference

        psi4.geometry(geometry)
        if self.verbose == False:
            psi4.core.be_quiet()
        psi4.core.clean()
        if os.path.exists(self.log_file):
            cha = 'a'
        else:
            cha = 'w'
        log = open(self.log_file, cha)
        print("Performing calculations on "+str(self.sys_name)+".")
        psi4.set_memory(self.mem)
        psi4.set_options({'reference': reference, 'd_convergence': self.d_convergence, 'scf_type': 'pk', 'maxiter': 100})

        if self.optimize != False:
            print("Optimizing geometry using Psi4...\n")
            psi4.set_options({'g_convergence': 'GAU_TIGHT', 'geom_maxiter': 500, 'opt_coordinates': self.optimize})
            E, wfnopt = psi4.optimize('b3lyp/6-311G(d,p)', return_wfn = True)
            log.write(self.sys_name)  
            log.write((wfnopt.molecule().create_psi4_string_from_molecule()))
            print("Geometry for "+str(self.sys_name)+" written to "+str(self.log_file)+".")

        psi4.set_options({'basis': basis})
        energies = {}
        print("\nEvaluating Psi4 energies...\n")
        self.hf_energy, wfn = psi4.energy('scf', return_wfn = True)
        if self.scf == True:
            energies['scf'] = self.hf_energy
            print("HF energy:".ljust(30)+("{0:20.16f}".format(self.hf_energy)))
        if self.mp2 == True:
            energies['mp2'] = psi4.energy('mp2')
            print("MP2 energy:".ljust(30)+("{0:20.16f}".format(energies['mp2'])))   
        if self.cisd == True:
            energies['cisd'] = psi4.energy('cisd')
            print("CISD energy:".ljust(30)+("{0:20.16f}".format(energies['cisd'])))   
        if self.ccsd == True:
            energies['ccsd'] = psi4.energy('ccsd')
            print("CCSD energy:".ljust(30)+("{0:20.16f}".format(energies['ccsd'])))   
        if self.ccsdpt == True:
            energies['ccsdpt'] = psi4.energy('ccsd(t)')
            print("CCSD(T) energy:".ljust(30)+("{0:20.16f}".format(energies['ccsdpt'])))   
        if self.fci == True:
            energies['fci'] = psi4.energy('fci')
            print("FCI energy:".ljust(30)+("{0:20.16f}".format(energies['fci'])))   
        if self.lccd == True:
            energies['lccd'] = psi4.energy('lccd')
            print("LCCD energy:".ljust(30)+("{0:20.16f}".format(energies['lccd'])))   
        if self.ocepa == True:
            energies['olccd'] = psi4.energy('olccd')
            print("OLCCD energy:".ljust(30)+("{0:20.16f}".format(energies['olccd'])))   
        if self.psi_cepa0 == True:
            energies['lccsd'] = psi4.energy('cepa(0)')
            print("LCCSD energy:".ljust(30)+("{0:20.16f}".format(energies['lccsd'])))   
        if self.psi_acpf == True:
            energies['acpf'] = psi4.energy('acpf')
            print("ACPF energy:".ljust(30)+("{0:20.16f}".format(energies['acpf'])))   
        if self.psi_aqcc == True:
            energies['aqcc'] = psi4.energy('aqcc')
            print("AQCC energy:".ljust(30)+("{0:20.16f}".format(energies['aqcc'])))   
        if self.cepa1 == True:
            energies['cepa1'] = psi4.energy('cepa(1)')
            print("CEPA(1) energy:".ljust(30)+("{0:20.16f}".format(energies['cepa1'])))   

        mints = psi4.core.MintsHelper(wfn.basisset())
        ca = wfn.Ca()
        cb = wfn.Cb()
        self.fa = wfn.Fa()
        self.fb = wfn.Fb()
        self.j_aaaa = np.array(mints.mo_eri(ca, ca, ca, ca))
        self.j_abab = np.array(mints.mo_eri(ca, ca, cb, cb))
        self.j_baba = np.array(mints.mo_eri(cb, cb, ca, ca))
        self.j_bbbb = np.array(mints.mo_eri(cb, cb, cb, cb))
        self.fa.transform(ca)
        if reference != 'rhf':
            self.fb.transform(cb)
        self.fa = np.array(self.fa)
        self.fb = np.array(self.fb)
        self.j_aaaa = self.j_aaaa.swapaxes(1, 2)
        self.j_abab = self.j_abab.swapaxes(1, 2)
        self.j_baba = self.j_baba.swapaxes(1, 2)
        self.j_bbbb = self.j_bbbb.swapaxes(1, 2)
        k_aaaa = copy.copy(self.j_aaaa)
        k_bbbb = copy.copy(self.j_bbbb)
        k_aaaa = k_aaaa.swapaxes(2, 3)
        k_bbbb = k_bbbb.swapaxes(2, 3)
        self.l_aaaa = self.j_aaaa - k_aaaa
        self.l_bbbb = self.j_bbbb - k_bbbb
        self.noa = wfn.Ca_subset("AO", "ACTIVE_OCC").shape[1]
        self.nob = wfn.Cb_subset("AO", "ACTIVE_OCC").shape[1]
        self.nva = wfn.Ca_subset("AO", "ACTIVE_VIR").shape[1]
        self.nvb = wfn.Cb_subset("AO", "ACTIVE_VIR").shape[1]
        self.aopairs = np.triu_indices(self.noa, k=1)
        self.avpairs = np.triu_indices(self.nva, k=1)
        self.bopairs = np.triu_indices(self.nob, k=1)
        self.bvpairs = np.triu_indices(self.nvb, k=1)
        psi4.core.clean()
        self.b = self.cepa_b()
        self.b = at.collapse_tensor(self.b, self)
        self.b = at.concatenate_amps(self.b, self)
        self.fock_time = 0
        self.fock_no = 0
        self.N = self.noa + self.nob
        self.Finv_op = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.Finv, rmatvec = self.Finv)
        self.F2inv_op = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.F2inv, rmatvec = self.F2inv)
        
        print("\nEvaluating proprietarily implemented energies...\n")
        
        if self.cepa0 == True: 
            outer_start = time.time()
            self.ucc_term = False
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
            x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
            energy = self.energy(x)
            print("LCCSD:")
            print("*"*60)
            print("    "+"{0:0.2f}".format(self.matvec_time)+ " seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['lccsd'] = copy.copy(energy)

        if self.shucc == True:
            outer_start = time.time()
            self.ucc_term = True
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
            x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
            energy = self.energy(x)
            print("UCCSD(2):")
            print("*"*60)
            print("    "+"{0:0.2f}".format(self.matvec_time)+ " seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['shucc'] = copy.copy(energy)

        if self.acpf == True: 
            outer_start = time.time()
            print("ACPF:")
            print("*"*60)
            self.ucc_term = False
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            self.shift = 0
            iter = 1
            energy = copy.copy(self.hf_energy)
            Done = False
            while Done == False:
                 Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
                 x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
                 dE = self.energy(x) - energy
                 if abs(dE) < self.tol**2:
                     Done = True 
                 energy = self.energy(x)
                 self.shift = -2*(energy - self.hf_energy)/self.N
                 print("    {0:2d}".format(iter)+".    "+"{0:20.16f}".format(energy))
                 iter += 1
            print("\n")
            print("    "+"{0:0.2f}".format(self.matvec_time)+" seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['acpf'] = copy.copy(energy)

        if self.aqcc == True: 
            outer_start = time.time()
            print("AQCC:")
            print("*"*60)
            self.ucc_term = False
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            self.shift = 0
            iter = 1
            energy = copy.copy(self.hf_energy)
            Done = False
            while Done == False:
                 Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
                 x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
                 dE = self.energy(x) - energy
                 if abs(dE) < self.tol**2:
                     Done = True 
                 energy = self.energy(x)
                 self.shift = -(1-((self.N-3)*(self.N-2)/(self.N*(self.N-1))))*(energy - self.hf_energy)
                 print("    {0:2d}".format(iter)+".    "+"{0:20.16f}".format(energy))
                 iter += 1
            print("\n")
            print("    "+"{0:0.2f}".format(self.matvec_time)+" seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['aqcc'] = copy.copy(energy)

        if self.uacpf == True: 
            outer_start = time.time()
            print("UACPF:")
            print("*"*60)
            self.ucc_term = True
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            self.shift = 0
            iter = 1
            energy = copy.copy(self.hf_energy)
            Done = False
            while Done == False:
                 Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
                 x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
                 dE = self.energy(x) - energy
                 if abs(dE) < self.tol**2:
                     Done = True 
                 energy = self.energy(x)
                 self.shift = -2*(energy - self.hf_energy)/self.N
                 print("    {0:2d}".format(iter)+".    "+"{0:20.16f}".format(energy))
                 iter += 1
            print("\n")
            print("    "+"{0:0.2f}".format(self.matvec_time)+" seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['uacpf'] = copy.copy(energy)

        if self.uaqcc == True: 
            outer_start = time.time()
            print("UAQCC:")
            print("*"*60)
            self.ucc_term = True
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            self.shift = 0
            iter = 1
            energy = copy.copy(self.hf_energy)
            Done = False
            while Done == False:
                 Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
                 x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
                 dE = self.energy(x) - energy
                 if abs(dE) < self.tol**2:
                     Done = True 
                 energy = self.energy(x)
                 self.shift = -(1-((self.N-3)*(self.N-2)/(self.N*(self.N-1))))*(energy - self.hf_energy)
                 print("    {0:2d}".format(iter)+".    "+"{0:20.16f}".format(energy))
                 iter += 1
            print("\n")
            print("    "+"{0:0.2f}".format(self.matvec_time)+" seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['uaqcc'] = copy.copy(energy)
 
        if self.tik_cepa == True:
            outer_start = time.time()
            self.ucc_term = False
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.tik_H, rmatvec = self.tik_H) 
            x, info = scipy.sparse.linalg.cg(Aop, -self.H(self.b), tol = self.tol, M = self.F2inv_op)
            energy = self.energy(x)
            print("Tikhonov-regularized CEPA:")
            print("*"*60)
            print("    "+"{0:0.2f}".format(self.matvec_time)+ " seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['tik_cepa'] = copy.copy(energy)

        if self.tik_shucc == True:
            outer_start = time.time()
            self.ucc_term = True
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.tik_H, rmatvec = self.tik_H) 
            x, info = scipy.sparse.linalg.cg(Aop, -self.H(self.b), tol = self.tol, M = self.F2inv_op)
            energy = self.energy(x)
            print("Tikhonov-regularized UCCSD(2):")
            print("*"*60)
            print("    "+"{0:0.2f}".format(self.matvec_time)+ " seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['tik_shucc'] = copy.copy(energy)

        if self.nik_cepa == True:
            outer_start = time.time()
            self.ucc_term = False
            self.shift = copy.copy(self.nik_omega**2)
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
            x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
            energy = self.energy(x)
            print("Nikhonov-regularized CEPA:")
            print("*"*60)
            print("    "+"{0:0.2f}".format(self.matvec_time)+ " seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['nik_cepa'] = copy.copy(energy)

        if self.nik_shucc == True:
            outer_start = time.time()
            self.ucc_term = True
            self.shift = copy.copy(self.nik_omega**2)
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
            x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
            energy = self.energy(x)
            print("Nikhonov-regularized UCCSD(2):")
            print("*"*60)
            print("    "+"{0:0.2f}".format(self.matvec_time)+ " seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['nik_shucc'] = copy.copy(energy)

        if self.ucc3 == True:
            outer_start = time.time()
            print("EN-UCCSD(3):")
            print("*"*60)
            self.ucc_term = True
            self.shift = 0
            self.count_no = 0
            self.fock_no = 0
            self.matvec_time = 0
            self.fock_time = 0
            self.shift = 0
            iter = 1
            energy = copy.copy(self.hf_energy)
            Done = False
            while Done == False:
                 Aop = scipy.sparse.linalg.LinearOperator((len(self.b), len(self.b)), matvec = self.H, rmatvec = self.H) 
                 x, info = scipy.sparse.linalg.cg(Aop, -self.b, tol = self.tol, M = self.Finv_op)
                 new_energy = self.energy(x)+(2/3)*np.multiply(self.b,x).T.dot(np.multiply(x,x))
                 if abs(new_energy-energy) < self.tol**2:
                     Done = True 
                 energy = copy.copy(new_energy)
                 self.shift = -2*np.multiply(self.b, x)
                 print("    {0:2d}".format(iter)+".    "+"{0:20.16f}".format(energy))
                 iter += 1
            print("\n")
            print("    "+"{0:0.2f}".format(self.matvec_time)+" seconds spent computing "+str(self.count_no)+" Hamiltonian actions.")
            print("    "+"{0:0.2f}".format(self.fock_time)+ " seconds spent computing "+str(self.fock_no)+" Fock actions.")
            print("    "+"{0:0.2f}".format(time.time() - outer_start)+ " seconds elapsed total.")
            print("    Converged energy:".ljust(30)+("{0:20.16f}".format(energy)))
            print("\n")
            energies['ucc3'] = copy.copy(energy)

        print("Summary of energies:")
        for key in energies:
            print(str(key+" energy:").ljust(30)+("{0:20.16f}".format(energies[key])))   
        self.energies = energies 
            
    def build_arr(self, op):
        arr = np.zeros((len(self.b), len(self.b)))
        for i in range(0, len(self.b)):
            x = np.zeros(len(self.b))
            x[i] = 1.0
            arr[:,i] = op(x)
        return arr
  
    def F(self, x):
        fstart = time.time()
        if self.reference == 'rhf':
            x = at.rhf_to_uhf(x, self)
        x = at.decatenate_amps(x, self, reference = 'uhf')
        x = at.expand_vector(x, self, reference = 'uhf')

        #Initialize Hessian Shapes
        Ax = {} 
        Ax['a'] = np.zeros((self.noa, self.nva))
        Ax['b'] = np.zeros((self.nob, self.nvb))
        Ax['aa'] = np.zeros((self.noa, self.noa, self.nva, self.nva))
        Ax['ab'] = np.zeros((self.noa, self.nob, self.nva, self.nvb))
        Ax['bb'] = np.zeros((self.nob, self.nob, self.nvb, self.nvb))

        #S->S F (Particle Interaction)
        Ax['a'] += contract('ab,ib->ia', self.fa[self.noa:, self.noa:], x['a'])
        if self.reference != 'rhf':
            Ax['b'] += contract('ab,ib->ia', self.fb[self.nob:, self.nob:], x['b'])

        #S->S F (Hole Interaction)
        Ax['a'] -= contract('ji,ja->ia', self.fa[:self.noa, :self.noa], x['a'])
        if self.reference != 'rhf':
            Ax['b'] -= contract('ji,ja->ia', self.fb[:self.nob, :self.nob], x['b'])


        #D->D F (Hole Interaction)
        Ax['ab'] -= contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['ab'])
        Ax['ab'] -= contract('ki,kjab->ijab', self.fa[:self.noa, :self.noa], x['ab'])
        if self.reference != 'rhf':     
            Ax['aa'] -= (contract('kj,ikab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['aa'] += (contract('ki,jkab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['bb'] -= (contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            Ax['bb'] += (contract('ki,jkab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
        else:
            Ax['aa'] -= .5*(contract('kj,ikab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['aa'] += .5*(contract('ki,jkab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['aa'] -= .5*(contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            Ax['aa'] += .5*(contract('ki,jkab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            
        #D->D F (Particle Interaction)
        Ax['ab'] += contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['ab'])
        Ax['ab'] += contract('ac,ijcb->ijab', self.fa[self.noa:, self.noa:], x['ab'])
        if self.reference != 'rhf':
            Ax['aa'] += contract('bc,ijac->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['aa'] -= contract('ac,ijbc->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['bb'] += contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['bb'])
            Ax['bb'] -= contract('ac,ijbc->ijab', self.fb[self.nob:, self.nob:], x['bb'])
        else:
            Ax['aa'] += .5*contract('bc,ijac->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['aa'] -= .5*contract('ac,ijbc->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['aa'] += .5*contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['bb'])
            Ax['aa'] -= .5*contract('ac,ijbc->ijab', self.fb[self.nob:, self.nob:], x['bb'])
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        self.fock_time += time.time()-fstart
        self.fock_no += 1
        return Ax

    def Finv(self, x):
        Fx = self.F(x)
        x2 = np.multiply(x, x)
        Finvx = np.divide(x2, Fx, out = np.zeros(len(x)), where = abs(Fx) > 1e-12)
        return Finvx

    def F2inv(self, x):
        F2x = self.F(self.F(x))
        x2 = np.multiply(x, x)
        F2invx = np.divide(x2, F2x, out = np.zeros(len(x)), where = abs(F2x) > 1e-12)
        return F2invx
 
    def shifted_A(self, x):
        x0 = copy.copy(x)
        if self.reference == 'rhf':
            selfx2 = at.rhf_to_uhf(self.x, self)

        if self.reference == 'rhf':
            x = at.rhf_to_uhf(x, self)
        
        N = self.noa + self.nob
      
        if self.shift == 'acpf':
            shift = 2*self.Ec/N
            shift = x*shift

        elif self.shift == 'aqcc':
            shift = (1-(N-3)*(N-2)/(N*(N-1)))*self.Ec
            shift = x*shift

        elif self.shift == 'ucc3':
            shift = []
            if self.reference == 'rhf':
                b2 = at.rhf_to_uhf(self.b, self)
                for i in range(0, len(x)):
                    shift.append(b2[i]*selfx2[i]*x[i])
                Ax0 = self.c3epa_A(x0)
                return Ax0 - at.uhf_to_rhf(np.array(shift), self)
            else:
                for i in range(0, len(x)):
                    shift.append(self.b[i]*self.x[i]*x0[i])
                Ax0 = self.c3epa_A(x0)
                return Ax0 - shift

        if self.s2_term == True:
            self.lam = 1
            Ax0 = self.c3epa_A(x0)
            if self.reference == 'rhf':
                #print(np.linalg.norm(at.uhf_to_rhf(at.rhf_to_uhf(Ax0, self), self) - Ax0))
                return Ax0 - at.uhf_to_rhf(shift, self)
            else:
                return Ax0-shift                
       
        elif self.s2_term == False:
            Ax0 = self.cepa_A(x0)
            if self.reference == 'rhf':       
                return Ax0 - at.uhf_to_rhf(shift, self)
            else:
                return Ax0-shift                

    def energy(self, x):
        if self.reference == 'rhf':
            b = at.rhf_to_uhf(self.b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + x.T.dot(b)
        else:   
            energy = self.hf_energy + x.T.dot(self.b)
        return energy
         
    def H(self, x):
        start = time.time()
        Hx = self.cepa_A(x)
        if self.ucc_term == True: 
            Hx += self.A2(x)
        Hx += np.multiply(self.shift, x)
        self.count_no += 1
        self.matvec_time += time.time()-start
        return Hx

    def tik_H(self, x):
        return self.H(self.H(x)) + x * self.tik_omega**2

    def cepa_A(self, x):
        if self.reference == 'rhf':
            x = at.rhf_to_uhf(x, self)
        x = at.decatenate_amps(x, self, reference = 'uhf')
        x = at.expand_vector(x, self, reference = 'uhf')

        #Initialize Hessian Shapes
        Ax = {} 
        Ax['a'] = np.zeros((self.noa, self.nva))
        Ax['b'] = np.zeros((self.nob, self.nvb))
        Ax['aa'] = np.zeros((self.noa, self.noa, self.nva, self.nva))
        Ax['ab'] = np.zeros((self.noa, self.nob, self.nva, self.nvb))
        Ax['bb'] = np.zeros((self.nob, self.nob, self.nvb, self.nvb))

        #S->S F (Particle Interaction)
        Ax['a'] += contract('ab,ib->ia', self.fa[self.noa:, self.noa:], x['a'])
        if self.reference != 'rhf':
            Ax['b'] += contract('ab,ib->ia', self.fb[self.nob:, self.nob:], x['b'])
       

        #S->S F (Hole Interaction)
        Ax['a'] -= contract('ji,ja->ia', self.fa[:self.noa, :self.noa], x['a'])
        if self.reference != 'rhf':
            Ax['b'] -= contract('ji,ja->ia', self.fb[:self.nob, :self.nob], x['b'])

        #S->S V 
        Ax['a'] += contract('jabi,jb->ia', self.l_aaaa[:self.noa, self.noa:, self.noa:, :self.noa], x['a'])
        Ax['a'] += contract('ajib,jb->ia', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['b'])
        if self.reference != 'rhf':
            Ax['b'] += contract('jabi,jb->ia', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['a'])
            Ax['b'] += contract('jabi,jb->ia', self.l_bbbb[:self.nob, self.nob:, self.nob:, :self.nob], x['b'])

        #S->D V (Hole Interaction)
        Ax['aa'] -= contract('kbij,ka->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, :self.noa], x['a'])
        Ax['aa'] += contract('kaij,kb->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, :self.noa], x['a'])
        Ax['ab'] -= contract('kbij,ka->ijab', self.j_abab[:self.noa, self.nob:, :self.noa, :self.nob], x['a'])
        Ax['ab'] -= contract('akij,kb->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, :self.nob], x['b'])
        if self.reference != 'rhf':
            Ax['bb'] -= contract('kbij,ka->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, :self.nob], x['b'])
            Ax['bb'] += contract('kaij,kb->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, :self.nob], x['b'])

        #S->D V (Particle Interaction)
        Ax['aa'] += contract('abcj,ic->ijab', self.l_aaaa[self.noa:, self.noa:, self.noa:, :self.noa], x['a'])
        Ax['aa'] -= contract('abci,jc->ijab', self.l_aaaa[self.noa:, self.noa:, self.noa:, :self.noa], x['a'])
        Ax['ab'] += contract('abcj,ic->ijab', self.j_abab[self.noa:, self.nob:, self.noa:, :self.nob], x['a'])
        Ax['ab'] += contract('abic,jc->ijab', self.j_abab[self.noa:, self.nob:, :self.noa, self.nob:], x['b'])
        if self.reference != 'rhf':
            Ax['bb'] += contract('abcj,ic->ijab', self.l_bbbb[self.nob:, self.nob:, self.nob:, :self.nob], x['b'])
            Ax['bb'] -= contract('abci,jc->ijab', self.l_bbbb[self.nob:, self.nob:, self.nob:, :self.nob], x['b'])

        
        #D->S V (Hole Interaction)
        rt2 = np.sqrt(2)

        if self.reference != 'rhf':
            Ax['a'] -= .5*contract('kjib,kjab->ia', self.l_aaaa[:self.noa, :self.noa, :self.noa, self.noa:], x['aa']) 
            Ax['a'] -= contract('kjib,kjab->ia', self.j_abab[:self.noa, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['b'] -= .5*contract('kjib,kjab->ia', self.l_bbbb[:self.nob, :self.nob, :self.nob, self.nob:], x['bb'])
            Ax['b'] -= contract('jkbi,jkba->ia', self.j_abab[:self.noa, :self.nob, self.noa:, :self.nob], x['ab'])
        else:
            Ax['a'] -= 1/2 * .5*contract('kjib,kjab->ia', self.l_aaaa[:self.noa, :self.noa, :self.noa, self.noa:], x['aa']) 
            Ax['a'] -= 1/2 * contract('kjib,kjab->ia', self.j_abab[:self.noa, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['a'] -= 1/2 * .5*contract('kjib,kjab->ia', self.l_bbbb[:self.nob, :self.nob, :self.nob, self.nob:], x['bb'])
            Ax['a'] -= 1/2 * contract('jkbi,jkba->ia', self.j_abab[:self.noa, :self.nob, self.noa:, :self.nob], x['ab'])
 
        #D->S V (Particle Interaction)

        if self.reference != 'rhf':
            Ax['a'] += .5*contract('ajcb,ijcb->ia', self.l_aaaa[self.noa:, :self.noa, self.noa:, self.noa:], x['aa'])
            Ax['a'] += contract('ajcb,ijcb->ia', self.j_abab[self.noa:, :self.nob, self.noa:, self.nob:], x['ab'])
            Ax['b'] += .5*contract('ajcb,ijcb->ia', self.l_bbbb[self.nob:, :self.nob, self.nob:, self.nob:], x['bb'])
            Ax['b'] += contract('jabc,jibc->ia', self.j_abab[:self.noa, self.nob:, self.noa:, self.nob:], x['ab'])
        else:
            Ax['a'] += 1/2 * .5*contract('ajcb,ijcb->ia', self.l_aaaa[self.noa:, :self.noa, self.noa:, self.noa:], x['aa'])
            Ax['a'] += 1/2 * contract('ajcb,ijcb->ia', self.j_abab[self.noa:, :self.nob, self.noa:, self.nob:], x['ab'])
            Ax['a'] += 1/2 * .5*contract('ajcb,ijcb->ia', self.l_bbbb[self.nob:, :self.nob, self.nob:, self.nob:], x['bb'])
            Ax['a'] += 1/2 * contract('jabc,jibc->ia', self.j_abab[:self.noa, self.nob:, self.noa:, self.nob:], x['ab'])
            
 
        #D->D F (Hole Interaction)

        Ax['ab'] -= contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['ab'])
        Ax['ab'] -= contract('ki,kjab->ijab', self.fa[:self.noa, :self.noa], x['ab'])
        if self.reference != 'rhf':     
            Ax['aa'] -= (contract('kj,ikab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['aa'] += (contract('ki,jkab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['bb'] -= (contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            Ax['bb'] += (contract('ki,jkab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
        else:
            Ax['aa'] -= .5*(contract('kj,ikab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['aa'] += .5*(contract('ki,jkab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
            Ax['aa'] -= .5*(contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            Ax['aa'] += .5*(contract('ki,jkab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            
        #D->D F (Particle Interaction)

        Ax['ab'] += contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['ab'])
        Ax['ab'] += contract('ac,ijcb->ijab', self.fa[self.noa:, self.noa:], x['ab'])
        if self.reference != 'rhf':
            Ax['aa'] += contract('bc,ijac->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['aa'] -= contract('ac,ijbc->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['bb'] += contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['bb'])
            Ax['bb'] -= contract('ac,ijbc->ijab', self.fb[self.nob:, self.nob:], x['bb'])
        else:
            Ax['aa'] += .5*contract('bc,ijac->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['aa'] -= .5*contract('ac,ijbc->ijab', self.fa[self.noa:, self.noa:], x['aa'])
            Ax['aa'] += .5*contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['bb'])
            Ax['aa'] -= .5*contract('ac,ijbc->ijab', self.fb[self.nob:, self.nob:], x['bb'])

        #D->D V (Particle, Particle Interaction)

        Ax['aa'] += .5*contract('abcd,ijcd->ijab', self.l_aaaa[self.noa:, self.noa:, self.noa:, self.noa:], x['aa'])
        Ax['ab'] += .5*contract('abcd,ijcd->ijab', self.j_abab[self.noa:, self.nob:, self.noa:, self.nob:], x['ab'])
        Ax['ab'] += .5*contract('abdc,ijdc->ijab', self.j_abab[self.noa:, self.nob:, self.noa:, self.nob:], x['ab'])
        if self.reference != 'rhf':
            Ax['bb'] += .5*contract('abcd,ijcd->ijab', self.l_bbbb[self.nob:, self.nob:, self.nob:, self.nob:], x['bb'])

        #D->D V (Hole, Hole Interaction)

        Ax['aa'] += .5*contract('ijkl,klab->ijab', self.l_aaaa[:self.noa, :self.noa, :self.noa, :self.noa], x['aa'])
        Ax['ab'] += .5*contract('ijkl,klab->ijab', self.j_abab[:self.noa, :self.nob, :self.noa, :self.nob], x['ab'])
        Ax['ab'] += .5*contract('ijlk,lkab->ijab', self.j_abab[:self.noa, :self.nob, :self.noa, :self.nob], x['ab'])
        if self.reference !=  'rhf':
            Ax['bb'] += .5*contract('ijkl,klab->ijab', self.l_bbbb[:self.nob, :self.nob, :self.nob, :self.nob], x['bb'])

        #D->D V (Hole, Particle Interaction)

        Ax['ab'] -= contract('akcj,ikcb->ijab', self.j_abab[self.noa:, :self.nob, self.noa:, :self.nob], x['ab'])
        Ax['ab'] -= contract('kbcj,ikca->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['aa'])
        Ax['ab'] += contract('akic,kjcb->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['bb'])
        Ax['ab'] -= contract('kbic,kjac->ijab', self.j_abab[:self.noa, self.nob:, :self.noa, self.nob:], x['ab'])
        Ax['ab'] += contract('kbcj,ikac->ijab', self.l_bbbb[:self.nob, self.nob:, self.nob:, :self.nob], x['ab'])
        Ax['ab'] -= contract('akci,kjcb->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['ab'])
        if self.reference != 'rhf':
            Ax['aa'] -= contract('akcj,ikcb->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])
            Ax['aa'] += contract('bkcj,ikca->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])
            Ax['aa'] += contract('akci,jkcb->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])
            Ax['aa'] -= contract('bkci,jkca->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])

            Ax['aa'] -= contract('akjc,ikbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['aa'] += contract('bkjc,ikac->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['aa'] += contract('akic,jkbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['aa'] -= contract('bkic,jkac->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
        
            Ax['bb'] -= contract('akcj,ikcb->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['bb'] += contract('bkcj,ikca->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['bb'] += contract('akci,jkcb->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['bb'] -= contract('bkci,jkca->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])

            Ax['bb'] -= contract('kacj,kicb->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['bb'] += contract('kbcj,kica->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['bb'] += contract('kaci,kjcb->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['bb'] -= contract('kbci,kjca->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
        else:
            Ax['aa'] -= .5*contract('akcj,ikcb->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])
            Ax['aa'] += .5*contract('bkcj,ikca->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])
            Ax['aa'] += .5*contract('akci,jkcb->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])
            Ax['aa'] -= .5*contract('bkci,jkca->ijab', self.l_aaaa[self.noa:, :self.noa, self.noa:, :self.noa], x['aa'])

            Ax['aa'] -= .5*contract('akjc,ikbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['aa'] += .5*contract('bkjc,ikac->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['aa'] += .5*contract('akic,jkbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
            Ax['aa'] -= .5*contract('bkic,jkac->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
        
            Ax['aa'] -= .5*contract('akcj,ikcb->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['aa'] += .5*contract('bkcj,ikca->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['aa'] += .5*contract('akci,jkcb->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['aa'] -= .5*contract('bkci,jkca->ijab', self.l_bbbb[self.nob:, :self.nob, self.nob:, :self.nob], x['bb'])

            Ax['aa'] -= .5*contract('kacj,kicb->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['aa'] += .5*contract('kbcj,kica->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['aa'] += .5*contract('kaci,kjcb->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['aa'] -= .5*contract('kbci,kjca->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
           
  
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        return Ax

    def A2(self, x):
        if self.reference == 'rhf':
            x = at.rhf_to_uhf(x, self)
        x = at.decatenate_amps(x, self, reference = 'uhf')
        x = at.expand_vector(x, self, reference = 'uhf')

        #Initialize Hessian Shapes
        Ax = {} 
        Ax['a'] = np.zeros((self.noa, self.nva))
        Ax['b'] = np.zeros((self.nob, self.nvb))
        Ax['aa'] = np.zeros((self.noa, self.noa, self.nva, self.nva))
        Ax['ab'] = np.zeros((self.noa, self.nob, self.nva, self.nvb))
        Ax['bb'] = np.zeros((self.nob, self.nob, self.nvb, self.nvb))
        
        if self.reference != 'rhf':
            Ax['a'] += contract('ijab,jb->ia', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:], x['a'])
            Ax['a'] += contract('ijab,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['b'])
            Ax['b'] += contract('ijab,jb->ia', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:], x['b'])
            Ax['b'] += contract('jiba,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['a'])
        else:
            Ax['a'] += .5*contract('ijab,jb->ia', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:], x['a'])
            Ax['a'] += .5*contract('ijab,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['b'])
            Ax['a'] += .5*contract('ijab,jb->ia', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:], x['b'])
            Ax['a'] += .5*contract('jiba,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['a'])
 
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        return Ax

    def cepa_b(self):
        grad = {} 
        grad['a'] = np.zeros((self.noa,self.nva))
        grad['b'] = np.zeros((self.nob,self.nvb))
        grad['aa'] = contract('ijab->ijab', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:])
        grad['ab'] = contract('ijab->ijab', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:])
        grad['bb'] = contract('ijab->ijab', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:])
        return grad
    

