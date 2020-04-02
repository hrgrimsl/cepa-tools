import numpy as np
from opt_einsum import contract
import copy

import tensortools as tt
import algebratools as at
import copy
import scipy
import os

class molecule:
    def __init__(self, geometry, basis, reference, **kwargs):
        import psi4
        self.optimize = False
        self.scf = False
        self.mp2 = False
        self.ccsd = False
        self.ccsdpt = False
        self.cepa0 = False
        self.shucc = False
        self.lam_cepa = None
        self.lccd = False
        self.acpf = False
        self.uacpf = False
        self.aqcc = False
        self.uaqcc = False
        self.run_ic3epa = False
        self.tol = 1e-12
        for key, value in kwargs.items():
            setattr(self, key, value)
        try:
            assert reference == 'rhf' or reference == 'uhf'
        except:
            print("Only \"rhf\" or \"uhf\" allowed.")
            exit()
        self.reference = reference

        psi4.geometry(geometry)
        psi4.core.be_quiet()

        psi4.set_memory('8GB')
        if self.optimize == True:
            psi4.set_options({'reference': self.reference, 'scf_type': 'pk', 'g_convergence': 'GAU_TIGHT', 'd_convergence': 1e-12})
            psi4.set_options({'opt_coordinates': 'both', 'geom_maxiter': 300})
            psi4.optimize('b3lyp/6-311G(d,p)')
 
        psi4.set_options({'reference': reference, 'basis': basis, 'd_convergence': 1e-12, 'scf_type': 'pk'})
        self.hf_energy, wfn = psi4.energy('scf', return_wfn = True)
        if self.scf == True:
            print("HF energy:".ljust(30)+("{0:20.16f}".format(self.hf_energy))) 
        if self.mp2 == True:
            print("MP2 energy:".ljust(30)+("{0:20.16f}".format(psi4.energy('mp2'))))   
        if self.ccsd == True:
            print("CCSD energy:".ljust(30)+("{0:20.16f}".format(psi4.energy('mp2'))))   
        if self.ccsdpt == True:
            print("CCSD(T) energy:".ljust(30)+("{0:20.16f}".format(psi4.energy('ccsd(t)'))))   
        if self.lccd == True:
            print("LCCD energy:".ljust(30)+("{0:20.16f}".format(psi4.energy('lccd'))))   
        if self.acpf == True:
            print("ACPF energy:".ljust(30)+("{0:20.16f}".format(psi4.energy('acpf'))))   
        if self.aqcc == True:
            print("AQCC energy:".ljust(30)+("{0:20.16f}".format(psi4.energy('aqcc'))))   

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
        if self.cepa0 == True:
            cepa = self.cepa()
            print("CEPA(0) energy:".ljust(30)+("{0:20.16f}".format(cepa)))   
        if self.shucc == True:
            self.lam = 1
            c3epa = self.c3epa()
            print("SHUCC energy:".ljust(30)+("{0:20.16f}".format(c3epa)))   
        if self.lam_cepa != None:
            self.lam = self.lam_cepa
            c3epa = self.c3epa()
            print("λ-C3EPA energy:".ljust(30)+("{0:20.16f}".format(c3epa)))   
        if self.run_ic3epa == True:
            self.lam = 0
            c3epa = self.ic3epa()
            print("IC3EPA energy:".ljust(30)+("{0:20.16f}".format(c3epa)))   
        if self.uacpf == True:
            self.s2_term = True
            self.shift = 'acpf'            
            uacpf = self.shifted_cepa()
            print("UACPF energy:".ljust(30)+("{0:20.16f}".format(uacpf)))   
        if self.uaqcc == True:
            self.s2_term = True
            self.shift = 'aqcc'            
            uaqcc = self.shifted_cepa()
            print("UAQCC energy:".ljust(30)+("{0:20.16f}".format(uaqcc)))   


    def cepa(self):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        Aop = scipy.sparse.linalg.LinearOperator((len(b), len(b)), matvec = self.cepa_A, rmatvec = self.cepa_A)
        x, info = np.array(scipy.sparse.linalg.cg(Aop, -b, tol = self.tol))
        if self.reference == 'rhf':
            Ax = at.rhf_to_uhf(self.cepa_A(x), self)
            b = at.rhf_to_uhf(b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(Ax)
        else:   
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.cepa_A(x)) 
        return energy

    def shifted_cepa(self):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        norm = 1
        old = b*0
        self.Ec = 0
        x0 = 0*b
        while abs(norm)>1e-14:
             
            Aop = scipy.sparse.linalg.LinearOperator((len(b), len(b)), matvec = self.shifted_A, rmatvec = self.shifted_A,)
            x, info = np.array(scipy.sparse.linalg.cg(Aop, -b, x0 = x0, tol = self.tol))
            norm = (old-x).dot(old-x)
            old = copy.copy(x)
            x0 = x
            if self.reference == 'rhf':
                Ax2 = at.rhf_to_uhf(self.shifted_A(x), self)
                b2 = at.rhf_to_uhf(b, self)
                x2 = at.rhf_to_uhf(x, self)
                energy = self.hf_energy + 2*x2.T.dot(b2) + x2.T.dot(Ax2)
            else:   
                energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.shifted_A(x))        
            self.Ec = energy-self.hf_energy
        return energy

    def c3epa(self, **kwargs):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        Aop = scipy.sparse.linalg.LinearOperator((len(b), len(b)), matvec = self.c3epa_A, rmatvec = self.c3epa_A)
        x, info = np.array(scipy.sparse.linalg.cg(Aop, -b, tol = self.tol))
        #x = self.cg_shucc(-b)
        if self.reference == 'rhf':
            Ax = at.rhf_to_uhf(self.c3epa_A(x), self)
            b = at.rhf_to_uhf(b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(Ax)
        else:   
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.c3epa_A(x)) 
        return energy
    
    def ic3epa(self, **kwargs):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        oldguess = np.zeros(len(b))
        norm = 1
        self.lam = 0
        self.x = 0*b
        while abs(norm) > 1e-12:
            Aop = scipy.sparse.linalg.LinearOperator((len(b), len(b)), matvec = self.ic3epa_A, rmatvec = self.ic3epa_A)
            x, info = np.array(scipy.sparse.linalg.cg(Aop, -b, x0 = self.x, tol = self.tol))
            self.ic3epa_energy(x)
            norm = (oldguess - x).dot(oldguess - x)
            oldguess = copy.copy(x)
            self.lam = .5*x.dot(x)

        if self.reference == 'rhf':
            Ax = at.rhf_to_uhf(self.ic3epa_A(x), self)
            b = at.rhf_to_uhf(b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(Ax)
        else:   
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.ic3epa_A(x)) 
        return energy

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
        Ax['a'] -= .5*contract('jkbi,jkba->ia', self.l_aaaa[:self.noa, :self.noa, self.noa:, :self.noa], x['aa']) 
        Ax['a'] -= contract('kjib,kjab->ia', self.j_abab[:self.noa, :self.nob, :self.noa, self.nob:], x['ab'])
        if self.reference != 'rhf':
            Ax['b'] -= .5*contract('jkbi,jkba->ia', self.l_bbbb[:self.nob, :self.nob, self.nob:, :self.nob], x['bb'])
            Ax['b'] -= contract('jkbi,jkba->ia', self.j_abab[:self.noa, :self.nob, self.noa:, :self.nob], x['ab'])
         
        #D->S V (Particle Interaction)
        Ax['a'] += .5*contract('jabc,jibc->ia', self.l_aaaa[:self.noa, self.noa:, self.noa:, self.noa:], x['aa'])
        Ax['a'] += contract('ajcb,ijcb->ia', self.j_abab[self.noa:, :self.nob, self.noa:, self.nob:], x['ab'])
        if self.reference != 'rhf':
            Ax['b'] += .5*contract('jabc,jibc->ia', self.l_bbbb[:self.nob, self.nob:, self.nob:, self.nob:], x['bb'])
            Ax['b'] += contract('jabc,jibc->ia', self.j_abab[:self.noa, self.nob:, self.noa:, self.nob:], x['ab'])
               
  
        #D->D F (Hole Interaction)
        Ax['aa'] -= (contract('kj,ikab->ijab', self.fa[:self.noa, :self.noa], x['aa']))
        Ax['aa'] += (contract('ki,jkab->ijab', self.fa[:self.noa, :self.noa], x['aa']))

        Ax['ab'] -= contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['ab'])
        Ax['ab'] -= contract('ki,kjab->ijab', self.fa[:self.noa, :self.noa], x['ab'])

        if self.reference != 'rhf':
            Ax['bb'] -= (contract('kj,ikab->ijab', self.fb[:self.nob, :self.nob], x['bb']))
            Ax['bb'] += (contract('ki,jkab->ijab', self.fb[:self.nob, :self.nob], x['bb']))

        #D->D F (Particle Interaction)
        Ax['aa'] += contract('bc,ijac->ijab', self.fa[self.noa:, self.noa:], x['aa'])
        Ax['aa'] -= contract('ac,ijbc->ijab', self.fa[self.noa:, self.noa:], x['aa'])
        Ax['ab'] += contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['ab'])
        Ax['ab'] += contract('ac,ijcb->ijab', self.fa[self.noa:, self.noa:], x['ab'])
        if self.reference != 'rhf':
            Ax['bb'] += contract('bc,ijac->ijab', self.fb[self.nob:, self.nob:], x['bb'])
            Ax['bb'] -= contract('ac,ijbc->ijab', self.fb[self.nob:, self.nob:], x['bb'])

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
        Ax['aa'] -= contract('kbic,kjac->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, self.noa:], x['aa'])
        Ax['aa'] += contract('kaic,kjbc->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, self.noa:], x['aa'])
        Ax['aa'] += contract('kbjc,kiac->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, self.noa:], x['aa'])
        Ax['aa'] -= contract('kajc,kibc->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, self.noa:], x['aa'])
        Ax['aa'] -= contract('bkic,jkac->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
        Ax['aa'] += contract('akic,jkbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
        Ax['aa'] += contract('bkjc,ikac->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])
        Ax['aa'] -= contract('akjc,ikbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['ab'])

        Ax['ab'] -= contract('kbic,kjac->ijab', self.j_abab[:self.noa, self.nob:, :self.noa, self.nob:], x['ab'])
        Ax['ab'] -= contract('akcj,ikcb->ijab', self.j_abab[self.noa:, :self.nob, self.noa:, :self.nob], x['ab'])    
        Ax['ab'] -= contract('kaic,kjcb->ijab', self.l_aaaa[:self.noa, self.noa:, :self.noa, self.noa:], x['ab'])
        Ax['ab'] -= contract('kbjc,ikac->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, self.nob:], x['ab'])
        Ax['ab'] -= contract('kbcj,kiac->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['aa'])
        Ax['ab'] -= contract('akic,kjbc->ijab', self.j_abab[self.noa:, :self.nob, :self.noa, self.nob:], x['bb'])

        if self.reference != 'rhf': 
            Ax['bb'] -= contract('kbic,kjac->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, self.nob:], x['bb'])
            Ax['bb'] += contract('kaic,kjbc->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, self.nob:], x['bb'])
            Ax['bb'] += contract('kbjc,kiac->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, self.nob:], x['bb'])
            Ax['bb'] -= contract('kajc,kibc->ijab', self.l_bbbb[:self.nob, self.nob:, :self.nob, self.nob:], x['bb'])

            Ax['bb'] -= contract('kbci,kjca->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['bb'] += contract('kaci,kjcb->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['bb'] += contract('kbcj,kica->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
            Ax['bb'] -= contract('kacj,kicb->ijab', self.j_abab[:self.noa, self.nob:, self.noa:, :self.nob], x['ab'])
      
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        return Ax
 
    def shifted_A(self, x):
        N = self.noa + self.nob
        if self.shift == 'acpf':
            shift = 2*self.Ec/N
        elif self.shift == 'aqcc':
            shift = (1-(N-3)*(N-2)/(N*(N-1)))*self.Ec          
        if self.s2_term == True:
            self.lam = 1
            Ax0 = self.c3epa_A(x)        
        elif self.s2_term == False:
            Ax0 = self.cepa_A(x)
        return Ax0 - shift*x 
 
    def c3epa_A(self, x):
        A0 = self.cepa_A(x)
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
        
        Ax['a'] += contract('ijab,jb->ia', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:], x['a'])
        Ax['a'] += contract('ijab,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['b'])
        if self.reference != 'rhf':
            Ax['b'] += contract('ijab,jb->ia', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:], x['b'])
            Ax['b'] += contract('jiba,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['a'])
        if self.reference == 'rhf':
            Ax['b'] = Ax['a'] 
        
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        return (self.lam*Ax)+A0

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
        
        Ax['a'] += contract('ijab,jb->ia', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:], x['a'])
        Ax['a'] += contract('ijab,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['b'])
        if self.reference != 'rhf':
            Ax['b'] += contract('ijab,jb->ia', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:], x['b'])
            Ax['b'] += contract('jiba,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['a'])
        if self.reference == 'rhf':
            Ax['b'] = Ax['a'] 
        
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        return Ax


    def ic3epa_A(self, x):
        term1 = self.cepa_A(x)
        term3 = self.x.T.dot(self.A2(self.x))*x

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
        
        Ax['a'] += contract('ijab,jb->ia', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:], x['a'])
        Ax['a'] += contract('ijab,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['b'])
        if self.reference != 'rhf':
            Ax['b'] += contract('ijab,jb->ia', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:], x['b'])
            Ax['b'] += contract('jiba,jb->ia', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:], x['a'])
        if self.reference == 'rhf':
            Ax['b'] = Ax['a'] 
        
        Ax = at.collapse_tensor(Ax, self)
        Ax = at.concatenate_amps(Ax, self)
        return term1+term3+2*self.lam*Ax





    def cepa_b(self):
        grad = {} 
        grad['a'] = np.zeros((self.noa,self.nva))
        grad['b'] = np.zeros((self.nob,self.nvb))
        grad['aa'] = contract('ijab->ijab', self.l_aaaa[:self.noa, :self.noa, self.noa:, self.noa:])
        grad['ab'] = contract('ijab->ijab', self.j_abab[:self.noa, :self.nob, self.noa:, self.nob:])
        grad['bb'] = contract('ijab->ijab', self.l_bbbb[:self.nob, :self.nob, self.nob:, self.nob:])
        return grad
    
    def cepa_energy(self, x):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        energy = self.hf_energy + 2* x.T.dot(b) + x.T.dot(self.cepa_A(x))
        if self.reference == 'rhf':
            Ax = at.rhf_to_uhf(self.cepa_A(x), self)
            b = at.rhf_to_uhf(b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(Ax)
        else:   
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.cepa_A(x))
        print(energy)
        return energy
        
    def shifted_energy(self, x):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        energy = self.hf_energy + 2* x.T.dot(b) + x.T.dot(self.shifted_A(x))
        if self.reference == 'rhf':
            Ax = at.rhf_to_uhf(self.shifted_A(x), self)
            b = at.rhf_to_uhf(b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(Ax)
        else:   
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.shifted_A(x))
        print(energy)
        return energy
    
    def ic3epa_energy(self, x):
        b = self.cepa_b()
        b = at.collapse_tensor(b, self)
        b = at.concatenate_amps(b, self)
        energy = self.hf_energy + 2* x.T.dot(b) + x.T.dot(self.ic3epa_A(x))
        if self.reference == 'rhf':
            Ax = at.rhf_to_uhf(self.ic3epa_A(x), self)
            b = at.rhf_to_uhf(b, self)
            x = at.rhf_to_uhf(x, self)
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(Ax)
        else:
            energy = self.hf_energy + 2*x.T.dot(b) + x.T.dot(self.ic3epa_A(x))
        print(energy)
        return energy

    def cg_shucc(self, b):
        x = 0*b
        r = b
        p = copy.copy(r)
        self.lam = 1
        while 1 == 1:
            Ap = self.c3epa_A(p)
            alpha = r.T.dot(r)/p.T.dot(Ap)
            x = x + alpha*p
            r2 = r - alpha*Ap
            if np.linalg.norm(r) <= 1e-7:
                break
            beta = r2.T.dot(r2)/(r.T.dot(r))
            r = r2
            p = r + beta*p

        return x 