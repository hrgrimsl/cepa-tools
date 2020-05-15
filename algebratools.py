import numpy as np
import copy
from opt_einsum import contract
def dict_dot(a, b, mol):
    #Works for anything dot would work on- should work for computing (x' * Hx)
    a2 = collapse_tensor(a, mol)
    b2 = collapse_tensor(b, mol)
    c = 0 
    for key in ['a', 'b', 'aa', 'ab', 'bb']:
        c += a2[key].dot(b2[key])
    try: 
        assert(abs(c-dict_dot_2(a, b, mol))<1e-10)
    except:
        print("Antisymmetry problem")

    return c

def dict_dot_2(a, b, mol):
    #Sanity check on antisymmetry of t's
    c = 0
    c += np.sum(contract('ia,ia->ia', a['a'], b['a']))
    c += np.sum(contract('ia,ia->ia', a['b'], b['b']))
    c += .25*np.sum(contract('ijab,ijab->ijab', a['aa'], b['aa']))
    c += np.sum(contract('ijab,ijab->ijab', a['ab'], b['ab']))
    c += .25*np.sum(contract('ijab,ijab->ijab', a['bb'], b['bb']))
    return c

def dict_lc(a, A, b, B):
    c = {}
    for key in ['a', 'b', 'aa', 'ab', 'bb']:
        c[key] = a*A[key]+b*B[key]
    return c 
    
def expand_vector(vec, mol, reference = None):
    if reference == None:
        reference = mol.reference

    ten = {}

    ten['a'] = np.reshape(vec['a'], (mol.noa, mol.nva))
    if reference != 'rhf':
        ten['b'] = np.reshape(vec['b'], (mol.nob, mol.nvb))

    oa = np.triu_indices(mol.noa, k = 1)
    va = np.triu_indices(mol.nva, k = 1)
    if len(list(vec['aa'])) != 0:
        ten['aa'] = np.zeros((mol.noa, mol.noa, mol.nva, mol.nva))
        aa = np.reshape(vec['aa'], ten['aa'][oa][(slice(None),)+va].shape)
        ten['aa'][oa[0][:,None],oa[1][:,None],va[0],va[1]] = aa
        ten['aa'][oa[0][:,None],oa[1][:,None],va[1],va[0]] = -aa
        ten['aa'][oa[1][:,None],oa[0][:,None],va[0],va[1]] = -aa
        ten['aa'][oa[1][:,None],oa[0][:,None],va[1],va[0]] = aa
    else:
        ten['aa'] = np.zeros((mol.noa, mol.noa, mol.nva, mol.nva))


    ten['ab'] = np.reshape(vec['ab'], (mol.noa, mol.nob, mol.nva, mol.nvb))

    ob = np.triu_indices(mol.nob, k = 1)
    vb = np.triu_indices(mol.nvb, k = 1) 
    if reference != 'rhf': 
        ten['bb'] = np.zeros((mol.nob, mol.nob, mol.nvb, mol.nvb))
        bb = np.reshape(vec['bb'], ten['bb'][ob][(slice(None),)+vb].shape)
        ten['bb'][ob[0][:,None],ob[1][:,None],vb[0],vb[1]] = bb
        ten['bb'][ob[0][:,None],ob[1][:,None],vb[1],vb[0]] = -bb
        ten['bb'][ob[1][:,None],ob[0][:,None],vb[0],vb[1]] = -bb
        ten['bb'][ob[1][:,None],ob[0][:,None],vb[1],vb[0]] = bb
        

    #   adict = np.reshape(aaaavec, (self.gaaaa[oa][(slice(None),)+va]).shape)
    #    trial['aaaa'] = 0*self.gaaaa
    #    trial['aaaa'][oa[0][:,None],oa[1][:,None],va[0],va[1]] = adict

    return ten

def collapse_tensor(ten, mol):
    vec = {}
    
    vec['a'] = np.reshape(ten['a'], mol.noa*mol.nva)

    vec['b'] = np.reshape(ten['b'], mol.nob*mol.nvb)

    oa = np.triu_indices(mol.noa, k = 1)
    va = np.triu_indices(mol.nva, k = 1)

    vec['aa'] = ten['aa'][oa][(slice(None),)+va].reshape(int(.25*mol.noa*(mol.noa-1)*mol.nva*(mol.nva-1)))

    vec['ab'] = ten['ab'].reshape(mol.noa*mol.nob*mol.nva*mol.nvb)

    ob = np.triu_indices(mol.nob, k = 1)
    vb = np.triu_indices(mol.nvb, k = 1)
   
    vec['bb'] = ten['bb'][ob][(slice(None),)+vb].reshape(int(.25*mol.nob*(mol.nob-1)*mol.nvb*(mol.nvb-1)))

    return vec

def concatenate_amps(vec, mol, reference = None):
    if reference == None:
        reference = mol.reference
    if reference != 'rhf':
        return np.concatenate((vec['a'], vec['b'], vec['aa'], vec['ab'], vec['bb']))    
    else:
        return np.concatenate((np.sqrt(2)*vec['a'], np.sqrt(2)*vec['aa'], vec['ab']))


def decatenate_amps(amps, mol, reference = None):
    if reference == None:
        reference = mol.reference
    if reference != 'rhf':
        na = mol.noa*mol.nva
        nb = na+mol.nob*mol.nvb 
        naa = nb+int(.25*mol.noa*(mol.noa-1)*mol.nva*(mol.nva-1))
        nab = naa+mol.noa*mol.nva*mol.nob*mol.nvb
        nbb = nab+int(.25*mol.nob*(mol.nob-1)*mol.nvb*(mol.nvb-1))
        vec = {}
        vec['a'] = amps[:na]
        vec['b'] = amps[na:nb]
        vec['aa'] = amps[nb:naa]
        vec['ab'] = amps[naa:nab]
        vec['bb'] = amps[nab:nbb]
    else:
        na = mol.noa*mol.nva
        naa = na+int(.25*mol.noa*(mol.noa-1)*mol.nva*(mol.nva-1))
        nab = naa+mol.noa*mol.nva*mol.nob*mol.nvb
        vec = {}
        vec['a'] = 1/np.sqrt(2)*amps[:na]
        vec['b'] = copy.copy(vec['a'])
        vec['aa'] = 1/np.sqrt(2)*amps[na:naa]
        vec['ab'] = amps[naa:nab]
        vec['bb'] = copy.copy(vec['aa'])
    return vec

def rhf_to_uhf(amps, mol):
    assert(mol.reference == 'rhf')
    vec = decatenate_amps(amps, mol)
    return concatenate_amps(vec, mol, reference = 'uhf') 

def uhf_to_rhf(amps, mol):
    assert(mol.reference == 'rhf')
    vec = decatenate_amps(amps, mol, reference = 'uhf')
    return concatenate_amps(vec, mol, reference = 'rhf')
    

def asym(ten):
    #Returns antisymmetrized (ijab-jiab-ijba+jiba)
    ten2 = -np.swapaxes(ten, 0, 1)
    ten3 = -np.swapaxes(ten, 2, 3)
    ten4 = np.swapaxes(np.swapaxes(ten, 0, 1), 2, 3)
    #assert(abs(np.linalg.norm(ten - ten2))<1e-10)
    #assert(abs(np.linalg.norm(ten2 - ten3))<1e-10)
    #assert(abs(np.linalg.norm(ten3 - ten4))<1e-10)
    return .25*(ten+ten2+ten3+ten4)
