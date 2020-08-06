
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit import Aer as aer

import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as optz

np.set_printoptions(linewidth=150)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

pl.ion()


U = [[-0.21338835+0.33838835j, -0.14016504-0.08838835j,  0.21338835-0.08838835j,
   0.03661165+0.08838835j,  0.08838835-0.03661165j, -0.08838835-0.21338835j,
  -0.08838835+0.14016504j,  0.33838835+0.21338835j,  0.21338835-0.08838835j,
   0.03661165+0.08838835j,  0.39016504+0.08838835j, -0.03661165+0.16161165j,
   0.16161165+0.03661165j,  0.08838835-0.39016504j,  0.08838835-0.03661165j,
  -0.08838835-0.21338835j],
 [-0.14016504-0.08838835j, -0.21338835+0.33838835j,  0.03661165+0.08838835j,
   0.21338835-0.08838835j, -0.08838835-0.21338835j,  0.08838835-0.03661165j,
   0.33838835+0.21338835j, -0.08838835+0.14016504j,  0.03661165+0.08838835j,
   0.21338835-0.08838835j, -0.03661165+0.16161165j,  0.39016504+0.08838835j,
   0.08838835-0.39016504j,  0.16161165+0.03661165j, -0.08838835-0.21338835j,
   0.08838835-0.03661165j],
 [ 0.21338835-0.08838835j,  0.03661165+0.08838835j, -0.21338835+0.33838835j,
  -0.14016504-0.08838835j, -0.08838835+0.14016504j,  0.33838835+0.21338835j,
   0.08838835-0.03661165j, -0.08838835-0.21338835j,  0.39016504+0.08838835j,
  -0.03661165+0.16161165j,  0.21338835-0.08838835j,  0.03661165+0.08838835j,
   0.08838835-0.03661165j, -0.08838835-0.21338835j,  0.16161165+0.03661165j,
   0.08838835-0.39016504j],
 [ 0.03661165+0.08838835j,  0.21338835-0.08838835j, -0.14016504-0.08838835j,
  -0.21338835+0.33838835j,  0.33838835+0.21338835j, -0.08838835+0.14016504j,
  -0.08838835-0.21338835j,  0.08838835-0.03661165j, -0.03661165+0.16161165j,
   0.39016504+0.08838835j,  0.03661165+0.08838835j,  0.21338835-0.08838835j,
  -0.08838835-0.21338835j,  0.08838835-0.03661165j,  0.08838835-0.39016504j,
   0.16161165+0.03661165j],
 [ 0.08838835-0.03661165j, -0.08838835-0.21338835j, -0.08838835+0.14016504j,
   0.33838835+0.21338835j, -0.21338835+0.33838835j, -0.14016504-0.08838835j,
   0.21338835-0.08838835j,  0.03661165+0.08838835j,  0.16161165+0.03661165j,
   0.08838835-0.39016504j,  0.08838835-0.03661165j, -0.08838835-0.21338835j,
   0.21338835-0.08838835j,  0.03661165+0.08838835j,  0.39016504+0.08838835j,
  -0.03661165+0.16161165j],
 [-0.08838835-0.21338835j,  0.08838835-0.03661165j,  0.33838835+0.21338835j,
  -0.08838835+0.14016504j, -0.14016504-0.08838835j, -0.21338835+0.33838835j,
   0.03661165+0.08838835j,  0.21338835-0.08838835j,  0.08838835-0.39016504j,
   0.16161165+0.03661165j, -0.08838835-0.21338835j,  0.08838835-0.03661165j,
   0.03661165+0.08838835j,  0.21338835-0.08838835j, -0.03661165+0.16161165j,
   0.39016504+0.08838835j],
 [-0.08838835+0.14016504j,  0.33838835+0.21338835j,  0.08838835-0.03661165j,
  -0.08838835-0.21338835j,  0.21338835-0.08838835j,  0.03661165+0.08838835j,
  -0.21338835+0.33838835j, -0.14016504-0.08838835j,  0.08838835-0.03661165j,
  -0.08838835-0.21338835j,  0.16161165+0.03661165j,  0.08838835-0.39016504j,
   0.39016504+0.08838835j, -0.03661165+0.16161165j,  0.21338835-0.08838835j,
   0.03661165+0.08838835j],
 [ 0.33838835+0.21338835j, -0.08838835+0.14016504j, -0.08838835-0.21338835j,
   0.08838835-0.03661165j,  0.03661165+0.08838835j,  0.21338835-0.08838835j,
  -0.14016504-0.08838835j, -0.21338835+0.33838835j, -0.08838835-0.21338835j,
   0.08838835-0.03661165j,  0.08838835-0.39016504j,  0.16161165+0.03661165j,
  -0.03661165+0.16161165j,  0.39016504+0.08838835j,  0.03661165+0.08838835j,
   0.21338835-0.08838835j],
 [ 0.21338835-0.08838835j,  0.03661165+0.08838835j,  0.39016504+0.08838835j,
  -0.03661165+0.16161165j,  0.16161165+0.03661165j,  0.08838835-0.39016504j,
   0.08838835-0.03661165j, -0.08838835-0.21338835j, -0.21338835+0.33838835j,
  -0.14016504-0.08838835j,  0.21338835-0.08838835j,  0.03661165+0.08838835j,
   0.08838835-0.03661165j, -0.08838835-0.21338835j, -0.08838835+0.14016504j,
   0.33838835+0.21338835j],
 [ 0.03661165+0.08838835j,  0.21338835-0.08838835j, -0.03661165+0.16161165j,
   0.39016504+0.08838835j,  0.08838835-0.39016504j,  0.16161165+0.03661165j,
  -0.08838835-0.21338835j,  0.08838835-0.03661165j, -0.14016504-0.08838835j,
  -0.21338835+0.33838835j,  0.03661165+0.08838835j,  0.21338835-0.08838835j,
  -0.08838835-0.21338835j,  0.08838835-0.03661165j,  0.33838835+0.21338835j,
  -0.08838835+0.14016504j],
 [ 0.39016504+0.08838835j, -0.03661165+0.16161165j,  0.21338835-0.08838835j,
   0.03661165+0.08838835j,  0.08838835-0.03661165j, -0.08838835-0.21338835j,
   0.16161165+0.03661165j,  0.08838835-0.39016504j,  0.21338835-0.08838835j,
   0.03661165+0.08838835j, -0.21338835+0.33838835j, -0.14016504-0.08838835j,
  -0.08838835+0.14016504j,  0.33838835+0.21338835j,  0.08838835-0.03661165j,
  -0.08838835-0.21338835j],
 [-0.03661165+0.16161165j,  0.39016504+0.08838835j,  0.03661165+0.08838835j,
   0.21338835-0.08838835j, -0.08838835-0.21338835j,  0.08838835-0.03661165j,
   0.08838835-0.39016504j,  0.16161165+0.03661165j,  0.03661165+0.08838835j,
   0.21338835-0.08838835j, -0.14016504-0.08838835j, -0.21338835+0.33838835j,
   0.33838835+0.21338835j, -0.08838835+0.14016504j, -0.08838835-0.21338835j,
   0.08838835-0.03661165j],
 [ 0.16161165+0.03661165j,  0.08838835-0.39016504j,  0.08838835-0.03661165j,
  -0.08838835-0.21338835j,  0.21338835-0.08838835j,  0.03661165+0.08838835j,
   0.39016504+0.08838835j, -0.03661165+0.16161165j,  0.08838835-0.03661165j,
  -0.08838835-0.21338835j, -0.08838835+0.14016504j,  0.33838835+0.21338835j,
  -0.21338835+0.33838835j, -0.14016504-0.08838835j,  0.21338835-0.08838835j,
   0.03661165+0.08838835j],
 [ 0.08838835-0.39016504j,  0.16161165+0.03661165j, -0.08838835-0.21338835j,
   0.08838835-0.03661165j,  0.03661165+0.08838835j,  0.21338835-0.08838835j,
  -0.03661165+0.16161165j,  0.39016504+0.08838835j, -0.08838835-0.21338835j,
   0.08838835-0.03661165j,  0.33838835+0.21338835j, -0.08838835+0.14016504j,
  -0.14016504-0.08838835j, -0.21338835+0.33838835j,  0.03661165+0.08838835j,
   0.21338835-0.08838835j],
 [ 0.08838835-0.03661165j, -0.08838835-0.21338835j,  0.16161165+0.03661165j,
   0.08838835-0.39016504j,  0.39016504+0.08838835j, -0.03661165+0.16161165j,
   0.21338835-0.08838835j,  0.03661165+0.08838835j, -0.08838835+0.14016504j,
   0.33838835+0.21338835j,  0.08838835-0.03661165j, -0.08838835-0.21338835j,
   0.21338835-0.08838835j,  0.03661165+0.08838835j, -0.21338835+0.33838835j,
  -0.14016504-0.08838835j],
 [-0.08838835-0.21338835j,  0.08838835-0.03661165j,  0.08838835-0.39016504j,
   0.16161165+0.03661165j, -0.03661165+0.16161165j,  0.39016504+0.08838835j,
   0.03661165+0.08838835j,  0.21338835-0.08838835j,  0.33838835+0.21338835j,
  -0.08838835+0.14016504j, -0.08838835-0.21338835j,  0.08838835-0.03661165j,
   0.03661165+0.08838835j,  0.21338835-0.08838835j, -0.14016504-0.08838835j,
  -0.21338835+0.33838835j]]


def trunc(x, ndig=3) :
    return np.around(10**ndig*x)/10**ndig

# the overall unitary we're trying to match
uu = np.array(U)
u22 = uu[0:2,0:2]

# shortcut for pi
pi = np.pi


# the unitary resulting from the circuit
#uc = np.identity(16)

# identity on 1 qubit
II = np.identity(2)


# cnot matrix on adjacent qubits
cnot=np.zeros((4,4), dtype=np.int)
cnot[0,0]=1
cnot[1,1]=1
cnot[2,3]=1
cnot[3,2]=1

# cnot with switched ctl/target qubits
cnot2=np.zeros((4,4), dtype=np.int)
cnot2[0,0]=1
cnot2[1,3]=1
cnot2[3,1]=1
cnot2[2,2]=1

hh=np.array([[1,1],[1,-1]]) / 2**.5
hh4=np.kron(hh, np.kron(hh, np.kron(hh,hh)))


u1=np.array([[1,2],[2,1]])
u2=np.array([[30,20],[20,30]])

X = np.array([[0,1],[1,0]])


# unitariy parametrized by 3 angles
def U3(th, phi, lam) :
    
    return np.array([[np.cos(th/2),               -np.exp(1j*lam)*np.sin(th/2)],
                     [np.exp(1j*phi)*np.sin(th/2), np.exp(1j*(lam+phi))*np.cos(th/2)]] )

def usym(th) :
    
    return np.array([[np.cos(th/2),    1j*np.sin(th/2)],
                     [1j*np.sin(th/2),    np.cos(th/2)]] )


def fourgates(th1, th2, th3, th4) : 
    
    u1 = usym(th1)
    u2 = usym(th2)
    u3 = usym(th3)
    u4 = usym(th4)
    
    return np.kron(u1, np.kron(u2, np.kron(u3, u4) ) )
    

def fourgates_h(th1, th2, th3, th4) : 
    
    u1 = hh @ usym(th1)
    u2 = usym(th2)
    u3 = usym(th3)
    u4 = usym(th4)
    
    return np.kron(u1, np.kron(u2, np.kron(u3, u4) ) )


def twogates(th1, th2) :
    
    u1 = usym(th1)
    u2 = usym(th2)

    return np.kron(u1,u2)
    




def cost(mtrx) :
    
    return np.linalg.norm(mtrx, ord=2)


#def cost2(mtrx) :
    


def costvsthetas(vth) :
    
    th1=vth[0]
    th2=vth[1]
    th3=vth[2]
    th4=vth[3]
    
    mtrx = fourgates(th1, th2, th3, th4)
    
    return cost(mtrx-uu)
    


def costvsthetas_h(vth) :
    
    th1=vth[0]
    th2=vth[1]
    th3=vth[2]
    th4=vth[3]
    
    mtrx = fourgates_h(th1, th2, th3, th4)
    
    return cost(mtrx-uu)


def costvsthetas_disc(vth) :
    
    th1=np.round(vth[0])/32*pi
    th2=np.round(vth[1])/32*pi
    th3=np.round(vth[2])/32*pi
    th4=np.round(vth[3])/32*pi
    
    qc = QuantumCircuit(4)
    qc.rx(th1, 0)
    qc.rx(th2, 1)
    qc.rx(th3, 2)
    qc.rx(th4, 3)
    
    mtrx = qi.Operator(qc).data
    
    #mtrx = fourgates(th1, th2, th3, th4)
    
    return cost(mtrx-uu)



def opt4gates() :

    #res = optz.minimize(costvsthetas, x0=[pi/4, 3*pi/4, -pi/4, -3*pi/4], bounds=[(-pi,pi)]*4, method='Nelder-Meade')
 
    #res = optz.differential_evolution(costvsthetas, bounds=[(-pi,pi)]*4, popsize=500, polish=True, workers=2, strategy='rand1exp')
    
    
    # discrete version
    res = optz.differential_evolution(costvsthetas_disc, bounds=[(-32,32)]*4, popsize=500, polish=True, workers=2, strategy='rand1exp')

    #for idep in range(1) :
        #unew = np.kron(
        #        np.kron(U3(pi/2, pi/2, -pi/2), II) 
        #        , cnot)
        #uc = unew @ uc @ unew
    
    uc = fourgates(res.x[0], res.x[1], res.x[2], res.x[3])
    
    
    pl.subplot(221)
    pl.imshow(uu.real)
    pl.subplot(222)
    pl.imshow((uc/uu).real)
    pl.subplot(223)
    pl.imshow(uu.imag)
    pl.subplot(224)
    pl.imshow((uc/uu).imag)

    return res, uc


def compru(u2by2) :
    
    uc22 = u2by2
    
    u4by4 = np.concatenate( (uc22, uc22) , axis=0)
    u4by4 = np.concatenate( (u4by4,u4by4) , axis=1)

    xuc22 = X @ uc22
    
    x4by4 = np.concatenate( (xuc22, xuc22), axis=0 )
    x4by4 = np.concatenate( (x4by4, x4by4), axis=1 )
    
    u8by8 = np.concatenate( (u4by4, x4by4), axis=1 )
    u8by8 = np.concatenate( 
            (u8by8, np.concatenate((x4by4,u4by4),axis=1)), axis=0)
    
    compr = np.concatenate( 
            ( np.concatenate((u8by8,u8by8),axis=1), 
              np.concatenate((u8by8,u8by8),axis=1) )
            , axis=0)
            
    return compr

qc0 = QuantumCircuit(4)
qc0.rx(pi*.75, 0)
qc0.rx(1.047, 1)
qc0.x(1)
qc0.rx(0.668, 2)
qc0.rx(0.615, 3)
qc0.x(3)


qc1 = QuantumCircuit(4)
qc1.rx(pi*.75, 0)
qc1.rx(1.047, 1)
qc1.rx(0.668, 2)
qc1.rx(0.615, 3)

qc2 = QuantumCircuit(4)
qc2.rx(pi*3/4, 0)
qc2.rx(pi*3/8, 1)
qc2.rx(pi*5/8, 2)
qc2.rx(pi*7/8, 3)

def circ2matrix(qc=None) :

    #pl.figure() #figsize=(20,9)) 

    pl.clf()   
        
    if not qc : qc = QuantumCircuit(4)
    else :
        uc = qi.Operator(qc).data
        pl.subplot(121)
        plu(uc)
        pl.subplot(122)
        plu(uc/compru(uc[0:2,0:2]), ampl=False)

    be = aer.get_backend('unitary_simulator')
    
    ucompr = compru(u22)
    
    gatestr = input('Next gate: ')
    while gatestr != 'q' : 
        
        if gatestr == 'r' : qc = QuantumCircuit(4)

        elif gatestr[0:3]!= 'qc.' :
            try : exec(gatestr)
            except : print('ignoring previous command')

        else :
            #exec('qc.'+gatestr.strip())
            try: 
                exec(gatestr)
                goodgate=True
            except:
                goodgate=False
                print('ignoring this gate')
        
            if goodgate :

                uc = qi.Operator(qc).data
                
                uc22 = uc[0:2,0:2]
                
                u4by4 = np.concatenate( (uc22, uc22) , axis=0)
                u4by4 = np.concatenate( (u4by4,u4by4) , axis=1)

                xuc22 = X @ uc22
                
                x4by4 = np.concatenate( (xuc22, xuc22), axis=0 )
                x4by4 = np.concatenate( (x4by4, x4by4), axis=1 )
                
                u8by8 = np.concatenate( (u4by4, x4by4), axis=1 )
                u8by8 = np.concatenate( 
                        (u8by8, np.concatenate((x4by4,u4by4),axis=1)), axis=0)
                
                compr = np.concatenate( 
                        ( np.concatenate((u8by8,u8by8),axis=1), 
                          np.concatenate((u8by8,u8by8),axis=1) )
                        , axis=0)
                
                """
                pl.subplot(221)
                plu(uu)
                pl.subplot(222)
                plu(uu/ucompr)
                """
                pl.clf()
                pl.subplot(121)
                plu(uc)
                #pl.show(block=False)
                pl.subplot(122)
                plu(uc/compr, ampl=False)
                #pl.show(block=False)        
        
        gatestr = input('Next gate: ')

    #if gatestr == 'q' : pl.clf()

def plu(mat, ampl=True) :

    if ampl : pl.imshow(np.abs(mat), vmin=0, vmax=1)
    else : pl.imshow(np.abs(mat))
    
    for ii in range(mat.shape[0]) :
        for jj in range(mat.shape[1]) :
            
            if np.isfinite( np.angle(mat[ii,jj]) ) :
                pl.text(-.4+ii, .2+jj, str(
                        np.int(np.angle(mat[ii,jj])*180/pi)), 
                        color='white', size='x-small')





