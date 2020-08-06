

import qiskit



qc = qiskit.QuantumCircuit(1, 1)


qc.h(0)

qc.measure(0,0)

print(qc.draw())

be = qiskit.Aer.get_backend('statevector_simulator')


for ii in range(100) : 
#if True :

    job = qiskit.execute(qc, be) #, shots=100)

    res = job.result()

    print(res.get_statevector())
    print(res.get_counts())

