import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from importlib import reload
from numpy.typing import NDArray

import tgs
reload(tgs)

class RGS:
    def __init__(self, branch_param: NDArray, N: int, miu: tgs.Miu) -> None:
        # tree for logical qubits, P_succ, 
        self.N = N # no. of core nodes
        self.m = N/2  
        self.miu = miu
        self.logical_qubit: tgs.Tree = None
        self.miu_total:float = np.nan
        self.P_succ: float = np.nan
        pass

    @property
    def num_photons(self) -> int:
        return self.N * (self.logical_qubit.n_total + 1)

    def T_graph(self, time: tgs.Time) -> float:
        pass

class RGS_ancilla(RGS):
    def __init__(self, branch_param: NDArray, N: int, miu: tgs.Miu) -> None:
        super().__init__(branch_param, N, miu)

        self.logical_qubit = tgs.Tree_ancilla(branch_param, miu)
        self.miu_total = self.logical_qubit.miu_total(delay_apply=False)

        self.prob_BSM = 0.5*(1-self.miu_total)**2
        self.prob_X_measurement = self.logical_qubit.R[0]
        self.prob_Z_measurement = (1 - self.miu_total + self.miu_total*self.logical_qubit.R[1])**self.logical_qubit.b[0]

        # P_succ between neighbouring nodes
        self.P_succ = (1 - (1-self.prob_BSM)**(N/2)) * self.prob_X_measurement**2 * self.prob_Z_measurement**(N-2)

    def T_graph(self, time: tgs.Time) -> float:
        num_qubits = self.logical_qubit.num_qubits
        depth = self.logical_qubit.depth

        num_last_layer = num_qubits(depth, depth)
        num_1st_to_d_minus_1 = num_qubits(1, depth-1)
        
        return self.N*((1+num_last_layer)*time.P_a + num_1st_to_d_minus_1*time.E_a + (2+num_1st_to_d_minus_1)*time.CZ_a + 2*time.M) + time.M

class RGS_feedback(RGS):
    def __init__(self, branch_param: NDArray, N: int, miu: tgs.Miu) -> None:
        super().__init__(branch_param, N, miu)

        self.logical_qubit = tgs.Tree_feedback(branch_param, miu)
        self.miu_total = self.logical_qubit.miu_total(delay_apply=False)

        self.prob_BSM = 0.5*(1-self.miu_total)**2
        self.prob_X_measurement = self.logical_qubit.R[0]
        self.prob_Z_measurement = (1 - self.miu_total + self.miu_total*self.logical_qubit.R[1])**self.logical_qubit.b[0]

        # P_succ between neighbouring nodes
        self.P_succ = (1 - (1-self.prob_BSM)**(N/2)) * self.prob_X_measurement**2 * self.prob_Z_measurement**(N-2)

    def T_graph(self, time: tgs.Time) -> float:
        num_qubits = self.logical_qubit.num_qubits
        depth = self.logical_qubit.depth
        
        b0 = self.logical_qubit.b[0]
        num_last_layer = num_qubits(depth, depth)
        num_1st_to_d_minus_1 = num_qubits(1, depth-1)
    
        return self.N*(num_last_layer*time.P_f + (1/time._beta + num_1st_to_d_minus_1)*time.E_f + (b0+num_1st_to_d_minus_1)*time.CZ_f) + time.M

class RGS_Error(tgs.Error):
    def __init__(self, rgs:RGS, time: tgs.Time, t_spin_coherence, epsilon_depolarization=0.00005, epsilon_sp_measure_set=None):
        epsilon_decoh = 3/4 * (1 - np.exp(-rgs.T_graph(time)/t_spin_coherence/rgs.num_photons))
        super().__init__(rgs.logical_qubit, time, t_spin_coherence, epsilon_depolarization, epsilon_decoh=epsilon_decoh, epsilon_sp_measure_set=epsilon_sp_measure_set)
        self.rgs = rgs


        self.epsilon_X = self.e_I_k[0] # logical X measurement error (A17)

        b0 = self.rgs.logical_qubit.b[0]
        R1 = self.rgs.logical_qubit.R[1]
        miu = self.rgs.miu_total

        self.epsilon_Z = 0. # logical Z measurement error (A14)
        for lk in range(b0+1): # lk: # of qubits in k+2 level that can only be directly measured
            counting = comb(b0, lk)
            # single_qubit_successful_indirect_measurement_given_can_be_measured 
            p  = R1/(1-miu+miu*R1)
            parity_error_prob = 0.5 * (1-(1-2*self.epsilon_sp_measure)**(lk)*(1-2*self.e_I_k[1])**(b0-lk)) # A14
            self.epsilon_Z += counting * (1-p)**lk * p**(b0-lk) * parity_error_prob

def RGS_fidelity(e_m, e_X, e_Z: float, num_core_nodes:int, num_repeater_nodes:int) -> float:
    
    m = num_core_nodes/2
    n = num_repeater_nodes

    E_Z = 0.25 - (0.25*(1-2*e_m)**(2*(n+1)) * (1-2*e_X)**(2*n))
    E_X = E_Z

    E_Y = 0.25 + ( 0.25 * (1-2*e_m)**(2*(n+1)) * (1-2*e_X)**(2*n) ) - ( 0.5 * (1-2*e_m)**(2*(n+1)) * (1-2*e_X)**n * (1-2*e_Z)**((2*m-2)*n) )

    # print((1-2*e_Z)**((2*m-2)*n))
    # print((1-2*e_X)**n)
    # print("EZ: ", E_Z)
    # print("EY: ", E_Y)
    # print("ratio with approx: ", E_Z/((n+1)*e_m))
    # print("ratio with approx: ", E_Y/((n+1)*e_m))
    return 1 - E_X - E_Y - E_Z
    
def effective_key_rate(rgs: RGS, time: tgs.Time, error:RGS_Error, M:int, n:int, L:float = 1000e3, L_ATT: float =20e3):
    
    # print("logical X measurement error: ", error.epsilon_X)
    # print("logical Z measurement error: ", error.epsilon_Z)
    fidelity = RGS_fidelity(error.epsilon_sp_measure, error.epsilon_X, error.epsilon_Z, rgs.N, M)
    # print("fidelity:", fidelity)
    r = tgs.key_rate(fidelity)  
    # print("key_rate:", r)

    return r*rgs.P_succ**(M+1)/rgs.T_graph(time)/M/n*L/L_ATT