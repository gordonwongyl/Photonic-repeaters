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
        pass

    @property
    def num_photons(self) -> int:
        return self.N * (self.logical_qubit.n_total + 1)

    def T_graph(self, time: tgs.Time) -> float:
        pass

    def P_succ2(self, m: int) -> float:
        N = self.N
        return (1 - (1-self.prob_BSM)**(N/2))**(m+1) * self.prob_X_measurement**(2*m) * self.prob_Z_measurement**((N-2)*m)


class RGS_ancilla(RGS):
    def __init__(self, branch_param: NDArray, N: int, miu: tgs.Miu) -> None:
        super().__init__(branch_param, N, miu)

        self.logical_qubit = tgs.Logical_qubit_ancilla(branch_param, miu)
        self.miu = [self.logical_qubit.miu.total_tree_a(delay_apply=False),
        self.logical_qubit.miu.total_tree_a(delay_apply=True)]

        self.prob_BSM = 0.5*(1-self.miu[0])**2
        self.prob_X_measurement = self.logical_qubit.R[0]
        self.prob_Z_measurement = (1 - self.miu[1] + self.miu[1]*self.logical_qubit.R[1])**self.logical_qubit.b[0]

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

        self.logical_qubit = tgs.Logical_qubit_feedback(branch_param, miu)

        self.miu = [self.logical_qubit.miu.total_tree_f(delay_apply=False, no_feedback=0), 
        self.logical_qubit.miu.total_tree_f(delay_apply=True, no_feedback=2)] # The photon for BSM is not delayed, and not scattered again


        self.prob_BSM = 0.5*(1-self.miu[0])**2
        self.prob_X_measurement = self.logical_qubit.R[0]
        self.prob_Z_measurement = (1 - self.miu[1] + self.miu[1]*self.logical_qubit.R[1])**self.logical_qubit.b[0]

        # P_succ between neighbouring nodes
        self.P_succ = (1 - (1-self.prob_BSM)**(N/2)) * self.prob_X_measurement**2 * self.prob_Z_measurement**(N-2)
    
    def T_graph(self, time: tgs.Time) -> float:
        num_qubits = self.logical_qubit.num_qubits
        depth = self.logical_qubit.depth
        
        b0 = self.logical_qubit.b[0]
        num_last_layer = num_qubits(depth, depth)
        num_1st_to_d_minus_1 = num_qubits(1, depth-1)
    
        return self.N*(num_last_layer*time.P_f + time.E_f_short_photon + num_1st_to_d_minus_1*time.E_f + (b0+num_1st_to_d_minus_1)*time.CZ_f) + time.M

class RGS_Error(tgs.Error):
    def __init__(self, rgs:RGS, time: tgs.Time, t_spin_coherence, epsilon_depolarization=0.00005, epsilon_sp_measure_set=None):
        epsilon_decoh = 3/4 * (1 - np.exp(-rgs.T_graph(time)/t_spin_coherence/rgs.num_photons))
        super().__init__(rgs.logical_qubit, time, t_spin_coherence, epsilon_depolarization, epsilon_decoh=epsilon_decoh, epsilon_sp_measure_set=epsilon_sp_measure_set)
        self.rgs = rgs


        self.epsilon_X = self.e_I_k[0] # logical X measurement error (A17)

        b0 = self.rgs.logical_qubit.b[0]
        R1 = self.rgs.logical_qubit.R[1]
        miu = self.rgs.logical_qubit.miu_total()[0]

        self.epsilon_Z = 0. # logical Z measurement error (A14)
        for lk in range(b0+1): # lk: # of qubits in k+2 level that can only be directly measured
            counting = comb(b0, lk)
            # single_qubit_successful_indirect_measurement_given_can_be_measured 
            p  = R1/(1-miu+miu*R1)
            parity_error_prob = 0.5 * (1-((1-2*self.epsilon_sp_measure)**(lk))*((1-2*self.e_I_k[1])**(b0-lk))) # A14
            self.epsilon_Z += counting * ((1-p)**lk) * (p**(b0-lk)) * parity_error_prob

def RGS_fidelity(e_m, e_X, e_Z: float, num_core_nodes:int, num_repeater_nodes:int, E_Y: float=None) -> float:
    
    m = num_core_nodes/2
    n = num_repeater_nodes

    E_Z = 0.25 - (0.25*(1-2*e_m)**(2*(n+1)) * (1-2*e_X)**(2*n))
    E_X = E_Z

    E_Y = 1/4 + ( 1/4 * (1-2*e_m)**(2*(n+1)) * (1-2*e_X)**(2*n) ) \
        - 1/2 * (1-2*e_m)**(2*(n+1)) * (1-2*e_X)**(n) * (1-2*e_Z)**((2*m-2)*n) 

    # print((1-2*e_Z)**((2*m-2)*n))
    # print((1-2*e_X)**n)
    # print("EZ: ", E_Z)
    # print("EY: ", E_Y)
    # print("ratio with approx: ", E_Z/((n+1)*e_m))
    # print("ratio with approx: ", E_Y/((n+1)*e_m))
    return 1 - (E_X + E_Y + E_Z)
    # return 1 - 2/3* (E_X + E_Y + E_Z)
    
def effective_key_rate(rgs: RGS, time: tgs.Time, error:RGS_Error, M:int, n:int, L:float = 1000e3, L_ATT: float =20e3):
    
    # print("logical X measurement error: ", error.epsilon_X)
    # print("logical Z measurement error: ", error.epsilon_Z)
    fidelity = RGS_fidelity(error.epsilon_sp_measure, error.epsilon_X, error.epsilon_Z, rgs.N, M)
    # print("fidelity:", fidelity)
    r = tgs.key_rate(fidelity)  
    # print("key_rate:", r)
    # return r*rgs.P_succ**(M+1)/rgs.T_graph(time)/M/n*L/L_ATT
    return r*rgs.P_succ2(M)/rgs.T_graph(time)/M/n*L/L_ATT


if __name__ == "__main__":
    BRANCH_PARAM = [[16,14,1], [11,11,1]]
    ep_0 = 0.2
    for i in range(2):
        miu = tgs.Miu(0, 0, L_delay=0, m=0, L_feedback=0, miu_set=ep_0)
        rgs = RGS_ancilla(BRANCH_PARAM[i], 1, miu=miu)
        time = tgs.Time(1, 500)
        error = RGS_Error(rgs, time, t_spin_coherence=1, epsilon_sp_measure_set=2.8e-5)
        print(rgs.logical_qubit.miu_total())
        print(f"Number of qubits: {rgs.logical_qubit.num_qubits(1,3)}")
        print(f"1 - PZ: {1 - (1 - ep_0 + ep_0*rgs.logical_qubit.R[1])**rgs.logical_qubit.b[0]}")
        print(f"1 - PX: {1 - rgs.logical_qubit.R[0]}")

        print(f"e_Z: {error.epsilon_Z}")
        print(f"e_X: {error.epsilon_X}")

    print('\n')
    BRANCH_PARAM = [[17,28,2], [12,23,2]]
    ep_0 = 0.2697
    for i in range(2):
        miu = tgs.Miu(0, 0, L_delay=0, m=0, L_feedback=0, miu_set=ep_0)
        rgs = RGS_ancilla(BRANCH_PARAM[i], 1, miu=miu)
        time = tgs.Time(1, 500)
        error = RGS_Error(rgs, time, t_spin_coherence=1, epsilon_sp_measure_set=5.6e-5)
        print(rgs.logical_qubit.miu_total())
        print(f"Number of qubits: {rgs.logical_qubit.num_qubits(1,3)}")
        print(f"1 - PZ: {1 - (1 - ep_0 + ep_0*rgs.logical_qubit.R[1])**rgs.logical_qubit.b[0]}")
        print(f"1 - PX: {1 - rgs.logical_qubit.R[0]}")

        print(f"e_Z: {error.epsilon_Z}")
        print(f"e_X: {error.epsilon_X}")

    
