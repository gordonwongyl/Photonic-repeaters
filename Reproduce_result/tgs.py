import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.special import comb


# Table 1
# loss: loss probability 

class Miu:
    def __init__(self, L, L_att, L_delay, m, L_feedback=np.nan, miu_set=None):
        self._L = L
        self._L_att = L_att
        self.L_delay = L_delay  # does this depends on the position of qubit?
        self.m = m
        self.L_f = L_feedback
        self.coup = 0.05
        self.int_ancilla = 0
        self.miu_set = miu_set
        # self.depth = 3

    @property
    def ext(self):
        return 1 - np.exp(-self._L/(self.m+1)/self._L_att) # input L as L/2 for RGS
    
    # @property
    # def ext_RGS(self):
    #     return 1 - np.exp(-self._L/2/(self.m+1)/self._L_att)
    
    def int_feedback(self, no_feedback): # not sure how to implement this
        return 1 - np.exp(-self.L_f*no_feedback/self._L_att)
        # return 0
    
    @property
    def delay(self):
        return  1 - np.exp(-self.L_delay/self._L_att)

    def total_tree_a(self, apply):
        if self.miu_set is not None:
            return self.miu_set
        if apply:
            return 1 - (1-self.ext)*(1-self.coup)*(1-self.int_ancilla)*(1-self.delay)
        else:
            return 1 - (1-self.ext)*(1-self.coup)*(1-self.int_ancilla) 

    def total_tree_f(self, apply, no_feedback):
        if self.miu_set is not None:
            return self.miu_set
        if apply:
            return 1 - (1-self.ext)*(1-self.coup)*(1-self.int_feedback(no_feedback))*(1-self.delay)
        else:
            return 1 - (1-self.ext)*(1-self.coup)*(1-self.int_feedback(no_feedback)) 


# time for operations:
class Time:
    def __init__(self, gamma, beta=500):
        self._gamma = gamma
        self._beta = beta 

        self.H = 100e-12  # s
        self.CZ_a = 100e-9 # s
        self.CZ_f = self.E_f # s
    
    @property
    def gamma(self):
        return self._gamma
    @property
    def P_a(self):
        return 1/self._gamma
    @property
    def P_f(self):
        return 1/self._gamma
    @property
    def M(self):
        return 10/self._gamma
    @property
    def E_a(self):    
        return self.P_a + self.H + self.M
    @property
    def E_f(self):
        return self._beta/self._gamma + self.H + self.M

# tree properties
class Tree:
    def __init__(self, branch_param: NDArray, miu: Miu) -> None:
        self.b = branch_param # array 
        self.depth = len(self.b)
        self.miu = miu # same for all photons? Assume No

        self.n_total = self.num_qubits(1, self.depth)

        # Probability of obtaining an outcome from an indirect Z measurement of any photon in i-th level of the tree (eq. 5)
        # Here is 1 indexed (same as paper)
        self.R = np.zeros(self.depth, dtype=np.float128)
        # self.R[0] = np.nan (R0 is required for calculating prob_X_measurement in RGS)

        # 
        # miu = self.miu.total_tree_a(apply=True)
        # miu1 = self.miu.total_tree_a(apply=True)

        miu = self.miu_total(delay_apply=False)
        miu1 = miu
        
        for i in range(self.depth-1, -1, -1): # d-1 ... 0
            if i == self.depth-1: 
                R_i_plus_2 = np.float128(0)
                b_i_plus_1 = 0 # no qubits at depth+1 level
                # self.R[i] = 0
                # continue
            elif i == self.depth-2: 
                R_i_plus_2 = np.float128(0) # qubits at the last level cannot be indirectly measured
                b_i_plus_1 = self.b[i+1] 
            else:
                b_i_plus_1 = self.b[i+1] 
                R_i_plus_2 = self.R[i+2] 
            
            # print(R_i_plus_2)
            # print(b_i_plus_1)
            self.R[i] = np.float128(1 - (1 - (1-miu)*(1-miu+miu*R_i_plus_2)**b_i_plus_1)**self.b[i]) # confirm this

        # Success probability between neighbouring nodes (eq. 3, without ^m+1)
        if self.depth >= 3:
            R_2 = self.R[2] 
        else:   
            R_2 = np.float128(0)
        self.P_succ = ((1-miu1+miu1*self.R[1])**self.b[0] - (miu1*self.R[1])**self.b[0])*(1-miu+miu*R_2)**self.b[1]
    
    

    # Number of qubits from depth i to j 
    def num_qubits(self, start: int, end: int) -> int:
        return np.sum([np.product(self.b[0:i]) for i in range(start, end+1)])
    

    def miu_list(self, apply_list) -> list:
        pass

    def miu(self, apply) -> float:
        pass
        # return self.miu.total_tree_a(apply)

    # Generation time (s)
    def T_tree(self, time: Time) -> float:
        pass
    
# Ancilla and Feedback Schemes changes
class Tree_ancilla(Tree):
    def __init__(self, branch_param: NDArray, miu: Miu) -> None:
        super().__init__(branch_param, miu)
    
    def miu_total(self, delay_apply) -> float:
        return self.miu.total_tree_a(delay_apply)

    def T_tree(self, time:Time) -> float:
        num_last_layer = self.num_qubits(self.depth,self.depth)
        num_2nd_to_d_minus_1 = self.num_qubits(2,self.depth-1)
        num_1st_to_d_minus_1 = self.num_qubits(1,self.depth-1)
        return num_last_layer*time.P_a + (time._beta*self.b[0] + num_2nd_to_d_minus_1)*time.E_a + num_1st_to_d_minus_1*time.CZ_a
    
class Tree_feedback(Tree):
    def __init__(self, branch_param: NDArray, miu: Miu) -> None:
        super().__init__(branch_param, miu)
    
    def miu_total(self, delay_apply) -> float:
        no_feedback = 0 # Assume the worst case for all qubits
        return self.miu.total_tree_f(delay_apply, no_feedback)
    
    def T_tree(self, time: Time) -> float:
        num_last_layer = self.num_qubits(self.depth,self.depth)
        num_1st_to_d_minus_1 = self.num_qubits(1,self.depth-1)
        return num_last_layer*time.P_f + (self.b[0]+ num_1st_to_d_minus_1)*time.E_f + num_1st_to_d_minus_1*time.CZ_f 
 
# Generation time (s)
# def T_tree_a(tree: Tree, time: Time) -> float:
#     num_last_layer = tree.num_qubits(tree.depth,tree.depth)
#     num_2nd_to_d_minus_1 = tree.num_qubits(2,tree.depth-1)
#     num_1st_to_d_minus_1 = tree.num_qubits(1,tree.depth-1)
#     return num_last_layer*time.P_a + (tree.b[0] + num_2nd_to_d_minus_1)*time.E_a + num_1st_to_d_minus_1*time.CZ_a

# Error from single-photon measurement
class Error:
    def __init__(self, tree:Tree, time:Time,  t_spin_coherence, epsilon_depolarization=5e-5, epsilon_decoh=None, epsilon_sp_measure_set=None):
        if epsilon_sp_measure_set is None:
            if epsilon_decoh is None: # for TGS
                self.epsilon_decoh =  3/4*(1-np.exp(-tree.T_tree(time)/t_spin_coherence/tree.n_total))
            else:
                self.epsilon_decoh = epsilon_decoh # for RGS
            self.epsilon_sp_measure = 2/3*(self.epsilon_decoh + epsilon_depolarization - 4/3*self.epsilon_decoh*epsilon_depolarization) # error of single photon measurement
        else:
            self.epsilon_sp_measure = epsilon_sp_measure_set
        # print("e_sp: ", self.epsilon_sp_measure)


        miu = tree.miu_total(delay_apply=False)
        miu1 = tree.miu_total(delay_apply=False)
        # print("miu", miu)
        # miu = tree.miu_total(delay_apply=True)
        # miu1 = tree.miu_total(delay_apply=True)

        self.tree = tree
        self.R = np.zeros(tree.depth)
        self.R[0] = np.nan


        # e_I_k: average error probablity of indirect measurements on a qubit A in the kth level
        self.e_I_k = np.zeros(tree.depth)
        # self.e_I_k[0] = np.nan

        for k in range(tree.depth-1, -1, -1): # e_I_0 is for error_X_measurement


            if k == tree.depth-1: # no qubits at depth+1 level
                R_k_plus_2 = 0.
                b_k_plus_1 = 0
                e_I_k_plus_2 = 0.
            elif k == tree.depth-2: # qubits at the last level cannot be indirectly measured
                b_k_plus_1 = tree.b[k+1] 
                R_k_plus_2 = 0.
                e_I_k_plus_2 = 0.
            else:
                b_k_plus_1 = tree.b[k+1] 
                R_k_plus_2 = tree.R[k+2] 
                e_I_k_plus_2 = self.e_I_k[k+2] 

            
            # e_I_k|B: error prob from measurements of any qubit B and its children
            e_I_k_B = 0.
            for lk in range(b_k_plus_1+1): # lk: # of qubits in k+2 level that can only be directly measured
                counting = comb(b_k_plus_1, lk)
                # single_qubit_successful_indirect_measurement_given_can_be_measured 
                p  = R_k_plus_2/(1-miu+miu*R_k_plus_2)
                parity_error_prob = 0.5 * (1-(1-2*self.epsilon_sp_measure)**(1+lk)*(1-2*e_I_k_plus_2)**(b_k_plus_1-lk)) # how
                e_I_k_B += counting * (1-p)**lk * p**(b_k_plus_1-lk) * parity_error_prob
            
            # majority vote over measuring outcomes of mk qubits B1 ... B_mk
            S_k = (1-miu)*(1-miu+miu*R_k_plus_2)**b_k_plus_1 # A7

            def T_k(mk: int): #A9
                return comb(tree.b[k], mk) * S_k**mk * (1-S_k)**(tree.b[k]-mk)
            
            def e_I_k_mk(mk: int): # A10
                if mk%2 == 1:
                    return np.sum([comb(mk, j) * (e_I_k_B)**j * (1-e_I_k_B)**(mk-j) for j in range(int(np.ceil(mk/2)), mk+1)])
                else:
                    return np.sum([comb(mk-1, j) * (e_I_k_B)**j * (1-e_I_k_B)**(mk-1-j) for j in range(int(np.ceil(mk/2)), mk)])
                    # return np.sum([comb(mk, j) * (e_I_k_B)**j * (1-e_I_k_B)**(mk-j) for j in range(int(np.ceil(mk/2)), mk+1)])
            
            self.R[k] = np.sum([T_k(mk) for mk in range(1, tree.b[k]+1)]) # is this equal to tree.R[k]? #A12

            self.e_I_k[k] = np.sum([T_k(mk)*e_I_k_mk(mk) for mk in range(1, tree.b[k]+1)])/self.R[k] #A11




    # e_incorrect: average probability of incorrect decoding
        b0 = tree.b[0]
        b1 = tree.b[1]
        R1 = tree.R[1]
        
        if tree.depth >= 3:
            R2 = tree.R[2]
        else:
            R2 = 0.

        e_incorrect = 0.
        for l in range(b0 - 1 + 1):
            for n in range(b0 - l + 1):
                for m in range(b1 + 1):
                    counting = comb(b0, l)*comb(b0-l, n)*comb(b1, m)
                    successfully_decode_prob = (miu1*R1)**l * ((1-miu1)*(1-R1))**n * ((1-miu1)*R1)**(b0-l-n) * ((1-miu)*(1-R2))**m * R2**(b1-m)
                    e_incorrect += counting * successfully_decode_prob * self.e_n_m(n,m)
        self.e_incorrect = e_incorrect

    # n: 1st level qubits that can only be directly measured
    # m: 2nd level qubits that can only be directly measured 
    def e_n_m(self, n, m):
        
        e_I_1 = self.e_I_k[1]
        if self.tree.depth > 2:
            e_I_2 = self.e_I_k[2]
        else:
            e_I_2 = 0
        e_m = self.epsilon_sp_measure
        # correct parity for 1st level qubits
        b0 = self.tree.b[0]
        b1 = self.tree.b[1]
        correct_parity_prob_n = 1
        for i in range(n+1): # i: # of incorrectly measured (direct) qubits in 1st level 
            direct = comb(n, i) * e_m **i * (1-e_m)**(n-i)
            # j: # of incorrectly measured (indirect) qubits in 1st level 
            # i + j must be odd so that the overall parity is wrong
            j_list = np.arange(b0-1-n + 1)
            j_list = j_list[np.where((i+j_list) % 2 ==1)] # Why must 0 be included?
            indirect = 0
            for j in j_list:
                indirect += comb(b0 - 1 - n, j)* e_I_1**j * (1-e_I_1)**(b0-1-n-j)
            correct_parity_prob_n -= direct*indirect
        
       # correct parity for 2nd level qubits 
        correct_parity_prob_m = 1
        for i in range(m+1): # i: # of incorrectly measured (direct) qubits in 1st level 
            direct = comb(m, i) * e_m **i * (1-e_m)**(m-i)
            # j: # of incorrectly measured (indirect) qubits in 1st level 
            # i + j must be odd so that the overall parity is wrong
            j_list = np.arange(b1-m + 1)
            j_list = j_list[np.where((i+j_list) % 2 ==1)] # Why must 0 be included?
            indirect = 0
            for j in j_list:
                indirect += comb(b1 - 1 - m, j)* e_I_2**j * (1-e_I_2)**(b1-m-j)
            correct_parity_prob_m -= direct*indirect
    
        return 1 - (1-e_m)*correct_parity_prob_n*correct_parity_prob_m
    
# effective secret key rate (R_eff) (eq. 1)
# Binary entropy function
def h(x): 
    return  - x*np.log2(x) - (1-x)*np.log2(1-x)
# secret fraction for six-state protocol
def key_rate(F): # invalid when F is too small
    return F - h(1-F) - F*h((3*F-1)/(2*F))

def effective_key_rate(tree: Tree, time: Time, error: Error, M: int, n: int, L: float = 1000e3, L_ATT: float = 20e3):
    fidelity = (1-error.e_incorrect/tree.P_succ)**(M+1)
    # print("fidelity:", fidelity)
    r = key_rate(fidelity)  
    # print("key_rate:", r)
    return r*tree.P_succ**(M+1)/tree.T_tree(time)/M/n*L/L_ATT