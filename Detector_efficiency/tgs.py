import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.special import comb


# Table 1
# loss: loss probability 

class Miu:
    def __init__(self, L, L_att, L_delay, m, L_feedback=0, miu_set=None):
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
    
    def int_feedback(self, no_feedback): # not sure how to implement this
        return 1 - np.exp(-self.L_f*no_feedback/self._L_att)
        # return 0
    
    def delay(self, delay_apply):
        if delay_apply:
            return  1 - np.exp(-self.L_delay/self._L_att)
        else: return 0.

    def total_tree_a(self, delay_apply):
        if self.miu_set is not None:
            return self.miu_set
        return 1 - (1-self.ext)*(1-self.coup)*(1-self.int_ancilla)*(1-self.delay(delay_apply))
        

    def total_tree_f(self, delay_apply, no_feedback):
        if self.miu_set is not None:
            return self.miu_set
        return 1 - (1-self.ext)*(1-self.coup)*(1-self.int_feedback(no_feedback))*(1-self.delay(delay_apply))

# Detector error effects
class Detector():
    def __init__(self, eta_d=1, p_dc=0) -> None:
        self.eta_d = eta_d
        self.p_dc = p_dc


    # Probability of loss including realistic detector from miu_list
    def loss(self, miu_list: NDArray) -> NDArray:
        p_eff = (1 - miu_list)*self.eta_d # effective suriving probability with detection efficiency < 1
        dc_1 = 2*(1-self.p_dc)*self.p_dc # P(dc at one of the detectors)
        return 1 - (p_eff + (1-p_eff)*dc_1)
    
    # Additional depolarization error in addition from existing epsilon
    def depolarizing_error(self, original_epsilon:float, miu_list: NDArray) -> NDArray:
        miu_eff = 1 - (1-miu_list)*self.eta_d
        epsilon_dc =  0.75*(miu_eff* 2*self.p_dc*(1-self.p_dc) + (1-miu_eff)*self.p_dc)
        return original_epsilon + epsilon_dc - 4/3 * original_epsilon * epsilon_dc

    # Additional depolarization error from Erroneous BSM, eta_t: transmission prob for photons from both sides
    def P_tomix_from_BSM_error(self, eta_t: float):
        eta_d = self.eta_d
        p_dc = self.p_dc

        # probability of becoming I/2 
        P_tomix = 1 - (-eta_d*eta_t + p_dc*(4*eta_d*eta_t - eta_t - 1))/(-eta_d*eta_t + p_dc*(8*eta_d*eta_t - 4*eta_t - 4))
        return P_tomix

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
    
    @property
    def E_a_long_photon(self):
        return self._beta*self.P_a + self.H + self.M
    
    @property
    def E_f_short_photon(self):
        return 1/self._gamma + self.H + self.M

# tree properties
class Tree:
    def __init__(self, branch_param: NDArray, miu: Miu, detector: Detector = Detector()) -> None:
        self.b = branch_param # array 
        self.depth = len(self.b)
        self.miu = miu # same for all photons? Assume No
        self.detector = detector

        self.n_total = self.num_qubits(1, self.depth)

        # Probability of obtaining an outcome from an indirect Z measurement of any photon in i-th level of the tree (eq. 5)
        # Here is 1 indexed (same as paper)
        self.R = np.zeros(self.depth, dtype=np.float128)

        # Including detector loss
        miu: NDArray = self.detector.loss(self.miu_total()) # miu[i] is the loss for the i+1 level photons
        self.miu_list = miu
        
        for i in range(self.depth-1, -1, -1): # d-1 ... 0
            if i == self.depth-1: 
                R_i_plus_2 = 0
                b_i_plus_1 = 0 # no qubits at depth+1 level
                miu_i_plus_1 = 0

            elif i == self.depth-2: 
                R_i_plus_2 = 0 # qubits at the last level cannot be indirectly measured
                b_i_plus_1 = self.b[i+1] 
                miu_i_plus_1 = miu[i+1]
            else:
                b_i_plus_1 = self.b[i+1] 
                R_i_plus_2 = self.R[i+2] 
                miu_i_plus_1 = miu[i+1]

            self.R[i] = 1 - (1 - (1-miu[i])*(1-miu_i_plus_1+miu_i_plus_1*R_i_plus_2)**b_i_plus_1)**self.b[i] # confirm this

        # Success probability between neighbouring nodes (eq. 3, without ^m+1)
        if self.depth >= 3:
            R_2 = self.R[2] 
        else:   
            R_2 = 0
        self.P_succ = ((1-miu[0]+miu[0]*self.R[1])**self.b[0] - (miu[0]*self.R[1])**self.b[0])*(1-miu[1]+miu[1]*R_2)**self.b[1]
    
    

    # Number of qubits from depth i to j 
    def num_qubits(self, start: int, end: int) -> int:
        return np.sum([np.product(self.b[0:i]) for i in range(start, end+1)])
    
    # Generation time (s)
    def T_tree(self, time: Time) -> float:
        pass

    def miu_total(self) -> NDArray:
        pass
    
# Ancilla and Feedback Schemes changes
class Tree_ancilla(Tree):
    def __init__(self, branch_param: NDArray, miu: Miu, detector: Detector = Detector()) -> None:
        super().__init__(branch_param, miu, detector)
    
    def miu_total(self) -> NDArray:
        miu_list = np.zeros(self.depth)
        for i in range(self.depth):
            miu_list[i] = self.miu.total_tree_a(delay_apply=i) # Apply delay line except for the first layer photons miu[0]
        return miu_list

    def T_tree(self, time:Time) -> float:
        num_last_layer = self.num_qubits(self.depth,self.depth)
        num_2nd_to_d_minus_1 = self.num_qubits(2,self.depth-1)
        num_1st_to_d_minus_1 = self.num_qubits(1,self.depth-1)
        return num_last_layer*time.P_a + self.b[0]*time.E_a_long_photon + num_2nd_to_d_minus_1*time.E_a + num_1st_to_d_minus_1*time.CZ_a

class Logical_qubit_ancilla(Tree_ancilla):
    def __init__(self, branch_param: NDArray, miu: Miu, detector: Detector = Detector()) -> None:
        super().__init__(branch_param, miu, detector)

    def miu_total(self) -> NDArray:
        miu_list = np.zeros(self.depth)
        for i in range(self.depth):
            miu_list[i] = self.miu.total_tree_a(delay_apply=True) # Apply delay line to all the first layer photons miu[0] as well
        return miu_list

class Tree_feedback(Tree):
    def __init__(self, branch_param: NDArray, miu: Miu, detector: Detector = Detector()) -> None:
        super().__init__(branch_param, miu, detector)
    
    def miu_total(self) -> NDArray:
        miu_list = np.zeros(self.depth)
        miu_list[0] = self.miu.total_tree_f(delay_apply=False, no_feedback=1) # First level do not go through the delay line
        for i in range(1, self.depth-1):
            miu_list[i] = self.miu.total_tree_f(delay_apply=True, no_feedback=1)
        miu_list[self.depth-1] = self.miu.total_tree_f(delay_apply=True, no_feedback=0)  # Last level do not go through feedback line
        return miu_list

    def T_tree(self, time: Time) -> float:
        num_last_layer = self.num_qubits(self.depth,self.depth)
        num_1st_to_d_minus_1 = self.num_qubits(1,self.depth-1)
        return num_last_layer*time.P_f + (self.b[0]+ num_1st_to_d_minus_1)*time.E_f + num_1st_to_d_minus_1*time.CZ_f 
 
class Logical_qubit_feedback(Tree_feedback):
    def __init__(self, branch_param: NDArray, miu: Miu, detector: Detector = Detector()) -> None:
        super().__init__(branch_param, miu, detector)
    
    def miu_total(self) -> NDArray:
        miu_list = np.zeros(self.depth)
        # in RGS, the 1st level photons go through delay line and is scattered twice
        miu_list[0] = self.miu.total_tree_f(delay_apply=True, no_feedback=2) 
        for i in range(1, self.depth-1):
            miu_list[i] = self.miu.total_tree_f(delay_apply=True, no_feedback=1)
        miu_list[self.depth-1] = self.miu.total_tree_f(delay_apply=True, no_feedback=0) # Last level do not go through feedback line 
        return miu_list

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
        # print(self.epsilon_sp_measure)

        # the indexing of epsilon_sp_measure is similar to miu
        # Errors introduced by depolarizing error from realistic detection
        miu = tree.miu_list
        self.epsilon_sp_measure = tree.detector.depolarizing_error(original_epsilon=self.epsilon_sp_measure,
                                                                   miu_list=miu)
        # print(self.epsilon_sp_measure)
        # print("e_sp: ", self.epsilon_sp_measure)
        self.tree = tree
        self.R = np.zeros(tree.depth)
        self.R[0] = np.nan


        # e_I_k: average error probablity of indirect measurements on a qubit A in the kth level
        self.e_I_k = np.zeros(tree.depth)

        for k in range(tree.depth-1, -1, -1): # e_I_0 is for error_X_measurement


            if k == tree.depth-1: # no qubits at depth+1 level
                R_k_plus_2 = 0.
                b_k_plus_1 = 0
                e_I_k_plus_2 = 0.
                miu_k_plus_1 = 0.
                e_sp_k_plus_1 = 0.

            elif k == tree.depth-2: # qubits at the last level cannot be indirectly measured
                b_k_plus_1 = tree.b[k+1] 
                e_sp_k_plus_1 = self.epsilon_sp_measure[k+1]
                miu_k_plus_1 = miu[k+1]
                R_k_plus_2 = 0.
                e_I_k_plus_2 = 0.
            else:
                b_k_plus_1 = tree.b[k+1] 
                miu_k_plus_1 = miu[k+1]
                R_k_plus_2 = tree.R[k+2] 
                e_I_k_plus_2 = self.e_I_k[k+2] 

            
            # e_I_k|B: error prob from measurements of any qubit B and its children
            e_I_k_B = 0.
            for lk in range(b_k_plus_1+1): # lk: # of qubits in k+2 level that can only be directly measured
                counting = comb(b_k_plus_1, lk)
                # single_qubit_successful_indirect_measurement_given_can_be_measured 
                p  = R_k_plus_2/(1-miu_k_plus_1+miu_k_plus_1*R_k_plus_2)
                parity_error_prob = 0.5 * (1-(1-2*self.epsilon_sp_measure[k])*(1-2*e_sp_k_plus_1)**(lk)*(1-2*e_I_k_plus_2)**(b_k_plus_1-lk)) 
                e_I_k_B += counting * (1-p)**lk * p**(b_k_plus_1-lk) * parity_error_prob
                
            # majority vote over measuring outcomes of mk qubits B1 ... B_mk
            S_k = (1-miu[k])*(1-miu_k_plus_1+miu_k_plus_1*R_k_plus_2)**b_k_plus_1 # A7

            def T_k(mk: int): #A9
                return comb(tree.b[k], mk) * S_k**mk * (1-S_k)**(tree.b[k]-mk)
            
            def e_I_k_mk(mk: int): # A10
                if mk%2 == 1:
                    return np.sum([comb(mk, j) * (e_I_k_B)**j * (1-e_I_k_B)**(mk-j) for j in range(int(np.ceil(mk/2)), mk+1)])
                else:
                    return np.sum([comb(mk-1, j) * (e_I_k_B)**j * (1-e_I_k_B)**(mk-1-j) for j in range(int(np.ceil(mk/2)), mk)])
            
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
                    # print(l,n,m)
                    counting = comb(b0, l)*comb(b0-l, n)*comb(b1, m)
                    successfully_decode_prob = (miu[0]*R1)**l * ((1-miu[0])*(1-R1))**n * ((1-miu[0])*R1)**(b0-l-n) * ((1-miu[1])*(1-R2))**m * R2**(b1-m)
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
        e_m_1st_level = self.epsilon_sp_measure[0]
        e_m_2nd_level = self.epsilon_sp_measure[1]
        # correct parity for 1st level qubits
        b0 = self.tree.b[0]
        b1 = self.tree.b[1]
        even_parity_n = (1 + (1-2*e_m_1st_level)**min(n, b0-1) * (1-2*e_I_1)**max(b0-1-n, 0))/2
        # correct parity for 2nd level qubits 
        even_parity_m = (1 + (1-2*e_m_2nd_level)**m * (1-2*e_I_2)**(b1-m))/2

        return 1 - (1-e_m_1st_level)*even_parity_n*even_parity_m
    
# effective secret key rate (R_eff) (eq. 1)
# Binary entropy function
def h(x): 
    return - x*np.log2(x) - (1-x)*np.log2(1-x)

# secret fraction for six-state protocol
def key_rate(F): # invalid when F is too small
    return F - h(1-F) - F*h((3*F-1)/(2*F))

# Fidelity for the complete channel 
def TGS_fidelity(tree: Tree, error: Error, m: int):
    e_I_1 = error.e_I_k[1]
    if error.tree.depth > 2:
        e_I_2 = error.e_I_k[2]
        R_2 = tree.R[2]
    else:
        e_I_2 = 0
        R_2 = 0
    
    e_m_1st_level = error.epsilon_sp_measure[0]
    e_m_2nd_level = error.epsilon_sp_measure[1]
    # correct parity for 1st level qubits
    b0 = tree.b[0]
    b1 = tree.b[1]
    R_1 = tree.R[1]
    
    miu0, miu1 = tree.miu_list[0], tree.miu_list[1]
    N_x = (1-miu0+miu0*R_1)**b0 - (miu0*R_1)**b0
    N_z = (1-miu1+miu1*R_2)**b1
    ep_Z = 0.5* (1 -  1/N_z *((1-miu1)*(1-R_2)*(1-2*e_m_2nd_level) + R_2*(1-2*e_I_2))**b1)

    e_X = [0.5* (1 - (1-2*e_m_1st_level)**min(n, b0-1) * (1-2*e_I_1)**max(b0-1-n, 0)) for n in range(0, b0 + 1)]
    g = [comb(b0, l)*(miu0*R_1)**l * ((1-miu0)*R_1)**(b0-l) for l in range(0, b0)]

    ep_X = 0

    for l in range(b0):
        for n in range(b0 - l + 1):
            ep_X += comb(b0-l, n) * (1/R_1 - 1)**n * g[l] * e_X[n]
    ep_X = ep_X/N_x
    # print(ep_X, ep_Z)

    print("e_decoding: ", error.e_incorrect/tree.P_succ)
    print("e_decoding 2: ", 1 - (1-e_m_1st_level)*(1-ep_X)*(1-ep_Z))

    return ( (1 - 2*ep_X)*(1 - 2*ep_Z)*(1 - 2*e_m_1st_level) )**(m+1) /4 + \
           ( (1 - 2*ep_X)*(1 - 2*e_m_1st_level) )**(m+1) /4 + \
           ( (1 - 2*ep_Z)*(1 - 2*e_m_1st_level) )**(m+1) /4 + 1/4



def effective_key_rate(tree: Tree, time: Time, error: Error, M: int, n: int, L: float = 1000e3, L_ATT: float = 20e3):
    fidelity = (1-error.e_incorrect/tree.P_succ)**(M+1)
    # print("fidelity1:", fidelity)
    r = key_rate(fidelity)  
    # print("key_rate:", r)
    return r*tree.P_succ**(M+1)/tree.T_tree(time)/M/n*L/L_ATT

def effective_key_rate2(tree: Tree, time: Time, error: Error, M: int, n: int, L: float = 1000e3, L_ATT: float = 20e3):
    fidelity = TGS_fidelity(tree, error, m=M)
    # print("fidelity2:", fidelity)
    r = key_rate(fidelity)  
    # print("key_rate:", r)
    return r*tree.P_succ**(M+1)/tree.T_tree(time)/M/n*L/L_ATT


if __name__=='__main__':
    # tree = Tree_feedback([4,2,3], miu=Miu(1000e3, 20e3, 500, 800., 200.))
    # time = Time(1e9)
    # error = Error(tree, time=None, t_spin_coherence=1, epsilon_sp_measure_set=0.15)

    #TGS_ancilla
    GAMMA = np.array([2e9, 100e9, 170e6, 100e9]) * 2 * np.pi
    T_SPIN_COHERENCE = [13e-3, 4e-6, 1., 1.]
    L_neighbour = [1.7e3, 30.3e3, 1.8e3, 1.7e3]
    BRANCH_PARAM = [np.array([4,16,5]), np.array([1,1,19]), np.array([4,18,5]), np.array([4,16,5])]
    L_delay = [398, 80.4, 627.9, 380.7]
    L = 1000e3
    L_ATT = 20e3

    for g, t, l_n, l_d, branch in zip(GAMMA, T_SPIN_COHERENCE, L_neighbour, L_delay, BRANCH_PARAM):
        M = L/l_n - 1
        miu = Miu(L, L_ATT, L_delay=l_d, m=M, L_feedback=0)
        tree = Tree_ancilla(branch, miu=miu)
        time = Time(g, 500)
        error = Error(tree, time, t_spin_coherence=t)
        print(effective_key_rate(tree, time, error, M, 3))
        print(effective_key_rate2(tree, time, error, M, 3))
        print()
    # M = 600
    # tree = Tree_ancilla([4,18,5], miu=Miu(1000e3, 20e3, 500, M, 200.))
    # error = Error(tree, time=None, t_spin_coherence=1, epsilon_sp_measure_set=0.005)
    # original_fidelity = (1-error.e_incorrect/tree.P_succ)**(1)
    # print(original_fidelity)

    # new_fidelity = TGS_fidelity(tree, error, 1)
    # print(new_fidelity)
