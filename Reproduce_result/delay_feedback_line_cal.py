import numpy as np
from importlib import reload
import tgs
import rgs 


reload(tgs)
reload(rgs)

V_DELAY = 2e8 # m/s
V_FEEDBACK = 2e8 # m/s

def delay_line_tgs_ancilla(branch_param: np.ndarray, time: tgs.Time) -> float:
    t_P = time.P_a
    t_E = time.E_a
    t_CZ = time.CZ_a
    beta = time._beta

    tree = tgs.Tree_ancilla(branch_param=branch_param, miu=tgs.Miu(L=0., L_att=0., L_delay=0., m=0, miu_set=0.))
    d = tree.depth
    n_P = tree.num_qubits(d, d)/tree.b[0]
    n_E = tree.num_qubits(1,d-1)/tree.b[0] + beta
    n_CZ = tree.num_qubits(2, d-1)/tree.b[0] # or (2, d-1)?

    return float((n_P*t_P + n_E*t_E + n_CZ*t_CZ)*V_DELAY)

def delay_line_tgs_feedback(branch_param: np.ndarray, time: tgs.Time) -> float:

    return float((len(branch_param) -1 + 1/branch_param[0])*feedback_line_tgs_feedback(branch_param=branch_param, time=time))

def feedback_line_tgs_feedback(branch_param: np.ndarray, time: tgs.Time) -> float:
    t_E = time.E_f
    t_P = time.P_f
    tree = tgs.Tree_feedback(branch_param=branch_param, miu=tgs.Miu(L=0., L_att=0., L_delay=0., m=0, miu_set=0.))
    d = tree.depth
    n_d_minus_1 = tree.num_qubits(d-1, d-1) # d-1 level of tree b[0] ... b[d-2]
    n_d_minus_2 = tree.num_qubits(d-2,d-2) # d-2 level
    b_d_minus_1 = branch_param[d-1]
    
    return float(((n_d_minus_1 + n_d_minus_2 - 1)*t_E + b_d_minus_1*(n_d_minus_1-n_d_minus_2)*t_P)*V_FEEDBACK)

def delay_line_rgs_ancilla(branch_param: np.ndarray, n: int, time: tgs.Time) -> float:
    t_P = time.P_a
    t_E = time.E_a
    t_CZ = time.CZ_a

    logical_qubit = tgs.Tree_ancilla(branch_param=branch_param, miu=tgs.Miu(L=0., L_att=0., L_delay=0., m=0, miu_set=0.))
    d = logical_qubit.depth

    n_P = 1 + logical_qubit.num_qubits(d, d)
    n_E_CZ = logical_qubit.num_qubits(1, d-1)

    return float((n_P*t_P + n_E_CZ*(t_E + t_CZ))*V_DELAY)

def delay_line_rgs_feedback(branch_param: np.ndarray, n: int, time: tgs.Time) -> float:
    return float((len(branch_param) - 1 + 1/n) * feedback_line_rgs_feedback(branch_param=branch_param, n=n, time=time))

def feedback_line_rgs_feedback(branch_param: np.ndarray, n: int, time: tgs.Time) -> float:
    N = n
    t_E = time.E_f
    t_P = time.P_f
    logical_qubit = tgs.Tree_feedback(branch_param=branch_param, miu=tgs.Miu(L=0., L_att=0., L_delay=0., m=0, miu_set=0.))
    d = logical_qubit.depth
    n_d_minus_1 = logical_qubit.num_qubits(d-1, d-1) # d-1 level of tree b[0] ... b[d-2]
    n_d_minus_2 = logical_qubit.num_qubits(d-2,d-2) # d-2 level
    b_d_minus_1 = branch_param[d-1]
    
    return float(N*((n_d_minus_1 + n_d_minus_2 - 1)*t_E + b_d_minus_1*(n_d_minus_1-n_d_minus_2)*t_P)*V_FEEDBACK)


if __name__ == "__main__":
    GAMMA = np.array([2e9, 100e9, 170e6, 100e9]) * 2 * np.pi
    T_SPIN_COHERENCE = [13e-3, 4e-6, 1., 1.]
    # tgs_a
    BRANCH_PARAM = [np.array([4,16,5]), np.array([1,1,19]), np.array([4,18,5]), np.array([4,16,5])]

    for i in range(4):
        name = 'tgs_a'
        time = tgs.Time(GAMMA[i])
        print(name + ": delay_line = {:.2f}".format(delay_line_tgs_ancilla(BRANCH_PARAM[i], time)))

    print()
    BRANCH_PARAM = [np.array([4,16,5]), np.array([4,22,6]), np.array([1,1,18]), np.array([4,15,5])]
    for i in range(4):
        name = 'tgs_f'
        time = tgs.Time(GAMMA[i])
        print(name + ": feedback_line = {:.2f}".format(feedback_line_tgs_feedback(BRANCH_PARAM[i], time)))
        print(name + ": delay_line = {:.2f}".format(delay_line_tgs_feedback(BRANCH_PARAM[i], time)))
        print()

    print()
    print()

    N = [32, 4, 32, 32]
    BRANCH_PARAM = [np.array([24,7]), np.array([1,1]), np.array([24,7]), np.array([24,7])]
    for i in range(4):
        name = 'rgs_a'
        time = tgs.Time(GAMMA[i]) 
        print(name + ": delay_line = {:.2f}".format(delay_line_rgs_ancilla(BRANCH_PARAM[i], N[i],time)))
    print()
    N = [14, 32, 4, 32]
    BRANCH_PARAM = [np.array([13,5]), np.array([25, 7]), np.array([4, 2]), np.array([24, 7])]
    for i in range(4):
        name = 'rgs_f'
        time = tgs.Time(GAMMA[i]) 
        print(name + ": feedback_line = {:.2f}".format(feedback_line_rgs_feedback(BRANCH_PARAM[i], N[i],time)))
        print(name + ": delay_line = {:.2f}".format(delay_line_rgs_feedback(BRANCH_PARAM[i], N[i],time)))
        print()
            