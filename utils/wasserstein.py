import torch
import math
import torch.linalg as linalg
from torch.autograd import Variable
import pdb

"""
This code is taken from: https://github.com/tianlinxu312/cot-gan-pytorch/blob/main/gan_utils.py
on 18.08.2023
"""

def cost_matrix(x, y, p=2):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param p: power
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    b = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    c = torch.sum(b, -1)
    return c


def modified_cost(x, y, h, M):
    '''
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L2_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    '''
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaMt = M[:, 1:, :] - M[:, :-1, :]
    ht = h[:, :-1, :]
    time_steps = ht.shape[1]
    sum_over_j = torch.sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], -1)
    C_hM = torch.sum(sum_over_j, -1) / time_steps

    # Compute L2 cost $\sum_t^T |x^i_t - y^j_t|^2$
    cost_xy = cost_matrix(x, y)

    return cost_xy + C_hM


def compute_sinkhorn(x, y, h, M, epsilon=0.1, niter=10):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    n = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    C = modified_cost(x, y, h, M) # shape: [batch_size, batch_size]b

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = 1. / n * torch.ones(n, requires_grad=False, device=x.device)
    nu = 1. / n * torch.ones(n, requires_grad=False, device=x.device)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-4)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.logsumexp(A, dim=-1, keepdim=True)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).item():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def scale_invariante_martingale_regularization(M, reg_lam):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m, t, j = M.shape
    # m = torch.tensor(m).type(torch.FloatTensor)
    # t = torch.tensor(m).type(torch.FloatTensor)
    # compute delta M matrix N
    N = M[:, 1:, :] - M[:, :-1, :]
    N_std = N / (torch.std(M, (0, 1)) + 1e-06)

    # Compute \sum_i^m(\delta M)
    sum_m_std = torch.sum(N_std, 0) / m
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = torch.sum(torch.abs(sum_m_std)) / t
    # the total pM term
    pm = reg_lam * sum_across_paths
    return pm


def compute_mixed_sinkhorn_loss(f_real, f_fake, m_real, m_fake, h_fake, sinkhorn_eps, sinkhorn_l,
                                f_real_p, f_fake_p, m_real_p, h_real_p, h_fake_p, scale=False):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    f_real = f_real.reshape(f_real.shape[0], f_real.shape[1], -1)
    f_fake = f_fake.reshape(f_fake.shape[0], f_fake.shape[1], -1)
    f_real_p = f_real_p.reshape(f_real_p.shape[0], f_real_p.shape[1], -1)
    f_fake_p = f_fake_p.reshape(f_fake_p.shape[0], f_fake_p.shape[1], -1)
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xyp = compute_sinkhorn(f_real_p, f_fake_p, h_fake_p, m_real_p, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xx = compute_sinkhorn(f_real, f_real_p, h_real_p, m_real, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_yy = compute_sinkhorn(f_fake, f_fake_p, h_fake_p, m_fake, sinkhorn_eps, sinkhorn_l, scale=scale)

    loss = loss_xy + loss_xyp - loss_xx - loss_yy
    return loss

# taken from https://gist.github.com/Flunzmas/6e359b118b0730ab403753dcc2a447df on 18.08.2023
def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    # mean_term = torch.sum(torch.mul(diff, diff))  # scalar
    mean_term = diff.abs().mean()
    # put it together
    return (trace_term + mean_term).float()
