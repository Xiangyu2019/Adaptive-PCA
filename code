import numpy as np

######Adaptive Loss######
sample, feature = X.shape
ori_dim = sample
pro_dim = 2
data_size = feature

lambda_ = 0.001
mu = lambda_
sigma = 0.5

X1 = normalize(X, axis=1, norm='l2')

I = np.identity(ori_dim)
U = I[:, :pro_dim]
niter = 300
V = np.random.rand(pro_dim, data_size)

U_step = []
V_step = []

for i in range(0, niter):
    SUM = sigma * (1 + sigma) / ((np.linalg.norm(X1 - (U @ V)) + sigma) ** 2)
    M = 2 * SUM * (X1 - (U @ V)) @ np.transpose(V) + mu * U
    s_U, s_d_U, d_U = np.linalg.svd(M, full_matrices=0, compute_uv=1)
    U = s_U @ np.transpose(d_U)
    
    for j in range(0, data_size):
        A = 2 * SUM * (X1[:, j] - (U @ V)[:, j]) / ((np.linalg.norm(X1 - (U @ V)) + sigma) ** 2)
        V[:, j] = (lambda_ - 2) * V[:, j] + (2 * A @ np.transpose(U) @ X[:, j])
        V[:, j] = V[:, j] / np.linalg.norm(V[:, j])
    
    N = 2 * SUM * (np.transpose(U) @ (X1 - U @ V)) + lambda_ * V
    s_V, s_d_V, d_V = np.linalg.svd(N, full_matrices=0, compute_uv=1)
    V = s_V @ d_V

    V_step.append(V.copy())
    U_step.append(U.copy())

print('U', U, U.shape)
print('V', V, V.shape)
