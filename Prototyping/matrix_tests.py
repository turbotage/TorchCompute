import torch
import time

#torch.set_num_threads(6)
print(torch.get_num_threads())

gpu = False
n = 1000000
m = 2
epochs = 3

dev_str = "cpu"
if gpu:
    dev_str = "cuda:0"


A = torch.rand(2*n,m,m, device=dev_str)
#A[n:,:] = torch.bmm(A[n:,:].transpose(1,2), A[n:,:])
A = torch.bmm(A.transpose(1,2), A) + 0.2*torch.eye(m, device=dev_str).repeat(2*n,1,1) # This should be PD
b = torch.rand(2*n,m,1, device=dev_str)



# Test LU
t1 = time.perf_counter()

LU, pivots, LU_info = torch.lu(A, get_infos=True)
x = torch.lu_solve(b, LU, pivots)

t2 = time.perf_counter()
print(t2-t1)



# Test cholesky
t1 = time.perf_counter()

LL, LL_info = torch.linalg.cholesky_ex(A)
x = torch.cholesky_solve(b, LL)

t2 = time.perf_counter()
print(t2-t1)

LU_time = 0.0
Chol_time = 0.0

for i in range(0,epochs):

    A = torch.rand(2*n,m,m, device=dev_str)
    #A[n:,:] = torch.bmm(A[n:,:].transpose(1,2), A[n:,:])
    A = torch.bmm(A.transpose(1,2), A) + 0.2*torch.eye(m, device=dev_str).repeat(2*n,1,1) # This should be PD
    b = torch.rand(2*n,m,1, device=dev_str)



    # Test LU
    t1 = time.perf_counter()

    LU, pivots, LU_info = torch.lu(A, get_infos=True)
    x = torch.lu_solve(b, LU, pivots)

    if gpu:
        torch.cuda.synchronize()

    t2 = time.perf_counter()
    LU_time += t2 - t1



    # Test cholesky
    t1 = time.perf_counter()

    LL, LL_info = torch.linalg.cholesky_ex(A)
    x = torch.cholesky_solve(b, LL)

    if gpu:
        torch.cuda.synchronize()

    t2 = time.perf_counter()
    Chol_time += t2 - t1

print("avg LU-time: " + str(LU_time / epochs))
print("avg Chol-time: " + str(Chol_time / epochs))