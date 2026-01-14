import torch

t1= torch.Tensor([[1,2],[2,3]])
print(t1)

print(t1.shape)

def f(x):
    return x*2
print(f(3))

f= lambda x:x*2
print(f(3))  ##lambda 함수 익명 함수
         ## 함수명 = lamdba x:

t2=torch.FloatTensor([[1,2],[2,3]])
print(t2.shape)
print(torch.mm(t1,t2)) ##2D  only
print(torch.matmul(t1,t2)) ## 행렬곱 matrix multiplication

t3=torch.rand(1,2,3)
print(t3)

t4=torch.FloatTensor([[1,2,3],[3,4,5],[6,7,8]])
print(t4.shape)
print(t4.dim())


t5=torch.IntTensor([[1,2,3],[3,4,5],[6,7,8]])
print(t5)

print(t1+t2)
print(torch.add(t1,t2))
print(t1.add(t2))

print(t1.argmax())

print(torch.stack([t1,t2]))
print(torch.cat([t1,t2],dim=0))
print(torch.cat([t1,t2],dim=1))