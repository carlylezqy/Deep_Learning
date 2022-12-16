import time
import torch

if __name__ == "__main__":
    cuda  = torch.device('cuda')     # Default CUDA device
    cuda0 = torch.device('cuda:0')
    cuda2 = torch.device('cuda:2')   # GPU 2 (these are 0-indexed)

    x = torch.tensor([1., 2.], device="cuda:0")
    y = torch.tensor([1., 2.], device="cuda:1")
    z = torch.tensor([1., 2.], device="cuda:2")

    try:
        z = x + y
    except RuntimeError:
        print("RuntimeError: This operation will result in an error.")

    #for i in [x, y, z]:
    #    print(i.device)

    start_time = time.time()

    stream_1 = torch.cuda.Stream()
    stream_2 = torch.cuda.Stream()
    
    A = torch.rand(100, 100, device='cuda:1')
    B = torch.rand(100, 100, device='cuda:2')

    torch.cuda.synchronize()
    with torch.cuda.stream(stream_1):
        C = torch.mm(A, A).to('cuda:1')
    
    with torch.cuda.stream(stream_1):
        D = torch.mm(B, B).to('cuda:2')
    
    print((C + D).shape)

    torch.cuda.synchronize()

    spending_time = time.time() - start_time
    print(spending_time, C.shape, D.shape)
    