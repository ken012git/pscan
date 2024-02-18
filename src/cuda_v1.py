import torch

import pscan_cuda_v1

class FastPScan(torch.autograd.Function):
    @staticmethod
    def pscan_fn(A, X):
        A_ = A[:,None,:].repeat(1, X.size(2), 1) # [bsize, seqlen] -> [bsize, dim, seqlen]
        A_, X_ = pscan_cuda_v1.forward(A_, X) # [bsize, dim, seqlen], [bsize, seqlen, dim]
        return A_, X_

    @staticmethod
    def pscan_fn_backward(A, X):
        A_ = A[:,None,:].repeat(1, X.size(2), 1)
        A_, X_ = pscan_cuda_v1.backward(A_, X)
        return A_, X_   


    @staticmethod
    def forward(ctx, A, X, Y_init):
        """
        deltaA.shape:  torch.Size([1, 1536, 14, 16])
        x.shape:  torch.Size([1, 1536, 16])                                                                        
        deltaB_u.shape:  torch.Size([1, 1536, 14, 16])
        u.shape:  torch.Size([1, 1536, 14]) 14
        """
        ctx.A = A.clone() # [bsize, seqlen]
        ctx.Y_init = Y_init[:, None, :].clone() # [bsize, dim] -> [bsize, 1, dim]
        ctx.A_star, ctx.X_star = FastPScan.pscan_fn(A, X) # [bsize, seqlen], [bsize, seqlen, dim]
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star
        A = ctx.A
        R = grad_output.contiguous()
        A_ = torch.roll(A, -1, dims=1)
        _, R = FastPScan.pscan_fn_backward(A_, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)

fn = FastPScan.apply


if __name__ == "__main__":

    # N, T, D = 4, 32, 64
    # A = torch.rand(N, T, dtype=torch.float32).requires_grad_().cuda()
    # X = torch.rand(N, T, D, dtype=torch.float32).requires_grad_().cuda()
    # Y_init = torch.rand(N, D, dtype=torch.float32).requires_grad_().cuda()

    # fn(A, X, Y_init)


    N, T, D = 4, 32, 64
    A = torch.ones(N, T, dtype=torch.int8).cuda()
    X = torch.ones(N, T, D, dtype=torch.int8).cuda()
    Y_init = torch.zeros(N, D, dtype=torch.int8).cuda()
    with torch.no_grad():
        fn(A, X, Y_init)