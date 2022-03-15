# def prox_ind(p, lamda):
#     p_norm = torch.sqrt((p**2).sum(3,keepdims=True))
#     return p/torch.maximum(p_norm/lamda,torch.ones_like(p_norm))

# def primal_step(u, p, ATq, tau, hz):
#     u_tmp = u - tau*(backward_torch(p, hz) + ATq)
#     return u_tmp

# def dual_step(p, u, hz, sigma, lamda):
#     p_tmp = p + sigma[None,None,None,:]*forward_torch(u, hz)
#     return prox_ind(p_tmp, lamda)

# ####################################################################
# # test primal update step
# print('\ntest primal update step: ')

# p = torch.randn(Z, M, N, 3).cuda()
# u = torch.randn(Z, M, N).cuda()
# ATq = torch.randn(Z, M, N).cuda()
# tau = 0.001 #tau = torch.randn(Z, M, N).cuda() #
# hz = 0.5

# ts = time.time()
# # pytorch result
# u_old = u.clone()
# u_update_torch = primal_step(u, p, ATq, tau, hz)
# print('primal - u unchanged by pytorch op: ', torch.allclose(u_old, u))
# print('primal - elapsed time pytorch: ', (time.time()-ts))

# ts = time.time()
# # cuda kernel result
# reco3d_tv.primal_step(u, p, ATq, tau, hz)
# print('primal - elapsed time cuda kernel: ', (time.time()-ts))

# # infinity norm to measure difference
# print('primal - Check diff with infty norm: ', torch.max(torch.abs(u_update_torch - u)).item())

# ####################################################################
# # test dual update step
# print('\ntest dual update step: ')

# sigma = torch.Tensor([0.5, 0.5, 0.25]).cuda()
# lamda = 0.01
# hz = 0.5
# ts = time.time()
# # pytorch result
# p_old = p.clone()
# p_update_torch = dual_step(p, u, hz, sigma, lamda)
# print('dual - p unchanged by pytorch op: ', torch.allclose(p_old, p))
# print('dual - elapsed time pytorch: ', (time.time()-ts))
# ts = time.time()
# # cuda kernel result
# reco3d_tv.dual_step(p, u, sigma, hz, lamda)
# print('dual - elapsed time cuda kernel: ', (time.time()-ts))

# # infinity norm to measure difference
# print('dual - Check diff with infty norm: ', torch.max(torch.abs(p_update_torch - p)).item())

# ####################################################################
# # test prox l2
# print('\ntest prox l2: ')

# q = torch.randn(Z, M, N).cuda()
# sigma = torch.abs(torch.randn(Z)).cuda()

# ts = time.time()
# # pytorch result
# sigma_old = sigma.clone()
# q_torch = q/(1+sigma[:,None,None])
# print('prox l2 - sigma remains unchanged by pytorch op: ', torch.allclose(sigma_old, sigma))
# print('prox l2 - elapsed time pytorch: ', (time.time()-ts))

# ts = time.time()
# # cuda kernel result
# reco3d_tv.prox_l2(q, sigma)
# print('prox l2 - elapsed time cuda kernel: ', (time.time()-ts))

# # infinity norm to measure difference
# print('prox l2 - Check diff with infty norm: ', torch.max(torch.abs(q_torch - q)).item())