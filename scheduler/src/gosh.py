from .gobi import *
from .adahessian import Adahessian

def gosh_opt(model, cpu, app, init):
    optimizer = Adahessian([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100
    while iteration < 200:
        alloc_old = deepcopy(init.data)
        z, _ = model(cpu, app, init)
        optimizer.zero_grad(); z.backward(create_graph=True); optimizer.step(); scheduler.step()
        init.data = scale(init.data, 0)
        equal = equal + 1 if torch.all(alloc_old - init < 0.01) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False 
    return init