import torch
import torch.nn.functional as F

n, k = 100, 10
beta = 1.0
lr = 0.05
steps = 300
l1_lambda = 0.01

delta_phi = torch.randn(n, k)

p = torch.randn(k, requires_grad=True)
p.data /= p.norm()

optimizer = torch.optim.Adam([p], lr=lr)

for step in range(steps):
    optimizer.zero_grad()
    
    logits = beta * (delta_phi @ p)
    
    nll = -torch.log(torch.sigmoid(logits) + 1e-8).mean()
    
    l1_penalty = l1_lambda * torch.norm(p, 1)
    loss = nll + l1_penalty
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        p /= p.norm()

    if step % 25 == 0:
        print(f"Step {step:03d} | Loss: {loss.item():.4f} | L1 Penalty: {l1_penalty.item():.4f}")
