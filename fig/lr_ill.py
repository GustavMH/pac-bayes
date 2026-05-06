import torch
import matplotlib.pyplot as plt
from models.schedulers import TriangularCyclicLR, CyclicCosineAnnealingLR

model = torch.nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.step()
t = TriangularCyclicLR(optimizer)
c = CyclicCosineAnnealingLR(optimizer)

def s(sched):
    lr = sched.get_lr()[0]
    sched.step()
    return lr

fig, ax = plt.subplots(1, 2, figsize=(5.5, 2), layout="tight")
C = 20
a_lo, a_hi = 0.01, 0.1
ax[0].plot(list(range(1, 41)), [s(t) for i in range(40)], c="black")
ax[1].plot(list(range(1, 41)), [s(c) for i in range(40)], c="black")
ax[0].set_xlabel("Batch no.")
ax[1].set_xlabel("Epoch no.")
ax[0].set_title("Triangular")
ax[1].set_title("Cosine annealling")
ax[0].set_yticks([a_lo, a_hi], ["$\\alpha_{min}$", "$\\alpha_{max}$"])
ax[1].set_yticks([a_hi], ["$\\alpha_{0}$"])

plt.savefig("lr.pdf")
plt.close()
