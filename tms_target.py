"""Toy Model of Superposition (Elhage et al. 2022) target model + data.
5 features -> 2 hidden dims -> reconstruct 5. Ground-truth 'mechanisms' = the 5 features."""
import torch, torch.nn as nn, torch.nn.functional as F

class TMS(nn.Module):
    def __init__(self, n_features=5, n_hidden=2):
        super().__init__()
        self.n_features, self.n_hidden = n_features, n_hidden
        self.W = nn.Parameter(torch.randn(n_features, n_hidden) * 0.1)  # encode
        self.b = nn.Parameter(torch.zeros(n_features))                  # decode bias
    def forward(self, x):                # x: [B, n_features] sparse in [0,1]
        h = x @ self.W                   # [B, n_hidden]
        out = h @ self.W.t() + self.b    # tied decoder
        return F.relu(out)

def gen_batch(B, n_features=5, p=0.05, device="cpu"):
    mask = (torch.rand(B, n_features, device=device) < p).float()
    val  = torch.rand(B, n_features, device=device)
    return mask * val

def train_tms(steps=4000, B=2048, n_features=5, n_hidden=2, p=0.05, seed=0, device="cpu"):
    torch.manual_seed(seed)
    m = TMS(n_features, n_hidden).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    imp = torch.ones(1, n_features, device=device)  # EQUAL feature importance
    for s in range(steps):
        x = gen_batch(B, n_features, p, device)
        out = m(x)
        loss = ((imp * (out - x) ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return m, float(loss)

if __name__ == "__main__":
    m, loss = train_tms()
    print(f"TMS trained. final weighted-recon loss={loss:.5f}")
    print("W (feature->hidden):\n", m.W.detach().numpy().round(3))
