"""SPD decomposition with ANNEALED importance minimality.
Key change vs pilot: coeff_imp ramps 0 -> target over training, so components first learn to
reconstruct faithfully (gates open), THEN sparsity pressure carves them into specialized,
per-feature subcomponents — avoiding the all-zero gate collapse seen in the pilot."""
import torch, torch.nn as nn, torch.nn.functional as F
from tms_target import TMS, gen_batch, train_tms

class RankOneComponents(nn.Module):
    def __init__(self, d_in, d_out, C, ci_hidden=32):
        super().__init__()
        self.C,self.d_in,self.d_out=C,d_in,d_out
        self.U=nn.Parameter(torch.randn(C,d_in)*0.1)
        self.V=nn.Parameter(torch.randn(C,d_out)*0.1)
        self.ci=nn.Sequential(nn.Linear(d_in,ci_hidden),nn.ReLU(),nn.Linear(ci_hidden,C))
        nn.init.constant_(self.ci[-1].bias, 2.0)   # start gates open
    def gates(self,x): return torch.sigmoid(self.ci(x))
    def component_matrices(self): return torch.einsum("ci,co->cio",self.U,self.V)
    def full_weight(self): return torch.einsum("ci,co->io",self.U,self.V)
    def forward(self,x,mask=None):
        comp=self.component_matrices(); g=self.gates(x)
        if mask is not None: g=g*mask
        xc=torch.einsum("bi,cio->bco",x,comp)
        return torch.einsum("bc,bco->bo",g,xc)

def decompose(target, C=20, steps=12000, B=4096, p=0.05, seed=0,
              coeff_faith=1e4, coeff_stoch=1.0, imp_max=8e-3, anneal_frac=0.6, device="cpu"):
    torch.manual_seed(seed)
    nf,nh=target.n_features,target.n_hidden
    enc=RankOneComponents(nf,nh,C).to(device); dec=RankOneComponents(nh,nf,C).to(device)
    Wt=target.W.detach(); b=target.b.detach()
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
    def tfwd(x): h=x@Wt; return F.relu(h@Wt.t()+b)
    for s in range(steps):
        # anneal importance coeff: 0 for first (1-anneal_frac), then linear ramp to imp_max
        frac=s/steps
        if frac < (1-anneal_frac): coeff_imp=0.0
        else: coeff_imp=imp_max*((frac-(1-anneal_frac))/anneal_frac)
        x=gen_batch(B,nf,p,device)
        L_faith=((enc.full_weight()-Wt)**2).mean()+((dec.full_weight()-Wt.t())**2).mean()
        with torch.no_grad(): tgt=tfwd(x)
        ge=enc.gates(x)
        me=torch.bernoulli(ge.clamp(0,1)).detach()+ge-ge.detach()
        h=enc(x,mask=me); gd=dec.gates(h)
        md=torch.bernoulli(gd.clamp(0,1)).detach()+gd-gd.detach()
        out=F.relu(dec(h,mask=md)+b)
        L_stoch=((out-tgt)**2).mean()
        L_imp=ge.abs().mean()+gd.abs().mean()
        loss=coeff_faith*L_faith+coeff_stoch*L_stoch+coeff_imp*L_imp
        opt.zero_grad(); loss.backward(); opt.step()
        if s%2000==0:
            print(f"  step {s:5d} imp_coeff {coeff_imp:.1e} | faith {L_faith:.1e} stoch {L_stoch:.4f} imp {L_imp:.3f}")
    return enc,dec

if __name__=="__main__":
    m,_=train_tms(seed=0)
    enc,dec=decompose(m)
    torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"W":m.W.detach(),"b":m.b.detach()},"spd_anneal.pt")
    print("saved spd_anneal.pt")
