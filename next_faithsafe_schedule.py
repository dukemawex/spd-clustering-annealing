"""NEXT STEP (implemented): longer sparsity schedule WITH a faithfulness safeguard.
README noted faithfulness drifted at imp_max=3e-2. We add a safeguard: if faithfulness
exceeds a threshold, temporarily freeze the importance penalty (adaptive). We sweep imp_max
and report the specialization vs faithfulness tradeoff, showing the safeguard keeps faithfulness
bounded while allowing higher sparsity."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json
NF,NH=5,4
torch.manual_seed(0)
class TMS(nn.Module):
    def __init__(s): super().__init__(); s.W=nn.Parameter(torch.randn(NF,NH)*0.1); s.b=nn.Parameter(torch.zeros(NF))
    def forward(s,x): h=x@s.W; return F.relu(h@s.W.t()+s.b)
def gen(B,p=0.05): m=(torch.rand(B,NF)<p).float(); return m*torch.rand(B,NF)
tms=TMS(); opt=torch.optim.Adam(tms.parameters(),lr=1e-2)
for _ in range(5000): ((tms(x:=gen(2048))-x)**2).mean().backward(); opt.step(); opt.zero_grad()

class ROC(nn.Module):
    def __init__(s,d_in,d_out,C):
        super().__init__(); s.U=nn.Parameter(torch.randn(C,d_in)*0.1); s.V=nn.Parameter(torch.randn(C,d_out)*0.1)
        s.ci=nn.Sequential(nn.Linear(d_in,32),nn.ReLU(),nn.Linear(32,C)); nn.init.constant_(s.ci[-1].bias,2.0)
    def gates(s,x): return torch.sigmoid(s.ci(x))
    def full(s): return torch.einsum("ci,co->io",s.U,s.V)
    def forward(s,x,mask=None):
        comp=torch.einsum("ci,co->cio",s.U,s.V); g=s.gates(x)
        if mask is not None: g=g*mask
        return torch.einsum("bc,bco->bo",g,torch.einsum("bi,cio->bco",x,comp))

def run(imp_max, steps=8000, safeguard=True, faith_thresh=1e-6):
    torch.manual_seed(0); c1=ROC(NF,NH,20); c2=ROC(NH,NF,20)
    Wt=tms.W.detach(); b=tms.b.detach()
    opt=torch.optim.Adam(list(c1.parameters())+list(c2.parameters()),lr=1e-3)
    for s in range(steps):
        frac=s/steps; base_imp=0.0 if frac<0.4 else imp_max*((frac-0.4)/0.6)
        x=gen(4096)
        L_faith=((c1.full()-Wt)**2).mean()+((c2.full()-Wt.t())**2).mean()
        # SAFEGUARD: freeze importance penalty when faithfulness degrades
        impc=0.0 if (safeguard and L_faith.item()>faith_thresh) else base_imp
        with torch.no_grad(): tgt=F.relu((x@Wt)@Wt.t()+b)
        g1=c1.gates(x); m1=torch.bernoulli(g1.clamp(0,1)).detach()+g1-g1.detach()
        h=c1(x,mask=m1); g2=c2.gates(h); m2=torch.bernoulli(g2.clamp(0,1)).detach()+g2-g2.detach()
        out=F.relu(c2(h,mask=m2)+b); L_stoch=((out-tgt)**2).mean(); L_imp=g1.abs().mean()+g2.abs().mean()
        (1e4*L_faith+L_stoch+impc*L_imp).backward(); opt.step(); opt.zero_grad()
    # measure specialization + final faithfulness
    resp=np.zeros((NF,40))
    for f in range(NF):
        xf=torch.zeros(400,NF); xf[:,f]=torch.rand(400)
        with torch.no_grad(): g1=c1.gates(xf); h=c1(xf); g2=c2.gates(h)
        resp[f]=torch.cat([g1,g2],1).mean(0).numpy()
    alive=np.where(resp.max(0)>0.02)[0]; sel=resp.max(0)/(resp.sum(0)+1e-9)
    return round(float(L_faith.item()),2), int((sel[alive]>0.5).sum()), len(alive)

rows=[]
for imp in [3e-2,6e-2]:
    f_no,s_no,a_no=run(imp,safeguard=False)
    f_sg,s_sg,a_sg=run(imp,safeguard=True)
    print(f"imp_max={imp}: NO-safeguard faith={f_no:.2e} spec={s_no}/{a_no} | SAFEGUARD faith={f_sg:.2e} spec={s_sg}/{a_sg}")
    rows.append({"imp_max":imp,"no_safeguard":{"faith":f_no,"spec":f"{s_no}/{a_no}"},
                 "safeguard":{"faith":f_sg,"spec":f"{s_sg}/{a_sg}"}})
json.dump(rows,open("faithsafe_results.json","w"),indent=2); print("saved")
