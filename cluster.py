import torch, numpy as np, json
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tms_target import gen_batch
from spd_anneal import RankOneComponents

d=torch.load("spd_anneal.pt"); nf,nh=5,2
enc=RankOneComponents(nf,nh,20); enc.load_state_dict(d["enc"])
dec=RankOneComponents(nh,nf,20); dec.load_state_dict(d["dec"])

def signatures():
    resp=np.zeros((nf,40))
    for f in range(nf):
        xf=torch.zeros(400,nf); xf[:,f]=torch.rand(400)
        with torch.no_grad():
            ge=enc.gates(xf);h=enc(xf);gd=dec.gates(h)
        resp[f]=torch.cat([ge,gd],1).mean(0).numpy()
    return resp
resp=signatures(); gt=resp.argmax(0)
alive=np.where(resp.max(0)>0.02)[0]; gt_a=gt[alive]
print(f"alive components: {len(alive)}/40")

X=gen_batch(4000,nf,0.05)
with torch.no_grad():
    ge=enc.gates(X);h=enc(X);gd=dec.gates(h)
G=torch.cat([ge,gd],1).numpy()[:,alive]
d1=1-np.nan_to_num(np.corrcoef(G.T))
S=resp[:,alive].T; S=S/(np.linalg.norm(S,axis=1,keepdims=True)+1e-9); d2=1-np.clip(S@S.T,-1,1)
# H1+H2 fused
df=0.5*d1+0.5*d2

def run(dist,name):
    lab=AgglomerativeClustering(n_clusters=nf,metric="precomputed",linkage="average").fit_predict(dist)
    ari=adjusted_rand_score(gt_a,lab); nmi=normalized_mutual_info_score(gt_a,lab)
    print(f"  {name:34s} ARI={ari:.3f} NMI={nmi:.3f}")
    return round(ari,3),round(nmi,3)

print("\n=== Annealed decomposition — clustering recovery ===")
r1=run(d1,"H1 co-activation"); r2=run(d2,"H2 attribution"); rf=run(df,"H1+H2 fused")
json.dump({"alive":int(len(alive)),"H1":r1,"H2":r2,"fused":rf},open("cluster_results.json","w"),indent=2)
print("\nsaved cluster_results.json")
