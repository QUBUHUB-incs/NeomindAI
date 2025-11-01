# =====================================================
# NeoMind Micro – Phone / Low-end device version
# =====================================================

import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, random, glob, time, re

# ------------------------------
# 1️⃣ Minimal repo list
# ------------------------------
REPO_URLS = [
    "https://github.com/Web4application/EDQ-AI",
    "https://github.com/Web4application/Brain"
]

def clone_repos(base_dir="repos"):
    os.makedirs(base_dir, exist_ok=True)
    for url in REPO_URLS:
        name = url.rstrip("/").split("/")[-1]
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            os.system(f"git clone {url} {path}")
        else:
            os.system(f"cd {path} && git pull")
clone_repos()

# ------------------------------
# 2️⃣ Numeric features only
# ------------------------------
def extract_numeric(repo_path):
    features=[]
    for r,_,files in os.walk(repo_path):
        for f in files:
            if f.endswith(('.txt','.md','.py','.json')):
                try:
                    content=open(os.path.join(r,f),'r',errors='ignore').read()
                    features.append([content.count('\n'), len(content.split()), len(content)])
                except: pass
    return features

def build_numeric_data():
    data=[]
    for url in REPO_URLS:
        name=url.rstrip("/").split("/")[-1]
        path=os.path.join("repos", name)
        feats = extract_numeric(path)
        for f in feats:
            vec = f + [0]*(8-len(f)) if len(f)<8 else f[:8]
            data.append(vec)
    return torch.tensor(data, dtype=torch.float)

numeric_data = build_numeric_data()

# ------------------------------
# 3️⃣ Targets (self-supervised)
# ------------------------------
targets = numeric_data.clone()

# ------------------------------
# 4️⃣ Micro NeoMind network
# ------------------------------
class NeoMindMicro(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8,16)
        self.l2 = nn.Linear(16,8)
    def forward(self,x):
        x = F.gelu(self.l1(x))
        return self.l2(x)

# ------------------------------
# 5️⃣ Versioned weight saving
# ------------------------------
def save_weights(model,base="NeoMind_micro"):
    existing=glob.glob(f"{base}_v*.pth")
    if existing:
        versions=[int(re.search(r'_v(\d+)\.pth',f).group(1)) for f in existing]
        new_v=max(versions)+1
    else:
        new_v=1
    fn=f"{base}_v{new_v}.pth"
    torch.save(model.state_dict(),fn)
    print(f"✅ Weights saved: {fn}")
    return fn

# ------------------------------
# 6️⃣ Training
# ------------------------------
def train(model,X,Y,epochs=5,batch_size=4,lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    n = X.size(0)
    idx = list(range(n))
    for ep in range(epochs):
        random.shuffle(idx)
        epoch_loss=0
        for i in range(0,n,batch_size):
            b = idx[i:i+batch_size]
            xb = X[b].to(device)
            yb = Y[b].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out,yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()*xb.size(0)
        print(f"Epoch {ep+1} | Loss: {epoch_loss/n:.6f}")
    save_weights(model)
    return model

# ------------------------------
# 7️⃣ Self-updating loop
# ------------------------------
CHECK_INTERVAL = 600
def update_loop(model,X,Y):
    while True:
        updated=False
        for url in REPO_URLS:
            name=url.rstrip("/").split("/")[-1]
            path=os.path.join("repos",name)
            pull=os.system(f"cd {path} && git pull")
            if pull==0: updated=True
        if updated:
            print("🔄 Changes detected. Retraining Micro NeoMind...")
            numeric_data_new = build_numeric_data()
            train(model,numeric_data_new,numeric_data_new,epochs=3)
        else:
            print("⏳ No changes detected.")
        time.sleep(CHECK_INTERVAL)

# ------------------------------
# 8️⃣ Entrypoint
# ------------------------------
if __name__=="__main__":
    model = NeoMindMicro()
    model = train(model,numeric_data,targets,epochs=5)
    update_loop(model,numeric_data,targets)
