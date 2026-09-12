"""Fixed-split manifest dataset; code never creates or reassigns splits."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class ManifestSignalDataset(Dataset):
    def __init__(self,cfg,split,mean=None,std=None,augment=False):
        d=cfg["dataset"]
        if not d["manifest"]: raise ValueError("dataset.manifest is required")
        with open(d["manifest"],newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        needed={d["path_column"],d["label_column"],d["split_column"]}
        if not rows or not needed.issubset(rows[0]): raise ValueError(f"Manifest needs columns {sorted(needed)}")
        self.rows=[r for r in rows if r[d["split_column"]]==split]
        if not self.rows: raise ValueError(f"No fixed split {split!r}")
        self.root=Path(d["root"]); self.path_col=d["path_column"]; self.label_col=d["label_column"]
        labels=sorted({r[self.label_col] for r in rows}); self.label_to_id={v:i for i,v in enumerate(labels)}
        if d["num_classes"] and len(labels)!=d["num_classes"]: raise ValueError("num_classes conflicts with manifest")
        self.mean,self.std,self.augment,self.cfg=mean,std,augment,cfg
    def _load(self,row):
        path=self.root/row[self.path_col]
        if path.suffix.lower()!=".npy": raise ValueError(f"Only one 1-D .npy signal per manifest row is supported: {path}")
        x=np.load(path).astype(np.float32).squeeze()
        if x.ndim!=1: raise ValueError(f"Expected 1-D signal: {path} has {x.shape}")
        length=self.cfg["dataset"]["signal_length"]
        if length: x=x[:length] if len(x)>=length else np.pad(x,(0,length-len(x)))
        return x
    def __getitem__(self,i):
        x=self._load(self.rows[i])
        if self.augment:
            a=self.cfg["training"]["augmentation"]
            if a["amplitude_scale"]: x*=np.random.uniform(1-a["amplitude_scale"],1+a["amplitude_scale"])
            if a["gaussian_noise_std"]: x+=np.random.normal(0,a["gaussian_noise_std"],x.shape)
        if self.mean is not None: x=(x-self.mean)/max(self.std,1e-8)
        return torch.from_numpy(x).unsqueeze(0),self.label_to_id[self.rows[i][self.label_col]]
    def __len__(self): return len(self.rows)

def training_statistics(dataset):
    values=np.concatenate([dataset._load(row) for row in dataset.rows])
    return float(values.mean()),float(values.std())
