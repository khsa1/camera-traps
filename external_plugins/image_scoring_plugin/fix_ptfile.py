#!/usr/bin/env python3
import torch

device = 'cpu'
force_cpu = False
if not force_cpu:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    try:
        if torch.backends.mps.is_built and torch.backends.mps.is_available():
            device = 'mps'
    except AttributeError:
        pass
ckpt = torch.load('/md_v5a.0.0.pt', map_location=device)
for m in ckpt['model'].modules():
    if type(m) is torch.nn.Upsample:
        m.recompute_scale_factor = None
torch.save(ckpt, '/md_v5a.0.1.pt')
