"""Bounded static visual characterization; no fitting or optimizer."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from flybrain.flight3d.anatomy import load_manifest
from flybrain.flight3d.neural import FlightBrain

torch.set_num_threads(1)
m = load_manifest()
b = FlightBrain(m)
motors = [i for i,n in enumerate(b.nodes) if 'wing_steering' in n['roles'] or 'wing_power' in n['roles']]
dnp = [i for i,n in enumerate(b.nodes) if n['metadata'].get('cell_type') == 'DNp03']
print('motors',len(motors),'vision',len(b.roles['vision']))
for label,eye,columns in [('left',0,[3,4]),('right',1,[3,4]),('front',None,[7,0]),('rear',None,[0,7])]:
    rates=np.zeros(len(b.ids))
    for side in range(2):
        if eye is not None and side != eye: continue
        indices=[int(i) for i in b.roles['vision'] if b.nodes[i]['metadata'].get('side')==('left','right')[side]]
        col = columns if eye is not None else [columns[side]]
        for rank,i in enumerate(indices):
            if rank//6 in col and rank%6 in [2,3]: rates[i]=240
    for intervention in ('intact','dnp03'):
        b.reset();b.silenced=set() if intervention=='intact' else {b.ids[i] for i in dnp}
        b.advance_rates(rates,steps=1000)
        print(label,intervention,'inputs',int((rates>0).sum()),'DN',b.last_counts[dnp].tolist(),'motors',b.last_counts[motors].tolist())
