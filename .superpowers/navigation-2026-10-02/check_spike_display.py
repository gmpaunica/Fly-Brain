"""Two-second preview display diagnostic; no fitting or held-out evaluation."""
import json,platform
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from flybrain.flight3d.session import FlightSession
from flybrain.flight3d.world import make_trial
from flybrain.flight3d.anatomy import CACHE
from flybrain.flight3d.anatomical_display import NeuronIllumination

seed=970000;updates=100
session=FlightSession(make_trial(seed),'teacher')
ids=session.brain.ids
nodes={n['id']:n for n in session.manifest['nodes']}
allowed=np.array([(CACHE/'skeletons'/f'{id}.npz').exists() and nodes[id]['metadata']['region']!='ventral_nerve_cord' for id in ids])
display=NeuronIllumination(ids,session.brain.edge_order)
spikes=np.zeros(len(ids),dtype=int);highlights=np.zeros(len(ids),dtype=int)
spiking_hidden_intervals=np.zeros(len(ids),dtype=int);index={id:i for i,id in enumerate(ids)}
frames=[]
for _ in range(updates):
 snapshot=session.advance();counts=np.array(snapshot.spike_counts);spikes+=counts
 selected=display.update(seed,snapshot.time,snapshot.edge_activity,allowed)
 chosen=np.zeros(len(ids),dtype=bool)
 for id,_ in selected:chosen[index[id]]=True
 highlights+=chosen
 spiking_hidden_intervals+=(counts>0)&allowed&~chosen
 frames.append({'time_s':snapshot.time,'spiking_neurons':int((counts>0).sum()),'visible_spiking_neurons':int(((counts>0)&allowed).sum()),'highlights':int(chosen.sum()),'visible_spiking_not_highlighted':int(((counts>0)&allowed&~chosen).sum())})
rows=[{'id':id,'roles':nodes[id]['roles'],'region':nodes[id]['metadata']['region'],'visible_eligible':bool(allowed[i]),'spikes':int(spikes[i]),'highlighted_intervals':int(highlights[i]),'spiking_but_not_highlighted_intervals':int(spiking_hidden_intervals[i]),'retained_outgoing_edges':sum(e['pre']==id for e in session.manifest['edges'])} for i,id in enumerate(ids)]
summary={'simulated_neurons':len(ids),'spiked_at_least_once':int((spikes>0).sum()),'no_spikes_in_this_window':int((spikes==0).sum()),'visible_eligible':int(allowed.sum()),'visible_spiked_at_least_once':int(((spikes>0)&allowed).sum()),'visible_silent_in_this_window':int(((spikes==0)&allowed).sum()),'visible_spiked_but_never_highlighted':int(((spikes>0)&allowed&(highlights==0)).sum()),'unique_highlighted':int((highlights>0).sum()),'maximum_simultaneously_spiking':max(f['spiking_neurons'] for f in frames),'maximum_visible_spiking_not_highlighted':max(f['visible_spiking_not_highlighted'] for f in frames)}
result={'purpose':'Compare actual spikes with sparse outgoing-transmission display; no learning claim','hypothesis':'Some project cells are silent in a preview window; some spike without a warm highlight','seed':seed,'mode':'teacher','budget_control_updates':updates,'stop_condition':'Stop after 100 fixed 20 ms updates (2 simulated seconds)','duration_s':session.snapshot.time,'optimizer_updates':0,'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','summary':summary,'neurons':rows,'frames':frames}
folder=Path('.superpowers/navigation-2026-10-02')
(folder/'spike-display-diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
