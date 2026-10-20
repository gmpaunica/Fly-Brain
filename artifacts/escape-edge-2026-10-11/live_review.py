"""Native viewer API review with a separate 24,000-step engineering budget.

This exercises the real viewer spawn method and renders actual simulation frames.
It does not claim native mouse/keyboard automation, which is unavailable.
"""
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
from flybrain.escape_edge.delivery import verified_candidate
from flybrain.escape_edge.viewer import EdgeViewer
from flybrain.escape_edge.session import EdgeSession
from flybrain.escape_edge.validation import ART
from flybrain.flight3d.activity import SpikeRecorder
from flybrain.retreat3d.pump import InteractionPump
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write

candidate = verified_candidate()
assert candidate is not None
write(ART/'live-review-budget.json',dict(classification='engineering viewer API review',
    seed=1110000,initial_state='resting',seconds=12,max_individual_neural_steps=24000,
    native_input_automation=False,new_scientific_cases=0))
app = EdgeViewer(1110000,candidate,verified=True)
budget = ReservationJournal(ART/'live-review-progress.json',24000)
try:
    app.paused = True
    app.simulation.close()
    app.session = EdgeSession(1110000,candidate,'resting')
    app.session.verified_default = True
    app.session.snapshot = app.session._snapshot()
    app.simulation = InteractionPump(app.session,SpikeRecorder(app.session.snapshot))
    app.simulation.pause()
    app.spawn('predator')
    app.simulation.pause()
    rows = []
    for i in range(600):
        budget.reserve_interval()
        app.simulation.advance_fixed(1)
        s = app.session.snapshot
        assert app.session.pair.finite and app.session.pair.max_downstream_hz <= 240
        rows.append(dict(time=s.time,state=s.escape_state,position=list(s.position),
                         velocity=list(s.velocity),crashed=s.crashed))
        if i in (49,149,299,599):
            app.capture_paused(ART/f'live-{i+1:03}.png')
            print(f'Live viewer API: {s.time:.2f}s | {s.escape_state} | {s.position}',flush=True)
    s = app.session.snapshot
    assert s.escape_state=='edge_hover' and not s.crashed
    assert app.session.brain.steps + app.session.pair.reference.steps == budget.steps == 24000
    assert abs(s.time-12.) < 1e-12
    assert all(r['state']=='edge_hover' and not r['crashed'] for r in rows[-100:])
    assert min(*s.position[:2],.6-s.position[0],.6-s.position[1]) >= .005
    app.session.log.save(ART/'live-actions.json')
    write(ART/'live-review.json',dict(status='passed',classification='native viewer API review',
        native_input_automation=False,capture='Panda3D native window rendering',neural_steps=budget.steps,
        new_scientific_cases=0,trajectory=rows,final_state=s.escape_state))
    budget.close()
    print('Native viewer remains paused at verified edge hover; R starts a fresh trial.',flush=True)
    app.run()
finally:
    if not budget.stream.closed:
        budget.close()
    app.destroy()
