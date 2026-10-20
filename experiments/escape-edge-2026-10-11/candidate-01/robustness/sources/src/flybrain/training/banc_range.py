"""Range-interface software checks; scientific entry failed and is closed."""

from dataclasses import asdict
from pathlib import Path

import torch

from flybrain.neural.banc_range import make_range_controller
from flybrain.neural.banc_projection import ProjectionReadout
from .banc_coverage import CoverageRecorder
from .banc_readiness import weights_sha256
from .evaluation import locked_cases,run_episode,summarize


class RangeRecorder(CoverageRecorder):
    """Separate range shadow; old fixed controller retains every body decision."""

    def __init__(self,model,case_id: str,trajectory: str) -> None:
        super().__init__(model,case_id,trajectory)
        self.shadow=make_range_controller(ProjectionReadout(torch.zeros(20),torch.ones(20)))

    def reset(self) -> None:
        super().reset();self.rows=[];self.shadow.reset()

    def decide(self,observation):
        decision=super().decide(observation);self.shadow.decide(observation)
        self.rows[-1]['old_rates']=self.rows[-1]['rates']
        self.rows[-1]['rates']=self.shadow.decoder.feature_rates.tolist()
        return decision


def evaluate_models(trained,reloaded,untrained,cases,config):
    """Tiny numerical matched-control check; no scientific range fit exists."""
    if not cases or any(c not in locked_cases()['train'] for c in cases):raise ValueError('original training cases required')
    controllers=[];weights=[];episodes={}
    with torch.no_grad():
        for name,model in (('trained',reloaded),('untrained',untrained),('fixed',None)):
            group=[]
            for case in cases:
                c=make_range_controller(model);controllers.append(c);weights.append(weights_sha256(c))
                group.append(run_episode(case,config,lambda:c))
            episodes[name]=group
        c=make_range_controller(trained);controllers.append(c);weights.append(weights_sha256(c))
        replay=run_episode(cases[0],config,lambda:c)
    ticks=sum(r.ticks for group in episodes.values() for r in group)+replay.ticks
    return {'episodes':{n:[asdict(r) for r in group] for n,group in episodes.items()},
        'summaries':{n:summarize(group) for n,group in episodes.items()},'replay_episode':asdict(replay),
        'actual_episodes':len(controllers),'actual_world_ticks':ticks,
        'actual_neural_steps':sum(c.network.population.step_count for c in controllers),
        'checkpoint_replay_exact':replay==episodes['trained'][0],
        'effective_weights_unchanged':len(set(weights))==1 and all(weights_sha256(c)==h for c,h in zip(controllers,weights))}


def _closed(path: Path):
    if any((Path(path).parent/n).exists() for n in ('execution.json','results.json','samples.csv','trained.pt')):
        raise FileExistsError('range attempt consumed')
    raise RuntimeError('Range entry failed the declared contrast gate; collection and fitting are prohibited. Preserve the closed assay.')


def collect(path: Path):
    """Fail closed: no scientific range collection is permitted after entry failure."""
    return _closed(path)


def run_training(path: Path):
    """Fail closed: no range optimizer run is permitted after entry failure."""
    return _closed(path)
