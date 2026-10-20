"""One closed-loop-only frozen-candidate run; durable reservations and sealing."""
import csv
from dataclasses import asdict
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import torch
from flybrain.flight3d.anatomy import CACHE,digest
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write,verify_hashes
# Supplies the frozen helper's missing stdlib math without changing source bytes.
from flybrain.escape_recovery import execution as corrected_execution
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.turnflee.protocol import closed_cases,closed_gate,correct_side
from .acceptance import ART,verify_exception
from .metrics import physical_metrics
from .records import setup_case,advance_record,store_interval,save_actions

RUN=Path('experiments/escape-flight-validation-2026-10-08/run')
CSV_FIELDS=['phase','azimuth','state','intervention','seed','interval','end_seconds',
    'selected_side','escape_state','triggers','position','orientation','velocity','foot_contact',
    'predator_position','collided','recovered','max_downstream_hz']


class FlightBudget(ReservationJournal):
    """Exactly 14,400 paired intervals, reserving 40 individual 1 ms steps each."""
    def __init__(self,path:Path):super().__init__(path,576000)


def case_order() -> list[dict]:
    """Retain all 48 original cases in their original order."""
    return closed_cases()


def protocol() -> dict:
    """No new directional, calibration or learning exposure is authorized."""
    return dict(hypothesis='The accepted frozen directional controller supports reliable turn, takeoff where required, flee, weak-threat braking and stable recovery.',
        closed_loop=case_order(),closed_seconds=6,max_cases=48,max_seconds=288,max_kernel_steps=576000,
        intact_minimum=22,per_state_minimum=7,visual_silent_maximum_triggers=0,
        stop='source/receipt mismatch, missing measurement, nonfinite state, downstream rate >240 Hz, I/O failure or budget breach; otherwise finish 48 and score once; no retry',
        physical_criteria=dict(side_deadline_seconds=.5,turn_degrees=90,turn_tolerance_degrees=3,
            resting_height_m=.008,resting_intervals=3,separation_gain_m=.02,
            braking_speed_m_s=.02,recovery_intervals=15,recovery_height_m=[.075,.125]))


def source_hashes() -> dict[str,str]:
    """Bind the adjacent experiment, audit, supervisor and activation source."""
    return {p.as_posix():digest(p) for p in sorted(Path('src/flybrain/escape_flight_validation').glob('*.py'))}


def consume(directory:Path) -> None:
    """Exclusive parent marker rejects renamed destinations and consumed resumes."""
    if directory.exists():raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True)
    write(directory.parent/'flight-validation-once.json',dict(directory=str(directory.resolve()),consumed=True,retry_authorized=False))
    directory.mkdir()


def summary(case:dict,records:list[dict]) -> dict:
    """Combine timely neural selection with actual trajectory measurements."""
    onset=next((r['end_seconds'] for r in records if r['threat']['strong']),None)
    first=next((r for r in records if r['triggers']),None)
    trigger=first['end_seconds'] if first else None;side=first['selected_side'] if first else None
    metrics=physical_metrics(case,trigger,side,records)
    timely=bool(trigger is not None and onset is not None and 0<=trigger-onset<=.5+1e-12 and correct_side(case['azimuth'],side))
    return dict(**case,first_strong_seconds=onset,trigger_seconds=trigger,side=side,
        baseline_equal=None,baseline_label='previous directional evidence only',
        triggers=records[-1]['triggers'],finite=all(r['finite'] for r in records),
        max_downstream_hz=max(r['max_downstream_hz'] for r in records),timely_correct_side=timely,
        **metrics,success=bool(timely and metrics['physical_success']))


def closed_presentation(case:dict,candidate:CandidateConfig,trace,writer,budget,bound:dict) -> dict:
    """Advance exactly the retained six-second case, without directional run()."""
    session=setup_case(case,candidate);trajectory=[]
    for interval in range(300):
        verify_hashes(bound);budget.reserve_interval()
        record=advance_record(session,case,interval)
        trajectory.append(record);store_interval(trace,writer,record)
    actions=Path(trace.name).parent/'actions';actions.mkdir(exist_ok=True)
    save_actions(session,case,actions/f'{case["seed"]}-{case["state"]}-{case["intervention"]}.json')
    return summary(case,trajectory)


def preflight() -> dict:
    """Require fresh engineering proofs and the prospective fixed exception."""
    exception=json.loads((ART/'prototype-exception.json').read_text())
    candidate=verify_exception(exception)
    engineering=json.loads((ART/'acceptance.json').read_text())
    if engineering['status']!='passed':raise ValueError('engineering acceptance missing')
    verify_hashes(engineering['source_hashes']);verify_hashes(engineering['verification_hashes'])
    if engineering['source_hashes']!=source_hashes():raise ValueError('adjacent inventory mismatch')
    protected=json.loads((ART/'protected.json').read_text());verify_hashes(protected)
    return dict(schema='escape-flight-validation-v1',**protocol(),candidate=asdict(candidate),
        candidate_identity=candidate.identity,source_sha256=source_hashes(),protected=protected,
        exception=exception,exception_sha256=digest(ART/'prototype-exception.json'),
        acceptance_sha256=digest(ART/'acceptance.json'),
        python=platform.python_version(),platform=platform.platform(),device='cpu',commit=None,
        software={n:importlib.metadata.version(n) for n in ('torch','numpy','panda3d','pygame')},
        seed=[c['seed'] for c in case_order()],cells=589,anatomical_edges=4139,
        optimizer_updates=0,learning_rate=None,checkpoint=None)


def seal(directory:Path,result:dict,candidate_identity:str|None) -> None:
    """Write terminal outputs once; detach supervisor seals missing receipts."""
    write(directory/'results.json',result)
    outputs={p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
    write(directory/'receipt.json',dict(schema='escape-flight-receipt-v1',status=result['status'],
        candidate_identity=candidate_identity,outputs=outputs,retry_authorized=False))
    write(directory.parent/'flight-validation-seal.json',dict(status=result['status'],receipt_sha256=digest(directory/'receipt.json')))


def run(directory:Path=RUN) -> dict:
    """Consume one fixed physical experiment, sealing pass/fail/interruption."""
    torch.set_num_threads(1);config=preflight();consume(directory)
    closed=[];budget=None;status='interrupted';error=None;recorded=0
    try:
        write(directory/'config.json',config)
        candidate=CandidateConfig.from_record(config['candidate'])
        for name,h in {**dict(candidate.source_sha256),**config['source_sha256']}.items():
            archive=directory/'sources'/name;archive.parent.mkdir(parents=True,exist_ok=True)
            archive.write_bytes(Path(name).read_bytes())
            if digest(archive)!=h:raise ValueError('source archive mismatch')
        (directory/'circuit.json').write_bytes((CACHE/'circuit.json').read_bytes())
        budget=FlightBudget(directory/'progress.json')
        bound={**dict(candidate.source_sha256),**config['source_sha256'],
            str(ART/'prototype-exception.json'):config['exception_sha256'],
            str(ART/'acceptance.json'):config['acceptance_sha256'],**config['exception']['bound_files']}
        with gzip.open(directory/'traces.jsonl.gz','xt',encoding='utf-8') as trace,(directory/'traces.csv').open('x',newline='',encoding='utf-8') as table:
            writer=csv.DictWriter(table,fieldnames=CSV_FIELDS);writer.writeheader()
            for case in config['closed_loop']:
                candidate.verify()
                closed.append(closed_presentation(case,candidate,trace,writer,budget,bound));recorded+=300
                table.flush()
                print(f'closed {len(closed)}/48 | intact {sum(r["success"] for r in closed if r["intervention"]=="intact")} | {case["azimuth"]} {case["state"]} {case["intervention"]}',flush=True)
        status='passed' if closed_gate(closed) else 'closed-physical-failed'
        verify_hashes(config['protected']);verify_exception(config['exception']);candidate.verify()
    except BaseException as exc:
        status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if budget:budget.close()
        result=dict(status=status,error=error,closed_loop=closed,cases=len(closed),recorded_intervals=recorded,
            reserved_kernel_steps=budget.steps if budget else 0,simulated_seconds=(budget.steps/2000 if budget else 0),
            intact_successes=sum(r['success'] for r in closed if r['intervention']=='intact'),
            per_state={s:sum(r['success'] for r in closed if r['intervention']=='intact' and r['state']==s) for s in ('resting','hovering','cruising')},
            silent_triggers=sum(r['triggers'] for r in closed if r['intervention']=='visual-silent'),
            optimizer_updates=0,retry_authorized=False)
        seal(directory,result,config['candidate_identity'])
    return result


if __name__=='__main__':
    result=run();print(json.dumps({k:v for k,v in result.items() if k!='closed_loop'},indent=2))
