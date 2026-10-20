"""One predeclared, capped approach-and-stop readout fit on training cases."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import torch
from torch import Tensor
from torch.nn import functional as F

from flybrain.neural.banc_motor import BancMotorDecoder, make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.food_contact import FoodContactNetwork
from flybrain.neural.food_readout import (FoodDecoder, FoodReadout, food_contract,
                                          food_normalization, load_food_checkpoint,
                                          make_food_controller, save_food_checkpoint)
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly
from flybrain.world.food import FoodPatch, sample_food
from flybrain.world.odor import OdorSource

from .food_evaluation import run_controller_case
from .food_preflight import (DEFAULT_OUTPUT as PREFLIGHT, HORIZON_TICKS,
                             PROJECT_ROOT, SOURCE_X, SOURCE_Y, generate_cases)
from .food_teacher import FoodTeacher


DEFAULT_OUTPUT = PROJECT_ROOT / 'experiments/food-stop-fit-2026-09-28'
MAX_UPDATES = 1500
LEARNING_RATE = .001
BATCH_SIZE = 512
SEED = 1010


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_readout(model: FoodReadout, features: Tensor, targets: Tensor,
                *, updates: int = MAX_UPDATES) -> dict:
    """Adam on balanced moving/contact batches; only readout tensors update."""
    if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
        raise ValueError('updates must be in 1..1500')
    if (not isinstance(features, Tensor) or features.ndim != 2
            or features.shape[1] != 22 or not isinstance(targets, Tensor)
            or targets.shape != (len(features), 3)
            or not torch.isfinite(targets).all()
            or torch.any(targets[:, :2].abs() > 1)
            or torch.any((targets[:, 2] != 0) & (targets[:, 2] != 1))):
        raise ValueError('finite [N,22] neural rates and [N,3] targets required')
    model(features[:1])
    moving = torch.where(targets[:, 2] == 0)[0]
    stopped = torch.where(targets[:, 2] == 1)[0]
    if not len(moving) or not len(stopped):
        raise ValueError('moving and stopped training rows required')
    x, y = features.detach().float(), targets.detach().float()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    curve = []
    completed = nonzero = 0
    failure = None
    for update in range(1, updates + 1):
        moving_index = moving[torch.randint(len(moving), (BATCH_SIZE // 2,), generator=generator)]
        stop_index = stopped[torch.randint(len(stopped), (BATCH_SIZE // 2,), generator=generator)]
        batch = torch.cat((moving_index, stop_index))
        output = model(x[batch])
        speed_error = F.mse_loss(output[:BATCH_SIZE // 2, 0], y[moving_index, 0])
        turn_error = F.mse_loss(output[:BATCH_SIZE // 2, 1], y[moving_index, 1])
        stop_loss = F.binary_cross_entropy_with_logits(output[:, 2], y[batch, 2])
        loss = speed_error + turn_error + stop_loss
        if not torch.isfinite(loss):
            failure = 'nonfinite training loss'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            failure = 'missing or nonfinite gradients'
            break
        nonzero = max(nonzero, sum(int(torch.count_nonzero(g)) for g in gradients))
        optimizer.step()
        completed += 1
        if update == 1 or update % 100 == 0 or update == updates:
            curve.append({'update': update, 'speed_mse': float(speed_error.detach()),
                          'turn_mse': float(turn_error.detach()),
                          'balanced_stop_bce': float(stop_loss.detach()),
                          'combined_loss': float(loss.detach())})
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            failure = 'nonfinite readout parameter'
            break
    changed = sum(int(torch.count_nonzero(parameter.detach() != old))
                  for parameter, old in zip(model.parameters(), initial))
    return {'optimizer_steps': completed, 'gradient_nonzero_scalars': nonzero,
            'parameter_changed_scalars': changed, 'failure': failure, 'curve': curve}


def prepare_fit(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    """Lock data, hypothesis, gates, budget and source provenance before fitting."""
    if output_dir.exists():
        raise FileExistsError('fit output already exists')
    preflight = json.loads((PREFLIGHT / 'results.json').read_text(encoding='utf-8'))
    if preflight.get('passed') is not True or preflight.get('successes') != 40:
        raise ValueError('40/40 teacher preflight required')
    sources = ('src/flybrain/neural/food_contact.py',
               'src/flybrain/neural/food_readout.py',
               'src/flybrain/training/food_teacher.py',
               'src/flybrain/training/food_preflight.py',
               'src/flybrain/training/food_evaluation.py',
               'src/flybrain/training/food_fit.py')
    record = {
        'schema': 'food_stop_fit_v1',
        'hypothesis': 'One fixed 85-cell circuit and a 419-parameter neural-history readout can imitate sensory-only variable speed, steering and contact stop on the declared training starts.',
        'cases': {split: [asdict(case) for case in group]
                  for split, group in generate_cases().items()},
        'collection': 'All 40 teacher and 40 fixed-decoder 20-second trajectories; teacher labels use sensory observation only. No validation/final rows.',
        'normalization': 'Training-only conditioned feature mean and population std, scale floor 1 Hz.',
        'objective': 'Moving speed MSE + moving turn MSE + equal-class stop BCE; 256 moving and 256 stopped rows per update.',
        'seed': SEED, 'optimizer': 'Adam', 'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE, 'max_updates': MAX_UPDATES,
        'network_size': 85, 'trainable_parameters': 419,
        'training_gates': {'speed_mse_below_constant': True,
                           'turn_mse_below_constant': True,
                           'balanced_stop_accuracy_min': .95,
                           'train_successes_min': 35,
                           'train_paired_gain_min': 8,
                           'checkpoint_reload_exact': True},
        'budget': {'data_episodes': 80, 'training_case_episodes': 120,
                   'world_ticks': 200000, 'neural_steps': 4000000,
                   'optimizer_updates': MAX_UPDATES},
        'stop_condition': 'One seed-1010 candidate, at most 1500 updates. Stop on nonfinite values or any numerical or matched training gate failure. No fresh validation or retry after failure.',
        'preflight_sha256': {name: _digest(PREFLIGHT / name)
                             for name in ('protocol.json', 'execution.json', 'results.json')},
        'source_sha256': {name: _digest(PROJECT_ROOT / name) for name in sources},
        'contract': food_contract(),
        'software_device': {'python': platform.python_version(),
                            'torch': torch.__version__, 'device': 'CPU'},
        'commit': None, 'prepared_utc': datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True)
    path = output_dir / 'protocol.json'
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path


def _collect_case(case, *, teacher_drives: bool,
                  features: list[Tensor], targets: list[tuple[float, float, float]]) -> None:
    """Step fixed neurons on pre-movement sensory inputs, then move the body."""
    network = FoodContactNetwork()
    feature_decoder = FoodDecoder(network.topology,
                                  FoodReadout(torch.zeros(22), torch.ones(22), seed=SEED))
    fixed_decoder = BancMotorDecoder(network.topology)
    teacher = FoodTeacher()
    source = OdorSource(SOURCE_X, SOURCE_Y)
    patch = FoodPatch(SOURCE_X, SOURCE_Y)
    world = FixedStepSimulation(Arena(), Fly(case.x, case.y, case.heading),
                                odor_source=source)
    for _ in range(HORIZON_TICKS):
        observation = sample_food(world.fly, source, patch)
        activity = network.advance(observation)
        rates = feature_decoder.observe(activity)
        target = teacher.command(observation)
        features.append(rates.float())
        targets.append((target.forward, target.turn, float(observation.food_contact)))
        command = target if teacher_drives else fixed_decoder.decode(activity)
        world.step(command)
    if network.population.step_count != HORIZON_TICKS * 20:
        raise ValueError('neural and world clocks differ')


def collect_training_data() -> tuple[Tensor, Tensor]:
    """Collect exactly 80 training trajectories without future split access."""
    features: list[Tensor] = []
    targets: list[tuple[float, float, float]] = []
    cases = generate_cases()['train']
    with torch.no_grad():
        for case in cases:
            _collect_case(case, teacher_drives=True, features=features, targets=targets)
        for case in cases:
            _collect_case(case, teacher_drives=False, features=features, targets=targets)
    return torch.stack(features), torch.tensor(targets, dtype=torch.float32)


def _training_metrics(model: FoodReadout, features: Tensor, targets: Tensor) -> dict:
    with torch.no_grad():
        output = model(features)
    moving = targets[:, 2] == 0
    stopped = ~moving
    errors = ((output[moving, :2] - targets[moving, :2]) ** 2).mean(0)
    means = targets[moving, :2].mean(0)
    baseline = ((targets[moving, :2] - means) ** 2).mean(0)
    stop_accuracy = ((output[stopped, 2] > 0).float().mean()
                     + (output[moving, 2] <= 0).float().mean()) / 2
    return {'moving_rows': int(moving.sum()), 'stopped_rows': int(stopped.sum()),
            'speed_mse': float(errors[0]), 'turn_mse': float(errors[1]),
            'constant_speed_mse': float(baseline[0]),
            'constant_turn_mse': float(baseline[1]),
            'balanced_stop_accuracy': float(stop_accuracy),
            'mean_moving_command': means.tolist()}


def run_fit(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Consume the single fit; record failure and stop before any fresh case."""
    path = output_dir / 'protocol.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if (protocol.get('schema') != 'food_stop_fit_v1'
            or protocol.get('contract') != food_contract()
            or protocol.get('cases') != {split: [asdict(case) for case in group]
                                         for split, group in generate_cases().items()}
            or any(_digest(PROJECT_ROOT / name) != digest
                   for name, digest in protocol['source_sha256'].items())
            or any(_digest(PREFLIGHT / name) != digest
                   for name, digest in protocol['preflight_sha256'].items())):
        raise ValueError('fit protocol or accepted preflight differs')
    with (output_dir / 'execution.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'consumed', 'protocol_sha256': _digest(path),
                   'started_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _execute_fit(output_dir, protocol, _digest(path))
    finally:
        torch.set_num_threads(previous_threads)


def _execute_fit(output_dir: Path, protocol: dict, digest: str) -> dict:
    features, targets = collect_training_data()
    with (output_dir / 'data.pt').open('xb') as stream:
        torch.save({'features': features, 'targets': targets}, stream)
    mean, scale = food_normalization(features)
    model = FoodReadout(mean, scale, seed=SEED)
    untrained = FoodReadout(mean, scale, seed=SEED)
    fit = fit_readout(model, features, targets)
    metrics = _training_metrics(model, features, targets)
    numerical_pass = (fit['failure'] is None and fit['optimizer_steps'] == MAX_UPDATES
                      and fit['gradient_nonzero_scalars'] > 0
                      and fit['parameter_changed_scalars'] > 0
                      and metrics['speed_mse'] < metrics['constant_speed_mse']
                      and metrics['turn_mse'] < metrics['constant_turn_mse']
                      and metrics['balanced_stop_accuracy'] >= .95)
    save_food_checkpoint(output_dir / 'trained.pt', model, protocol_sha256=digest)
    save_food_checkpoint(output_dir / 'untrained.pt', untrained, protocol_sha256=digest)
    reloaded = load_food_checkpoint(output_dir / 'trained.pt', protocol_sha256=digest,
                                    mean=mean, scale=scale)
    with torch.no_grad():
        reload_exact = torch.equal(model(features[:64]), reloaded(features[:64]))
    result = {'protocol_sha256': digest, 'dataset_sha256': _digest(output_dir / 'data.pt'),
              'checkpoint_sha256': _digest(output_dir / 'trained.pt'),
              'untrained_sha256': _digest(output_dir / 'untrained.pt'),
              'normalization': {'mean': mean.tolist(), 'scale': scale.tolist()},
              'sample_count': len(features), 'positive_contact_rows': int(targets[:, 2].sum()),
              'fit': fit, 'train_metrics': metrics,
              'checkpoint_reload_exact': reload_exact,
              'numerical_gate_passed': bool(numerical_pass and reload_exact),
              'actual_data_episodes': 80, 'actual_data_world_ticks': 80000,
              'actual_data_neural_steps': 1600000,
              'training_cases': None, 'training_gate_passed': False,
              'fresh_evaluated': False}
    if result['numerical_gate_passed']:
        model.requires_grad_(False)
        untrained.requires_grad_(False)
        cases = generate_cases()['train']
        config = load_verified_banc_config()
        episodes = {}
        with torch.no_grad():
            for name, factory in (
                ('trained', lambda: make_food_controller(model)),
                ('untrained', lambda: make_food_controller(untrained)),
                ('fixed', lambda: make_banc_controller(config)),
            ):
                episodes[name] = [run_controller_case(case, factory()) for case in cases]
        successes = {name: sum(row['success'] for row in rows)
                     for name, rows in episodes.items()}
        result['training_cases'] = {'successes': successes, 'episodes': episodes}
        result['training_gate_passed'] = bool(
            successes['trained'] >= 35
            and all(successes['trained'] - successes[control] >= 8
                    for control in ('untrained', 'fixed')))
    (output_dir / 'results.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
