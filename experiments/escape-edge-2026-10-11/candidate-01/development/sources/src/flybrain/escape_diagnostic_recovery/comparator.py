"""Compare complete historical records at the same finite JSON boundary."""
import json


def assert_historical_equal(runtime: dict, historical: dict) -> dict:
    """Normalize tuple/list representation, retaining every value and field.

    No tolerance, rounding, field removal or mutation of the input. The returned
    comparison copy must never be used as a simulation input or neural state.
    """
    canonical=json.loads(json.dumps(runtime,allow_nan=False))
    if canonical!=historical:
        different=sorted(k for k in canonical.keys()|historical.keys()
                         if k not in canonical or k not in historical or canonical[k]!=historical[k])
        raise ValueError(f'historical mismatch: {different}')
    return canonical
