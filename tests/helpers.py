"""Round-trip helpers for tests; production uses enable_tacache directly."""
from types import SimpleNamespace
from tacache import enable_tacache


def calibrate(model, generate, config):
    with enable_tacache(model, 'calibrate', config) as cache:
        output = generate()
    return SimpleNamespace(output=output, **cache.last_calibration_result)


def infer(model, generate, skip_path, config):
    with enable_tacache(model, 'infer', config, skip_table=skip_path) as cache:
        output = generate()
    return SimpleNamespace(output=output, skip_count=cache.report['skip_count'])
