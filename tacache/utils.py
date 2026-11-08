"""Paths, input/output and JSON/CLI configuration shared by model scripts."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

def project_root():
    """Resolve bundled inputs relative to this repository, independent of cwd."""
    return Path(__file__).resolve().parents[1]


def project_file(value):
    """Resolve a project-relative setting for file access."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def relative_path(value):
    """Store/display filesystem locations relative to the delivery root."""
    text = os.fspath(value)
    path = Path(text).expanduser()
    if path.is_absolute():
        return Path(os.path.relpath(path, project_root())).as_posix()
    return Path(os.path.normpath(path)).as_posix()


def normalize_path_arguments(args):
    """Normalize standard filesystem path arguments."""
    names = ("checkpoint", "outdir", "skip_table", "prompts_file", "wan_root")
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            continue
        normalized = relative_path(value)
        setattr(args, name, Path(normalized) if isinstance(value, Path) else normalized)
    return args


@contextmanager
def project_working_directory():
    """Give CLI/native loaders one consistent base and restore caller state."""
    previous = Path.cwd()
    os.chdir(project_root())
    try:
        yield
    finally:
        os.chdir(previous)


def read_prompts_file(path):
    """Read a prompt set from a .txt file: one prompt per line; blank lines and
    lines starting with '#' are ignored."""
    prompts = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            s = line.strip()
            if s and not s.startswith("#"):
                prompts.append(s)
    return prompts


def parse_configured_args(parser, argv, default_config, aliases=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--config')
    selected, _ = selector.parse_known_args(argv)
    path = project_file(selected.config or default_config)
    defaults = json.loads(project_file(default_config).read_text())
    if selected.config:
        overrides = json.loads(path.read_text())
        defaults['tacache'] = {**defaults.get('tacache', {}), **overrides.pop('tacache', {})}
        defaults.update(overrides)
    defaults.update(defaults.pop('tacache', {}))
    aliases = aliases or {}
    defaults = {aliases.get(key, key): value for key, value in defaults.items()}
    actions = {action.dest: action for action in parser._actions}
    unknown = set(defaults) - set(actions)
    if unknown:
        parser.error('Unknown config keys: ' + ', '.join(sorted(unknown)))
    for key, value in list(defaults.items()):
        if actions[key].type is not None and value is not None:
            defaults[key] = ([actions[key].type(v) for v in value]
                             if isinstance(value, list) and actions[key].nargs else actions[key].type(value))
    for group in parser._mutually_exclusive_groups:
        explicit = [action for action in group._group_actions
                    if any(arg.split('=', 1)[0] in action.option_strings for arg in argv)]
        if explicit:
            for action in group._group_actions:
                if action not in explicit:
                    defaults.pop(action.dest, None)
        else:
            supplied = [a for a in group._group_actions if defaults.get(a.dest) is not None]
            if len(supplied) > 1:
                parser.error('Config specifies mutually exclusive inputs')
            if supplied:
                group.required = False
    parser.add_argument('--config', default=str(path), help='JSON model defaults; CLI flags override it')
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def add_arguments(parser):
    """Common flags; each native model file adds its own generation options."""
    parser.add_argument("--mode", choices=("baseline", "calibrate", "infer"), default="baseline")
    parser.add_argument("--checkpoint")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompts-file", help="UTF-8 file with one prompt per line")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-table", help="Skip table produced by calibration (required for infer mode)")
    parser.add_argument("--t-blocks", type=int, default=5)
    parser.add_argument("--tau-k", type=float, default=.03)
    parser.add_argument("--tau-theta", type=float, default=.30)
    parser.add_argument("--h-cap", type=int, default=3)
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--lambda-std", type=float, default=1.)
    parser.add_argument("--quiet", action="store_true")


def cache_options(args):
    """Build cache parameters from command-line options."""
    if args.steps < 3 or args.t_blocks < 1 or args.h_cap < 1 or args.min_h < 1:
        raise ValueError("steps must be >=3; blocks and skip limits must be positive")
    return dict(output_prefix=str(args.outdir / "tacache"), t_blocks=args.t_blocks,
                tau_k=args.tau_k, tau_theta=args.tau_theta, h_cap=args.h_cap,
                min_h=args.min_h, lambda_std=args.lambda_std, verbose=not args.quiet)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
