#!/usr/bin/env python3
import argparse
import json
import os
import sys

from hermes_cli.savana_evolution_guard import rollback_guarded_evolution


def build_parser():
    parser = argparse.ArgumentParser(
        description="Rollback one committed guarded Savana persona evolution."
    )
    parser.add_argument(
        "--hermes-home",
        default=os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--audit-id", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = rollback_guarded_evolution(
            args.hermes_home,
            args.profile_id,
            args.audit_id,
        )
    except Exception as exc:
        sys.stderr.write("Rollback failed: {0}\n".format(exc))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
