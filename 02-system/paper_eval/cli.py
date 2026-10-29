from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import evaluate_pdf
from .rules import load_rule_profile, validate_rule_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperEval on a research paper.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-rules")
    validate.add_argument("--rules", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--pdf", required=True)
    evaluate.add_argument("--rules", required=True)
    evaluate.add_argument("--out", default=str(Path("runs")))
    evaluate.add_argument("--index")
    evaluate.add_argument("--topic")
    evaluate.add_argument("--db")
    evaluate.add_argument("--agent-config")

    args = parser.parse_args()
    if args.command == "validate-rules":
        profile = load_rule_profile(args.rules)
        errors = validate_rule_profile(profile)
        result = {
            "ok": not errors,
            "errors": errors,
            "dimensions": len(profile.dimensions),
            "rules": len(profile.rules),
        }
    else:
        result = evaluate_pdf(
            args.pdf,
            args.rules,
            args.out,
            index_path=args.index,
            user_topic=args.topic,
            db_path=args.db,
            agent_config_path=args.agent_config,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
