#!/usr/bin/env python3
"""Redacted W&B access checker for the IsaacLab environment."""

from __future__ import annotations

import argparse
import base64
import json
import netrc
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_HOST = "api.wandb.ai"
API_URL = "https://api.wandb.ai/graphql"
DEFAULT_PROJECT = "IsaacLab22-SurgicalRobot5-FDPI-Reachability"


def print_kv(key: str, value: Any) -> None:
    print(f"{key}={value}")


def load_wandb_key() -> str | None:
    path = Path.home() / ".netrc"
    print_kv("netrc_path", path)
    print_kv("netrc_exists", path.exists())
    if not path.exists():
        return None
    try:
        auth = netrc.netrc(str(path)).authenticators(API_HOST)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print_kv("netrc_error", f"{type(exc).__name__}: {exc}")
        return None
    if not auth:
        print_kv("netrc_host_present", False)
        return None
    login, account, password = auth
    print_kv("netrc_host_present", True)
    print_kv("netrc_login", repr(login))
    print_kv("netrc_account_present", bool(account))
    print_kv("netrc_password_len", len(password or ""))
    return password


def run_cli_verify() -> None:
    try:
        proc = subprocess.run(
            ["wandb", "login", "--verify"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        print_kv("wandb_cli_verify", "wandb command not found")
        return
    except subprocess.TimeoutExpired:
        print_kv("wandb_cli_verify", "timeout")
        return
    print_kv("wandb_cli_verify_returncode", proc.returncode)
    for line in proc.stdout.splitlines():
        if "API" in line and "key" in line.lower():
            print("wandb_cli: <redacted sensitive line>")
        else:
            print(f"wandb_cli: {line}")


def graphql(key: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    token = base64.b64encode(("api:" + key).encode()).decode()
    req.add_header("Authorization", "Basic " + token)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": resp.status, "body": json.loads(resp.read())}
    except urllib.error.HTTPError as exc:
        try:
            body_text = exc.read().decode(errors="replace")[:1000]
        except Exception:  # pragma: no cover - diagnostic path
            body_text = ""
        return {"status": exc.code, "error": f"HTTPError: {exc}", "body_text": body_text}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"status": None, "error": f"{type(exc).__name__}: {exc}"}


def print_graphql_result(label: str, result: dict[str, Any]) -> None:
    print_kv(f"{label}_status", result.get("status"))
    if result.get("error"):
        print_kv(f"{label}_error", result["error"])
    if result.get("body_text"):
        print_kv(f"{label}_error_body", result["body_text"])
    body = result.get("body")
    if isinstance(body, dict) and body.get("errors"):
        print_kv(f"{label}_graphql_errors", json.dumps(body["errors"], ensure_ascii=False)[:1000])


def check_public_api() -> None:
    try:
        from wandb.apis.public import Api

        api = Api(timeout=15)
        viewer = api.viewer
        print_kv("public_api_viewer_ok", True)
        print_kv("public_api_username", getattr(viewer, "username", None))
        print_kv("public_api_entity", getattr(viewer, "entity", None))
    except Exception as exc:
        print_kv("public_api_viewer_ok", False)
        print_kv("public_api_error", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=None, help="W&B entity. Defaults to viewer.entity.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project name.")
    parser.add_argument("--list-projects", action="store_true", help="List a small project sample.")
    parser.add_argument("--list-runs", action="store_true", help="List a small run sample for --project.")
    parser.add_argument("--run", default=None, help="Run name/id for sampled history.")
    parser.add_argument("--show-history-keys", action="store_true", help="Show available history keys for --run.")
    parser.add_argument("--history-key-filter", default="", help="Only show history keys containing this text.")
    parser.add_argument("--max-history-keys", type=int, default=120, help="Max history keys to print; use 0 for no limit.")
    parser.add_argument("--history-keys", default="", help="Comma-separated history keys for --run.")
    parser.add_argument("--samples", type=int, default=500, help="Sample count for sampledHistory.")
    parser.add_argument("--per-page", type=int, default=10, help="Page size for project/run listing.")
    parser.add_argument("--skip-cli", action="store_true", help="Skip `wandb login --verify`.")
    parser.add_argument("--check-public-api", action="store_true", help="Also test wandb.Api(); may fail while GraphQL works.")
    args = parser.parse_args()

    print_kv("python", sys.executable)
    print_kv("cwd", os.getcwd())
    try:
        import wandb

        print_kv("wandb_installed", True)
        print_kv("wandb_version", getattr(wandb, "__version__", "unknown"))
    except Exception as exc:
        print_kv("wandb_installed", False)
        print_kv("wandb_import_error", f"{type(exc).__name__}: {exc}")
        return 2

    key = load_wandb_key()
    if not args.skip_cli:
        run_cli_verify()
    if args.check_public_api:
        check_public_api()
    if not key:
        return 2

    viewer_query = """
    query Viewer {
      viewer { id username entity }
    }
    """
    viewer_result = graphql(key, viewer_query)
    print_graphql_result("viewer", viewer_result)
    viewer = (viewer_result.get("body") or {}).get("data", {}).get("viewer") or {}
    if viewer:
        print_kv("viewer_username", viewer.get("username"))
        print_kv("viewer_entity", viewer.get("entity"))
    entity = args.entity or viewer.get("entity")
    if not entity:
        print_kv("entity_resolved", False)
        return 2
    print_kv("entity", entity)

    if args.list_projects:
        projects_query = """
        query NarrowProjects($entity: String!, $perPage: Int = 10) {
          models(entityName: $entity, first: $perPage) {
            edges { node { id name entityName createdAt } }
          }
        }
        """
        projects_result = graphql(key, projects_query, {"entity": entity, "perPage": args.per_page})
        print_graphql_result("projects", projects_result)
        edges = (((projects_result.get("body") or {}).get("data") or {}).get("models") or {}).get("edges") or []
        print_kv("project_count_sample", len(edges))
        for edge in edges:
            node = edge.get("node") or {}
            print(f"project={node.get('name')} entity={node.get('entityName')} created_at={node.get('createdAt')}")

    if args.project:
        project_query = """
        query ProjectCheck($project: String!, $entity: String!) {
          project(name: $project, entityName: $entity) { id name entityName runCount }
        }
        """
        project_result = graphql(key, project_query, {"entity": entity, "project": args.project})
        print_graphql_result("project", project_result)
        project = (((project_result.get("body") or {}).get("data") or {}).get("project") or {})
        if project:
            print_kv("project_name", project.get("name"))
            print_kv("project_entity", project.get("entityName"))
            print_kv("project_run_count", project.get("runCount"))

    if args.list_runs:
        runs_query = """
        query Runs($project: String!, $entity: String!, $perPage: Int = 10, $order: String) {
          project(name: $project, entityName: $entity) {
            name
            entityName
            runCount
            runs(first: $perPage, order: $order) {
              edges {
                node { name displayName state createdAt heartbeatAt group tags historyLineCount }
              }
            }
          }
        }
        """
        variables = {"entity": entity, "project": args.project, "perPage": args.per_page, "order": "-created_at"}
        runs_result = graphql(key, runs_query, variables)
        print_graphql_result("runs", runs_result)
        edges = (((((runs_result.get("body") or {}).get("data") or {}).get("project") or {}).get("runs") or {}).get("edges") or [])
        print_kv("run_count_sample", len(edges))
        for edge in edges:
            node = edge.get("node") or {}
            tags = ",".join(node.get("tags") or [])
            print(
                "run="
                f"{node.get('name')} display={node.get('displayName')} state={node.get('state')} "
                f"created_at={node.get('createdAt')} heartbeat_at={node.get('heartbeatAt')} "
                f"group={node.get('group')} history_lines={node.get('historyLineCount')} tags={tags}"
            )

    if args.run and args.show_history_keys:
        keys_query = """
        query RunHistoryKeys($project: String!, $entity: String!, $name: String!) {
          project(name: $project, entityName: $entity) {
            run(name: $name) { name displayName state historyLineCount historyKeys }
          }
        }
        """
        variables = {"entity": entity, "project": args.project, "name": args.run}
        keys_result = graphql(key, keys_query, variables)
        print_graphql_result("history_keys", keys_result)
        run = (((keys_result.get("body") or {}).get("data") or {}).get("project") or {}).get("run") or {}
        if run:
            print_kv("history_keys_run", run.get("name"))
            print_kv("history_keys_line_count", run.get("historyLineCount"))
            history_keys = run.get("historyKeys") or {}
            keys = history_keys.get("keys", history_keys) if isinstance(history_keys, dict) else history_keys
            print_kv("history_keys_count", len(keys))
            if isinstance(keys, dict):
                items = sorted(keys.items())
                if args.history_key_filter:
                    needle = args.history_key_filter.lower()
                    items = [(metric_key, meta) for metric_key, meta in items if needle in metric_key.lower()]
                print_kv("history_keys_matched", len(items))
                if args.max_history_keys > 0:
                    items = items[: args.max_history_keys]
                print_kv("history_keys_printed", len(items))
                for metric_key, meta in items:
                    counts = meta.get("typeCounts") if isinstance(meta, dict) else None
                    count = sum(item.get("count", 0) for item in counts or [] if isinstance(item, dict))
                    previous = meta.get("previousValue") if isinstance(meta, dict) else None
                    print(f"history_key={metric_key} count={count} previous={previous}")
            else:
                items = list(keys)
                if args.history_key_filter:
                    needle = args.history_key_filter.lower()
                    items = [metric_key for metric_key in items if needle in str(metric_key).lower()]
                print_kv("history_keys_matched", len(items))
                if args.max_history_keys > 0:
                    items = items[: args.max_history_keys]
                print_kv("history_keys_printed", len(items))
                for metric_key in items:
                    print(f"history_key={metric_key}")

    if args.run and args.history_keys:
        keys = [key.strip() for key in args.history_keys.split(",") if key.strip()]
        if "_step" not in keys:
            keys.insert(0, "_step")
        spec = json.dumps({"keys": keys, "samples": args.samples})
        history_query = """
        query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
          project(name: $project, entityName: $entity) {
            run(name: $name) { sampledHistory(specs: $specs) }
          }
        }
        """
        variables = {"entity": entity, "project": args.project, "name": args.run, "specs": [spec]}
        history_result = graphql(key, history_query, variables)
        print_graphql_result("history", history_result)
        sampled = (((((history_result.get("body") or {}).get("data") or {}).get("project") or {}).get("run") or {}).get("sampledHistory") or [])
        rows = sampled[0] if sampled else []
        print_kv("history_rows", len(rows))
        if rows:
            print("history_first=" + json.dumps(rows[0], ensure_ascii=False, sort_keys=True))
            print("history_last=" + json.dumps(rows[-1], ensure_ascii=False, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
