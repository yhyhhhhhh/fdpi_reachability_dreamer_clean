#!/usr/bin/env python3
"""检查新版本目录是否包含旧版本路径或名称残留。

示例：
python scripts/checks/check_version_dependency.py --target algorithms/method_v2 --old method_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

TEXT_SUFFIXES = {'.py', '.yaml', '.yml', '.toml', '.json', '.md', '.sh', '.txt'}


def iter_text_files(root: Path):
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description='检查新版本是否残留旧版本依赖。')
    parser.add_argument('--target', required=True, help='新版本目录，例如 algorithms/method_v2')
    parser.add_argument('--old', required=True, help='旧版本标识，例如 method_v1')
    args = parser.parse_args()

    target = Path(args.target)
    old = args.old
    if not target.exists():
        print(f'错误：目标目录不存在：{target}')
        return 2

    hits = []
    for path in iter_text_files(target):
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if old in line:
                hits.append((path, lineno, line.strip()))

    if not hits:
        print('检查通过：未发现旧版本标识残留。')
        return 0

    print('发现潜在旧版本依赖：')
    for path, lineno, line in hits:
        print(f'- {path}:{lineno}: {line}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
