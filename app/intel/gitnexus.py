"""GitNexus-style repository intelligence.

Analyzes a legacy codebase into a structured model that Graphify turns into a
knowledge graph: files, symbols (classes/functions/methods), imports, call
references, plus git history signals (hotspots, authors, last touch) when the
repo has a .git directory.

Python files get a real AST parse; other legacy languages (Java, C#, JS/TS,
C/C++, PHP, Ruby, Go) get a pragmatic regex parse - enough to build a useful
graph without per-language toolchains.
"""
import ast
import re
import subprocess
import time
from pathlib import Path

from loguru import logger

LANGUAGES = {
    ".py": "python", ".java": "java", ".cs": "csharp", ".js": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".php": "php", ".rb": "ruby", ".go": "go", ".sql": "sql",
    ".cbl": "cobol", ".cob": "cobol", ".f90": "fortran", ".for": "fortran",
    ".md": "markdown", ".txt": "text", ".yml": "yaml", ".yaml": "yaml",
    ".json": "json", ".xml": "xml", ".ini": "config", ".cfg": "config",
    ".toml": "config", ".properties": "config",
}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea",
             "dist", "build", "target", ".mypy_cache", ".pytest_cache", "bin",
             "obj", ".tox", "data", "logs"}
MAX_FILE_BYTES = 400_000

# regex parse for non-Python languages: (kind, pattern-with-name-group)
_GENERIC_PATTERNS = [
    ("class", re.compile(
        r"^\s*(?:public|private|protected|internal|abstract|final|static|export|partial|\s)*"
        r"(?:class|interface|struct|enum)\s+([A-Za-z_]\w*)", re.M)),
    ("function", re.compile(  # java/c#/c-like methods & functions
        r"^\s*(?:public|private|protected|internal|static|final|virtual|override|async|export"
        r"|function|def|func|void|int|long|float|double|bool|boolean|string|String|var|\s)+"
        r"([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{", re.M)),
    ("function", re.compile(  # js arrow / const fn
        r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\(", re.M)),
]
_GENERIC_IMPORT = re.compile(
    r"^\s*(?:import\s+([\w.{}*,\s/'\"@-]+?)(?:\s+from\s+['\"]([^'\"]+)['\"])?|"
    r"using\s+([\w.]+)|#include\s+[<\"]([^>\"]+)[>\"]|require\(['\"]([^'\"]+)['\"]\))",
    re.M)


def analyze_repo(repo_path: str) -> dict:
    """Full analysis of one repository. Returns the GitNexus model dict."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repo path not found: {root}")

    t0 = time.time()
    files, lang_counts = [], {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # skip-dirs are checked relative to the repo root, or an imported
        # repo under data/imports/ would match the "data" skip rule
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        lang = LANGUAGES.get(path.suffix.lower())
        if lang is None or path.stat().st_size > MAX_FILE_BYTES:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        info = _analyze_file(rel, text, lang)
        files.append(info)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    git = _git_intel(root)
    for f in files:  # attach per-file git signals
        f["git"] = git.get("files", {}).get(f["path"], {})

    analysis = {
        "name": root.name,
        "root": str(root),
        "analyzed_at": time.time(),
        "languages": lang_counts,
        "n_files": len(files),
        "total_loc": sum(f["loc"] for f in files),
        "git": {k: v for k, v in git.items() if k != "files"},
        "files": files,
    }
    logger.info("gitnexus analyzed '{}': {} files, {} LOC in {:.1f}s",
                root.name, len(files), analysis["total_loc"], time.time() - t0)
    return analysis


def _analyze_file(rel: str, text: str, lang: str) -> dict:
    info = {"path": rel, "language": lang, "loc": text.count("\n") + 1,
            "symbols": [], "imports": [], "calls": [], "text": text}
    if lang == "python":
        _parse_python(info, text)
    elif lang in ("markdown", "text", "yaml", "json", "xml", "config", "sql"):
        pass  # data/docs: indexed for retrieval, no symbols
    else:
        _parse_generic(info, text)
    return info


def _parse_python(info: dict, text: str) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        logger.debug("python parse failed for {}: {}", info["path"], exc)
        return

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def _add(self, node, kind):
            qual = ".".join(self.stack + [node.name])
            bases = [ast.unparse(b) for b in getattr(node, "bases", [])]
            info["symbols"].append({
                "name": node.name, "qualname": qual, "kind": kind,
                "start": node.lineno, "end": getattr(node, "end_lineno", node.lineno),
                "parent": ".".join(self.stack), "bases": bases,
                "doc": (ast.get_docstring(node) or "")[:300],
            })

        def visit_ClassDef(self, node):
            self._add(node, "class")
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self._add(node, "method" if self.stack else "function")
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            info["imports"].extend(a.name for a in node.names)

        def visit_ImportFrom(self, node):
            if node.module:
                info["imports"].append(node.module)

        def visit_Call(self, node):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None)
            if name:
                info["calls"].append(name)
            self.generic_visit(node)

    Visitor().visit(tree)
    info["calls"] = sorted(set(info["calls"]))
    info["imports"] = sorted(set(info["imports"]))


def _parse_generic(info: dict, text: str) -> None:
    seen = set()
    for kind, pattern in _GENERIC_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1)
            if name in seen or name in ("if", "for", "while", "switch", "catch", "return"):
                continue
            seen.add(name)
            line = text.count("\n", 0, m.start()) + 1
            info["symbols"].append({
                "name": name, "qualname": name, "kind": kind, "start": line,
                "end": min(line + 40, info["loc"]), "parent": "", "bases": [], "doc": "",
            })
    for m in _GENERIC_IMPORT.finditer(text):
        target = next((g for g in m.groups()[1:] if g), m.group(1))
        if target:
            info["imports"].append(target.strip().strip("{}").strip()[:80])
    info["imports"] = sorted(set(info["imports"]))
    # call references: identifiers followed by '(' that match nothing local
    info["calls"] = sorted(set(re.findall(r"\b([a-z]\w{2,})\s*\(", text)))[:200]


def _git_intel(root: Path) -> dict:
    """Commit-history signals; silently empty if git/.git is unavailable."""
    if not (root / ".git").exists():
        return {}

    def run(*args) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True,
                text=True, timeout=30, encoding="utf-8", errors="replace",
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""

    out = {"files": {}}
    log = run("log", "--pretty=format:%H|%an|%as|%s", "-n", "500")
    commits = [dict(zip(("hash", "author", "date", "subject"), line.split("|", 3)))
               for line in log.splitlines() if "|" in line]
    out["n_commits"] = len(commits)
    out["recent_commits"] = commits[:15]
    out["authors"] = sorted({c["author"] for c in commits})

    # change frequency per file -> "hotspot" score
    freq = run("log", "--name-only", "--pretty=format:", "-n", "500")
    counts: dict[str, int] = {}
    for line in freq.splitlines():
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    for path, n in counts.items():
        out["files"][path] = {"commits": n}
    return out
