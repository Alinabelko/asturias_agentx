"""
Tools available to the research agent.

Each tool is registered as an OpenAI function-calling schema.
The matching implementation is in TOOL_IMPLEMENTATIONS.

Safety model:
- All code execution is limited to WORKSPACE_ROOT (a temp dir per task).
- Commands run with a hard timeout.
- Potentially dangerous patterns are blocked before exec.
"""

import os
import shlex
import subprocess
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories at a given path inside the workspace. "
                "Returns names, types (file/dir), and sizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside workspace root. '.' = workspace root.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list recursively (max 200 entries).",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read content of a file inside the workspace. "
                "For large files, optionally read a slice (start_line to end_line, 1-indexed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "start_line": {"type": "integer", "description": "First line to return (1-indexed)."},
                    "end_line": {"type": "integer", "description": "Last line to return (inclusive)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "content": {"type": "string", "description": "File content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command inside the workspace directory. "
                "Use this to install dependencies, run scripts, execute Python/R/Julia code, etc. "
                "stdout and stderr are captured and returned. "
                "Commands are killed after timeout_seconds (default 120)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run (bash syntax).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Max seconds before killing the command. Default 120.",
                        "default": 120,
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Subdirectory of workspace to run from (relative). "
                            "Defaults to workspace root."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Grep for a pattern inside workspace files. "
                "Returns matching lines with file names and line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search."},
                    "path": {
                        "type": "string",
                        "description": "Relative path or directory to search in. Defaults to '.'.",
                        "default": ".",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files, e.g. '*.py'. Defaults to all files.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max lines to return. Default 50.",
                        "default": 50,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clone_repo",
            "description": (
                "Clone a git repository or download a zip archive into the workspace. "
                "Use this when the task provides a GitHub URL or zip download link."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Git repository URL (https://github.com/...) or direct zip URL.",
                    },
                    "dest": {
                        "type": "string",
                        "description": "Destination subdirectory name inside workspace. Defaults to 'repo'.",
                        "default": "repo",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_results",
            "description": (
                "Submit the final reproduction results. Call this when you have finished "
                "attempting to reproduce the paper's results. Include whether reproduction "
                "succeeded, the values you obtained, and comparison with paper's claims."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "True if results were successfully reproduced.",
                    },
                    "reproduced_values": {
                        "type": "object",
                        "description": "Key-value pairs of metric names to reproduced values.",
                    },
                    "expected_values": {
                        "type": "object",
                        "description": "Key-value pairs of metric names to values claimed in the paper.",
                    },
                    "match_summary": {
                        "type": "string",
                        "description": "Human-readable summary of how well results matched.",
                    },
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any errors or issues encountered during reproduction.",
                    },
                    "steps_taken": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Brief list of steps taken to reproduce.",
                    },
                },
                "required": ["success", "reproduced_values", "match_summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Blocked command prefixes — crude but enough for research sandbox
_BLOCKED = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "shutdown", "reboot", "wget http"]


def _safe_workspace_path(workspace: Path, rel_path: str) -> Path:
    """Resolve rel_path inside workspace, raise if it escapes."""
    resolved = (workspace / rel_path).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise ValueError(f"Path '{rel_path}' escapes workspace root")
    return resolved


def tool_list_files(workspace: Path, path: str, recursive: bool = False) -> str:
    target = _safe_workspace_path(workspace, path)
    if not target.exists():
        return f"ERROR: path does not exist: {path}"

    entries = []
    if recursive:
        for p in sorted(target.rglob("*"))[:200]:
            kind = "dir" if p.is_dir() else "file"
            size = p.stat().st_size if p.is_file() else ""
            entries.append(f"{kind:4}  {p.relative_to(workspace)}  {size}")
    else:
        for p in sorted(target.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            size = p.stat().st_size if p.is_file() else ""
            entries.append(f"{kind:4}  {p.name}  {size}")

    return "\n".join(entries) if entries else "(empty directory)"


def tool_read_file(
    workspace: Path, path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    target = _safe_workspace_path(workspace, path)
    if not target.exists():
        return f"ERROR: file not found: {path}"
    if not target.is_file():
        return f"ERROR: not a file: {path}"

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"ERROR reading file: {e}"

    if start_line is not None or end_line is not None:
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        lines = lines[s:e]

    # Cap at 400 lines to keep context manageable
    if len(lines) > 400:
        lines = lines[:400]
        lines.append(f"... (truncated, file has more lines)")

    numbered = [f"{i+1:4}: {l}" for i, l in enumerate(lines)]
    return "\n".join(numbered)


def tool_write_file(workspace: Path, path: str, content: str) -> str:
    target = _safe_workspace_path(workspace, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_run_command(
    workspace: Path,
    command: str,
    timeout_seconds: int = 120,
    workdir: str | None = None,
) -> str:
    # Safety check
    for blocked in _BLOCKED:
        if blocked in command:
            return f"ERROR: blocked command pattern detected: '{blocked}'"

    cwd = workspace
    if workdir:
        cwd = _safe_workspace_path(workspace, workdir)
        if not cwd.is_dir():
            return f"ERROR: workdir does not exist: {workdir}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout
        stderr = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        parts = [f"exit_code: {result.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout_seconds}s"
    except Exception as e:
        return f"ERROR: {e}"


def tool_search_in_files(
    workspace: Path,
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    max_results: int = 50,
) -> str:
    target = _safe_workspace_path(workspace, path)
    cmd = ["grep", "-rn", "--include", file_glob or "*", pattern, str(target)]
    if file_glob is None:
        cmd = ["grep", "-rn", pattern, str(target)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.splitlines()[:max_results]
        # Make paths relative to workspace
        cleaned = []
        for line in lines:
            if str(workspace) in line:
                line = line.replace(str(workspace) + "/", "")
                line = line.replace(str(workspace) + "\\", "")
            cleaned.append(line)
        return "\n".join(cleaned) if cleaned else "(no matches)"
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out"
    except FileNotFoundError:
        # grep not available (Windows) — fallback
        return tool_search_in_files_py(workspace, pattern, path, file_glob, max_results)
    except Exception as e:
        return f"ERROR: {e}"


def tool_search_in_files_py(
    workspace: Path,
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    max_results: int = 50,
) -> str:
    """Pure-Python fallback search for Windows."""
    import re
    target = _safe_workspace_path(workspace, path)
    glob = file_glob or "*"
    matches = []
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: invalid pattern: {e}"

    for filepath in sorted(target.rglob(glob)):
        if not filepath.is_file():
            continue
        try:
            for i, line in enumerate(filepath.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    rel = filepath.relative_to(workspace)
                    matches.append(f"{rel}:{i}: {line}")
                    if len(matches) >= max_results:
                        return "\n".join(matches) + "\n(max results reached)"
        except Exception:
            continue
    return "\n".join(matches) if matches else "(no matches)"


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

def tool_clone_repo(workspace: Path, url: str, dest: str = "repo") -> str:
    dest_path = _safe_workspace_path(workspace, dest)
    if dest_path.exists():
        return f"OK: '{dest}' already exists, skipping clone"

    # Zip download
    if url.endswith(".zip") or "zipball" in url or "archive" in url:
        import urllib.request, zipfile, io
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(dest_path)
            return f"OK: extracted zip to {dest}/ ({len(list(dest_path.rglob('*')))} files)"
        except Exception as e:
            return f"ERROR downloading zip: {e}"

    # Git clone
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        count = len(list(dest_path.rglob("*")))
        return f"OK: cloned to {dest}/ ({count} files)"
    return f"ERROR: git clone failed\n{result.stderr}"


def dispatch_tool(workspace: Path, name: str, args: dict) -> str:
    """Route tool call to the right implementation. Returns string result."""
    if name == "clone_repo":
        return tool_clone_repo(workspace, args["url"], args.get("dest", "repo"))
    elif name == "list_files":
        return tool_list_files(workspace, args.get("path", "."), args.get("recursive", False))
    elif name == "read_file":
        return tool_read_file(
            workspace,
            args["path"],
            args.get("start_line"),
            args.get("end_line"),
        )
    elif name == "write_file":
        return tool_write_file(workspace, args["path"], args["content"])
    elif name == "run_command":
        return tool_run_command(
            workspace,
            args["command"],
            args.get("timeout_seconds", 120),
            args.get("workdir"),
        )
    elif name == "search_in_files":
        return tool_search_in_files(
            workspace,
            args["pattern"],
            args.get("path", "."),
            args.get("file_glob"),
            args.get("max_results", 50),
        )
    elif name == "report_results":
        # This is handled specially in agent.py — just echo back as JSON string
        import json
        return json.dumps(args)
    else:
        return f"ERROR: unknown tool '{name}'"
