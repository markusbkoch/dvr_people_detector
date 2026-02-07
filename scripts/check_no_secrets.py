from __future__ import annotations

"""Fail commit when staged files contain likely secrets."""

import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ALLOWLIST = {
    ".secrets",
}


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("telegram_bot_token", re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{20,}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("github_classic_pat", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA) PRIVATE KEY-----")),
    ("password_assignment", re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{4,}['\"]")),
    ("token_assignment", re.compile(r"(?i)\b(?:token|api[_-]?key|secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]")),
    ("credential_url", re.compile(r"(?i)\b(?:https?|rtsp)://[^/\s:@]+:[^@\s]+@")),
]

IPV4_PATTERN = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
SEMVER4_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+==\d+\.\d+\.\d+\.\d+$")


def _is_private_or_local_ipv4(ip: str) -> bool:
    """Return True for private/local/reserved IPv4 ranges that are safe to ignore."""
    parts = [int(p) for p in ip.split(".")]
    if len(parts) != 4:
        return True
    a, b, _c, _d = parts
    if a == 10:
        return True
    if a == 127:
        return True
    if a == 0:
        return True
    if a == 169 and b == 254:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 192 and b == 168:
        return True
    return False


def _line_at(text: str, index: int) -> str:
    """Return full line that contains `index` in `text`."""
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end < 0:
        end = len(text)
    return text[start:end]


def _looks_like_placeholder_url(match_text: str) -> bool:
    """Return True when URL likely uses placeholders/template vars."""
    return any(ch in match_text for ch in ("<", ">", "{", "}"))


def _ipv4_context_suspicious(line: str) -> bool:
    """Return True when a line context suggests an IP is hardcoded config/secret."""
    stripped = line.strip()
    lower = stripped.lower()
    if not stripped or stripped.startswith("#"):
        return False
    if SEMVER4_PATTERN.fullmatch(stripped):
        return False
    # Likely assignment/config/URL context.
    if "://" in lower:
        return True
    if re.search(r"\b(?:ip|host|server|endpoint|url|camera|dvr|addr)\b", lower):
        return True
    if "=" in stripped or ":" in stripped:
        return True
    return False


def _run_git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=PROJECT_ROOT)
    return out.decode("utf-8", errors="replace")


def _staged_files() -> list[Path]:
    lines = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACM"]).splitlines()
    files: list[Path] = []
    for raw in lines:
        p = Path(raw.strip())
        if not raw.strip():
            continue
        if p.as_posix() in ALLOWLIST:
            continue
        files.append(PROJECT_ROOT / p)
    return files


def _is_binary(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return True
    if b"\x00" in data:
        return True
    return False


def main() -> int:
    try:
        files = _staged_files()
    except Exception as exc:
        print(f"[secrets-check] warning: failed to inspect staged files: {exc}")
        return 0

    findings: list[str] = []
    for path in files:
        if not path.exists() or not path.is_file():
            continue
        if _is_binary(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                if name == "credential_url" and _looks_like_placeholder_url(matched_text):
                    continue
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{line_no}: {name}")
        for match in IPV4_PATTERN.finditer(text):
            ip = match.group(0)
            if _is_private_or_local_ipv4(ip):
                continue
            line = _line_at(text, match.start())
            if not _ipv4_context_suspicious(line):
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append(f"{path.relative_to(PROJECT_ROOT)}:{line_no}: public_ipv4 ({ip})")

    if findings:
        print("[secrets-check] blocked commit. Potential secrets detected:")
        for item in findings:
            print(f"  - {item}")
        print("[secrets-check] if false positive, remove/obfuscate content before committing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
