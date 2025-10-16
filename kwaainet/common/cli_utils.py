"""
Common CLI utilities for consistent command-line interface across platforms
"""

import sys
from typing import Any, Dict, Optional


def format_status_output(status: Dict[str, Any]) -> str:
    """
    Format daemon status for console output

    Args:
        status: Status dictionary from daemon.get_status()

    Returns:
        Formatted status string
    """
    if not status.get("running"):
        return "❌ KwaaiNet daemon is not running"

    lines = [
        f"✅ KwaaiNet daemon is running (PID: {status.get('pid')})"
    ]

    # Add uptime if available
    if 'uptime' in status:
        uptime_str = format_uptime(status['uptime'])
        lines.append(f"   Uptime: {uptime_str}")

    # Add resource usage if available
    if 'cpu_percent' in status:
        lines.append(f"   CPU: {status['cpu_percent']:.1f}%")

    if 'memory_percent' in status:
        lines.append(f"   Memory: {status['memory_percent']:.1f}%")

    if 'memory_mb' in status:
        lines.append(f"   Memory: {status['memory_mb']:.1f} MB")

    # Add network info if available
    if 'connections' in status:
        lines.append(f"   Connections: {status['connections']}")

    return "\n".join(lines)


def format_uptime(seconds: float) -> str:
    """
    Format uptime in seconds to human-readable string

    Args:
        seconds: Uptime in seconds

    Returns:
        Formatted string (e.g., "2d 3h 15m")
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return ' '.join(parts) if parts else "< 1m"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def print_error(message: str):
    """
    Print error message to stderr

    Args:
        message: Error message
    """
    print(f"❌ Error: {message}", file=sys.stderr)


def print_success(message: str):
    """
    Print success message to stdout

    Args:
        message: Success message
    """
    print(f"✅ {message}")


def print_warning(message: str):
    """
    Print warning message to stdout

    Args:
        message: Warning message
    """
    print(f"⚠️  {message}")


def print_info(message: str):
    """
    Print info message to stdout

    Args:
        message: Info message
    """
    print(f"ℹ️  {message}")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask user for confirmation

    Args:
        prompt: Confirmation prompt
        default: Default answer if user just presses Enter

    Returns:
        True if user confirms, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        response = input(f"{prompt}{suffix}: ").strip().lower()

        if not response:
            return default

        return response in ('y', 'yes')

    except (KeyboardInterrupt, EOFError):
        print()  # New line after ^C
        return False


def format_config_display(config: Dict[str, Any]) -> str:
    """
    Format configuration dictionary for display

    Args:
        config: Configuration dictionary

    Returns:
        Formatted config string
    """
    lines = ["Configuration:"]

    # Key fields to display
    key_fields = [
        ('model', 'Model'),
        ('blocks', 'Blocks'),
        ('port', 'Port'),
        ('use_gpu', 'Use GPU'),
        ('log_level', 'Log Level'),
        ('public_name', 'Public Name'),
        ('public_ip', 'Public IP'),
    ]

    for key, label in key_fields:
        if key in config:
            value = config[key]
            lines.append(f"  {label}: {value}")

    return "\n".join(lines)


def truncate_text(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def format_table(headers: list, rows: list) -> str:
    """
    Format data as ASCII table

    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of values)

    Returns:
        Formatted table string
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]

    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Format header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    # Format rows
    row_lines = [
        " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        for row in rows
    ]

    return "\n".join([header_line, separator] + row_lines)
