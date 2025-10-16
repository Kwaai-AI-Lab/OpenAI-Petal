# KwaaiNet Common Module

**Shared cross-platform utilities for Linux, macOS, and Windows**

## Overview

This module provides high-value shared functionality to eliminate code duplication across platform-specific implementations while avoiding complex inheritance patterns.

### Design Philosophy: Pragmatic Hybrid Approach

After analysis, we chose **utility functions over base class inheritance** because:

1. ✅ **Avoids import complexity** - No package name shadowing issues
2. ✅ **Lower coupling** - Platforms can pick what they need
3. ✅ **Easier testing** - Individual functions tested in isolation
4. ✅ **Faster development** - 3-5 hours vs 11-15 hours for full refactoring
5. ✅ **Immediate value** - Shares critical code (locking, cleanup) immediately

## Module Structure

```
kwaainet/common/
├── __init__.py           # Public API exports
├── README.md             # This file
├── utils.py              # General utilities (IP detection, formatting)
├── platform.py           # Platform detection and helpers
├── daemon_utils.py       # Critical daemon functionality
└── cli_utils.py          # CLI formatting and interaction
```

---

## Quick Start

### Importing in Platform Code

```python
# At top of any platform file (daemon.py, runner.py, etc.)
import sys
import os

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import utilities
from kwaainet.common import daemon_utils, cli_utils
```

### Using Daemon Utilities

```python
# Process locking
lock_fd = daemon_utils.acquire_process_lock("/path/to/lock")
if not lock_fd:
    exit(1)

# Cleanup stale processes
daemon_utils.cleanup_stale_processes(['petals.cli.run_server', 'p2pd'])

# PID validation
pid = daemon_utils.validate_pid_file("/var/run/kwaainet.pid")
```

### Using CLI Utilities

```python
# Formatted output
cli_utils.print_success("Daemon started!")
cli_utils.print_error("Connection failed")

# Status display
status = daemon.get_status()
print(cli_utils.format_status_output(status))
```

---

## Critical Features

### 1. Process Locking (CRITICAL)

**Problem Solved:** macOS had locking, Linux/Windows didn't → race conditions

**Functions:**
- `acquire_process_lock(lock_file)` - Cross-platform locking
- `release_process_lock(lock_fd, lock_file)` - Release lock

### 2. Process Cleanup

**Problem Solved:** Duplicate cleanup code in all platforms

**Functions:**
- `cleanup_stale_processes(patterns, exclude_pids)` - Terminate stale processes

### 3. PID Management

**Problem Solved:** PID validation duplicated across platforms

**Functions:**
- `validate_pid_file(pid_file, patterns)` - Validate PID
- `write_pid_file(pid_file, pid)` - Write PID
- `cleanup_pid_file(pid_file, status_file)` - Cleanup files

---

## Impact Summary

**Lines Saved:** ~330 lines of duplicate code eliminated
**Critical Bug Fixed:** Process locking now available on all platforms
**Windows Ready:** 400 lines of utilities ready for Windows development

---

See inline documentation in each module for detailed API reference.
