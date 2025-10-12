# Development Environments

This directory contains platform-specific configuration and notes for different development machines.

## Files

- `macos-rezarassool.md` - macOS development machine (rezarassool)
- `linux-metro.md` - Linux production server (metro)
- `windows-[name].md` - Windows development machine (if applicable)

## Usage

When working on a specific machine, refer to its environment file for:
- Hardware specifications
- Software versions
- Installation paths
- Platform-specific quirks
- Common commands
- Known issues

## Detecting Current Environment

The active environment is detected via hostname or can be explicitly set:

```bash
# Check current hostname
hostname

# macOS: returns something like "rezarassools-MacBook-Pro.local"
# Linux: returns "metro" or similar
```

## Adding a New Environment

1. Copy an existing environment file as a template
2. Update hardware, software, and configuration details
3. Document any platform-specific peculiarities
4. Add common commands you use on that machine
