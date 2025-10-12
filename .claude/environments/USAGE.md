# Multi-Platform Development Environment Usage Guide

## Overview

This directory contains platform-specific configuration and notes for different development machines. This allows you to seamlessly switch between machines while maintaining context about each environment's peculiarities.

## Quick Start

### 1. Detect Your Current Environment

```bash
cd /path/to/OpenAI-Petal
./.claude/detect-environment.sh
```

**Output Examples:**
- macOS: `.claude/environments/macos-rezarassool.md`
- Linux: `.claude/environments/linux-metro.md`

### 2. Read Your Environment Configuration

```bash
# View environment details
cat $(./.claude/detect-environment.sh)

# Or directly
cat .claude/environments/macos-rezarassool.md
```

### 3. Update Environment Details

Edit the appropriate file for your machine:
```bash
# macOS
vim .claude/environments/macos-rezarassool.md

# Linux
vim .claude/environments/linux-metro.md
```

## File Structure

```
.claude/environments/
├── README.md                    # Overview of environment system
├── USAGE.md                     # This file - usage guide
├── .gitignore                   # Protects local-only files
├── macos-rezarassool.md        # macOS development machine
├── linux-metro.md              # Linux production server
└── [machine]-local.md          # Local modifications (gitignored)
```

## What to Document in Environment Files

### Hardware
- Model and architecture (ARM64, x86_64)
- RAM and storage specs
- GPU details (if applicable)

### Software Stack
- Operating system version
- Python environment (conda, venv, system)
- Container runtime (if using Docker/Podman)
- GPU toolkit versions

### Configuration
- Installation type (editable, pip, container)
- File paths (launchers, configs, logs)
- Service configuration (systemd, launchd)
- Network settings

### Platform-Specific Peculiarities
- Known issues and workarounds
- Architecture-specific paths
- Permission requirements
- Environment variables

### Common Commands
- Development workflow commands
- Service management
- Debugging commands
- Testing procedures

## Using Local-Only Files

For sensitive or machine-specific information that shouldn't be committed:

```bash
# Create a local-only file
cp .claude/environments/macos-rezarassool.md \
   .claude/environments/macos-rezarassool-local.md

# Edit with sensitive details
vim .claude/environments/macos-rezarassool-local.md
```

**Note:** All `*-local.md` files are automatically gitignored.

## Adding a New Machine

1. **Copy an existing template:**
   ```bash
   cp .claude/environments/macos-rezarassool.md \
      .claude/environments/macos-newmachine.md
   ```

2. **Update hardware and software details**

3. **Add platform-specific peculiarities**

4. **Update the detection script:**
   ```bash
   vim .claude/detect-environment.sh
   # Add hostname pattern matching for your new machine
   ```

5. **Test detection:**
   ```bash
   ./.claude/detect-environment.sh
   ```

6. **Commit the new environment file:**
   ```bash
   git add .claude/environments/macos-newmachine.md \
           .claude/detect-environment.sh
   git commit -m "Add macos-newmachine environment"
   ```

## Benefits of This System

### 1. Context Switching
- Instantly recall machine-specific details
- No more "How did I configure this again?"
- Quick reference for common commands

### 2. Debugging Across Platforms
- Document platform-specific issues
- Track workarounds and solutions
- Compare configurations between machines

### 3. Onboarding New Machines
- Use existing environment as template
- Consistent documentation structure
- Easier to replicate working setups

### 4. Collaboration
- Share machine configurations with team
- Document production vs development differences
- Track infrastructure changes over time

## Example Workflow

```bash
# Switch to your dev machine (macOS)
cd ~/Source/OpenAI-Petal

# Check which environment you're in
./.claude/detect-environment.sh
# Output: .claude/environments/macos-rezarassool.md

# Read common commands for this machine
grep -A 20 "## Common Commands" $(  /.claude/detect-environment.sh)

# Make changes and test

# Switch to production server (Linux)
ssh metro
cd ~/Source/OpenAI-Petal

# Check environment
./.claude/detect-environment.sh
# Output: .claude/environments/linux-metro.md

# Read platform-specific quirks
grep -A 20 "## Platform-Specific Peculiarities" $(./.claude/detect-environment.sh)

# Deploy with proper configuration
```

## Integration with Claude Code

When working with Claude Code, you can reference your environment:

```
User: "Check the logs on this machine"

Claude: "Let me check your environment configuration first..."
[runs ./.claude/detect-environment.sh]
[reads macos-rezarassool.md]
"On macOS, logs are at ~/.kwaainet/logs/. Let me check them..."
```

This allows Claude to:
- Use correct paths for your machine
- Run appropriate commands for your platform
- Reference platform-specific quirks
- Provide accurate troubleshooting

## Tips

1. **Keep it updated:** Update your environment file when you make configuration changes

2. **Document workarounds:** If you find a fix for a platform-specific issue, add it immediately

3. **Use TODOs:** Mark incomplete sections with `[TODO: specify]` so you know what to fill in later

4. **Reference from CLAUDE.md:** Link to environment files when documenting sessions

5. **Test detection script:** After hostname changes, verify the detection script still works

## See Also

- `.claude/environments/README.md` - System overview
- `CLAUDE.md` - Main development history (Lesson #5: Multi-Platform Development)
- Individual environment files for specific machine details
