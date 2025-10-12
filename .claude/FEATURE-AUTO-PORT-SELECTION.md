# Automatic Port Selection Feature

## Overview
Implemented automatic port fallback functionality that ensures KwaaiNet can always start and join the P2P network, even when the configured port is occupied.

**Status:** ✅ COMPLETED (2025-10-12)

## Problem Statement
Previously, if the configured port (e.g., 8080) was already in use, KwaaiNet would fail to start. This prevented nodes from joining the P2P network when there were port conflicts.

## Solution
Implemented smart port selection logic that:
1. **Tries the preferred port first** from config file
2. **Automatically finds alternatives** if preferred port is occupied
3. **Searches nearby ports** (±10 from preferred) for minimal disruption
4. **Falls back to wider range** (8000-9000) if needed
5. **Provides clear user feedback** about port changes

## Implementation Details

### Files Modified

#### 1. `Installer/macOS/kwaainet/utils.py`
**Added functions:**
- `is_port_available(port, host='0.0.0.0')` - Check if a specific port is available
- `find_available_port(preferred_port, start_range=8000, end_range=9000, max_attempts=50)` - Find available port with intelligent fallback

**Port Selection Algorithm:**
1. Try preferred port
2. Try nearby ports (preferred ± 10)
3. Search full range (8000-9000)
4. Raise RuntimeError if no ports available

#### 2. `Installer/macOS/kwaainet/runner.py`
**Modified:** `start()` method in `KwaaiNetRunner` class

**Integration:**
```python
# Find an available port (prefer configured port, fallback to alternatives)
preferred_port = self.config.get("port", 8080)
try:
    port, is_preferred = find_available_port(preferred_port)
    if not is_preferred:
        logger.warning(f"Port {preferred_port} was not available, using alternate port {port}")
        logger.info(f"💡 To use this port permanently, update config: kwaainet config --set port {port}")
except RuntimeError as e:
    logger.error(f"Failed to find available port: {e}")
    return False
```

## User Experience

### Scenario 1: Preferred Port Available
```
$ kwaainet start --daemon
Starting KwaaiNet node with model: unsloth/Llama-3.1-8B-Instruct
Sharing 4 blocks
Using device: MPS (Metal Performance Shaders)
Using preferred port: 8080
✅ KwaaiNet node started successfully
```

### Scenario 2: Preferred Port Occupied
```
$ kwaainet start --daemon
Starting KwaaiNet node with model: unsloth/Llama-3.1-8B-Instruct
Sharing 4 blocks
Using device: MPS (Metal Performance Shaders)
⚠️  Port 8080 was not available, using alternate port 8081
💡 To use this port permanently, update config: kwaainet config --set port 8081
✅ KwaaiNet node started successfully
```

### Scenario 3: Multiple Ports Occupied
```
$ kwaainet start --daemon
Starting KwaaiNet node with model: unsloth/Llama-3.1-8B-Instruct
Sharing 4 blocks
Using device: MPS (Metal Performance Shaders)
⚠️  Port 8080 was not available, using alternate port 8095
💡 To use this port permanently, update config: kwaainet config --set port 8095
✅ KwaaiNet node started successfully
```

### Scenario 4: No Available Ports (Error)
```
$ kwaainet start --daemon
Starting KwaaiNet node with model: unsloth/Llama-3.1-8B-Instruct
Sharing 4 blocks
Using device: MPS (Metal Performance Shaders)
❌ Failed to find available port: Could not find an available port after checking 50 ports.
```

## Testing

### Automated Tests
Created comprehensive test suite: `test_port_selection.py`

**Tests included:**
1. ✅ Basic port availability checking
2. ✅ Finding available port when preferred is available
3. ✅ Finding alternative when preferred is occupied
4. ✅ Finding port with multiple occupied ports
5. ✅ Error handling when no ports available

**Test Results:**
```
============================================================
   Port Selection Automatic Tests
============================================================

Test 1: Basic port availability checking
✅ Port 9999 is correctly identified as available
✅ Port 9998 is correctly identified as occupied

Test 2: Finding available port (preferred available)
✅ Correctly returned preferred port 8765

Test 3: Finding alternative port (preferred occupied)
✅ Correctly found alternative port 8544
✅ Alternative port 8544 is confirmed available

Test 4: Finding port with multiple occupied ports
✅ Found available port 8603 (avoiding 5 occupied ports)

Test 5: Error handling (no available ports)
✅ Correctly raised RuntimeError

============================================================
✅ All tests completed successfully!
============================================================
```

### Practical Test
Created real-world scenario test: `test_port_occupied_scenario.py`

**Test Result:**
```
✅ SUCCESS: kwaainet would use alternative port 8091
   (Original port 8090 was occupied)
```

## Configuration

### Default Configuration
- **Preferred port:** Taken from `~/.kwaainet/config.yaml` (default: 8080)
- **Fallback range:** 8000-9000
- **Max attempts:** 50 ports
- **Search strategy:** Nearby first (±10), then full range

### User Configuration
Users can update the preferred port at any time:
```bash
# View current config
kwaainet config --view

# Set new preferred port
kwaainet config --set port 8085

# Restart to apply
kwaainet restart
```

## Benefits

1. **Reliability:** Node can always start, even with port conflicts
2. **User-Friendly:** Clear messaging about port changes
3. **Smart Fallback:** Tries nearby ports first for minimal disruption
4. **P2P Compatibility:** Ensures node can join network on any available port
5. **Easy Configuration:** Users can permanently set discovered ports

## Future Enhancements

Possible future improvements:
1. Automatically update config with discovered port (optional flag)
2. Remember last successful port for next startup
3. Port conflict detection before daemon start
4. Integration with service manager for persistent port allocation

## Technical Notes

### Port Binding Strategy
- Uses `SO_REUSEADDR` for checking availability
- Binds to `0.0.0.0` (all interfaces) for maximum compatibility
- TCP sockets used for P2P communication

### Error Handling
- Graceful fallback on `OSError` during binding
- Clear error messages when no ports available
- Maintains existing error handling for other failures

### Cross-Platform Compatibility
- Socket module is cross-platform (works on macOS, Linux, Windows)
- Port range 8000-9000 chosen to avoid common service conflicts
- Compatible with rootless and rootful deployments

## Related Files
- `Installer/macOS/kwaainet/utils.py` - Port selection utilities
- `Installer/macOS/kwaainet/runner.py` - Integration into startup flow
- `test_port_selection.py` - Automated test suite
- `test_port_occupied_scenario.py` - Practical test scenario
- `~/.kwaainet/config.yaml` - User configuration file

## Documentation Updates
Updated CLAUDE.md session history with feature implementation details.
