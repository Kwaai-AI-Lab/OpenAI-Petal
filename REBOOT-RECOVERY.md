# Post-Reboot Recovery & Verification Guide

## Quick Reference: System Won't Login After Reboot

If the graphical login screen crashes or loops after reboot, this is caused by SELinux blocking GNOME Shell GPU access.

### Immediate Fix (From Console - Ctrl+Alt+F2)

```bash
# Login at text console
# Username: metro
# Password: password

# Fix SELinux contexts
sudo restorecon -v /dev/nvidia-modeset /dev/nvidia-uvm /dev/nvidia-uvm-tools

# Return to graphical login (Ctrl+Alt+F1 or reboot)
sudo systemctl restart gdm
```

After this immediate fix, the permanent systemd service should prevent future issues.

---

## Automated Post-Reboot Verification Checklist

Run these commands after any reboot to verify system health:

### 1. Check SELinux NVIDIA Device Contexts

```bash
ls -laZ /dev/nvidia* | grep -E "(nvidia-modeset|nvidia-uvm)"
```

**Expected Output:**
```
xserver_misc_device_t    /dev/nvidia-modeset  ✅
xserver_misc_device_t    /dev/nvidia-uvm      ✅
xserver_misc_device_t    /dev/nvidia-uvm-tools ✅
```

**If showing `device_t` instead:** SELinux fix didn't run - check systemd service below.

### 2. Check SELinux Fix Service

```bash
systemctl status nvidia-selinux-fix.service
```

**Expected Output:**
```
Active: active (exited)
Loaded: enabled
```

**If failed or not started:**
```bash
sudo systemctl start nvidia-selinux-fix.service
sudo journalctl -u nvidia-selinux-fix.service
```

### 3. Check SELinux Denials

```bash
journalctl -b -0 --no-pager | grep -i "selinux.*denied.*nvidia" | wc -l
```

**Expected:** `0` (no recent denials)

**If non-zero:** Contexts weren't applied at boot - check service dependency chain.

### 4. Verify Docker Containers Auto-Started

```bash
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected Output:**
```
NAMES          STATUS        PORTS
kwaainet-node  Up X minutes  0.0.0.0:8082->8080/tcp
kwaainet-api   Up X minutes  0.0.0.0:80->8000/tcp
```

**If not running:**
```bash
systemctl --user status kwaainet-compose.service
podman compose -f ~/compose.yml up -d
```

### 5. Check Systemd User Service

```bash
systemctl --user status kwaainet-compose.service
```

**Expected Output:**
```
Active: active (exited)
Loaded: enabled
```

**If failed:** Check logs with `journalctl --user -u kwaainet-compose.service`

### 6. Verify GPU Access in Containers

```bash
podman exec kwaainet-node ls -la /dev/nvidia* | head -5
```

**Expected:** All NVIDIA devices present (nvidia0, nvidiactl, nvidia-uvm, etc.)

**If missing devices:** Check nvidia-persistenced service status.

### 7. Check Node Network Connectivity

```bash
podman logs kwaainet-node | grep -E "(Announced|Running Petals|Model already)"
```

**Expected Output:**
```
Model already exists. Skipping download.
Running Petals 2.3.0.dev2
Announced that blocks [0-31] are joining
```

**If "Downloading..."**: Model cache issue - check volume mounts in compose.yml.

### 8. Verify User Lingering

```bash
loginctl show-user metro | grep Linger
```

**Expected:** `Linger=yes`

**If no:** `sudo loginctl enable-linger metro`

---

## Technical Background: Why SELinux Fix is Needed

### The Problem

**NVIDIA driver creates device files directly (not via udev):**
- `/dev/nvidia-modeset` - Created by nvidia-modeset kernel module
- `/dev/nvidia-uvm` - Created by nvidia-persistenced on first GPU access
- `/dev/nvidia-uvm-tools` - Created by nvidia-persistenced

**Default SELinux context:** `device_t` (generic device)
**Required context:** `xserver_misc_device_t` (X server device)

**Impact:** GNOME Shell (`gnome-session-check-accelerated-gl-helper`) cannot access GPU with `device_t` context, causing infinite crash loop at login.

### Why Udev Rules Don't Work

**Original approach (FAILED):**
```bash
# /etc/udev/rules.d/71-nvidia-selinux.rules
KERNEL=="nvidia-modeset", RUN+="/usr/sbin/restorecon /dev/nvidia-modeset"
```

**Why it fails:**
1. NVIDIA devices created by `nvidia-modprobe` binary, not udev
2. No udev ADD event triggered when devices appear
3. Devices exist but udev never processes them as "new" devices
4. `RUN+=` commands never execute

### Correct Solution: Systemd Service

**Approach:** Run `restorecon` after nvidia-persistenced creates all devices.

**Service file:** `/etc/systemd/system/nvidia-selinux-fix.service`

```ini
[Unit]
Description=Fix SELinux contexts for NVIDIA devices
After=nvidia-persistenced.service
Requires=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/restorecon -v /dev/nvidia-modeset /dev/nvidia-uvm /dev/nvidia-uvm-tools
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

**Installation:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable nvidia-selinux-fix.service
```

**Why this works:**
1. Service runs after nvidia-persistenced finishes creating devices
2. All devices guaranteed to exist before restorecon runs
3. Executes every boot via systemd dependency chain
4. Failures are logged and visible via systemctl status

---

## Troubleshooting Common Issues

### Issue: Login Still Fails After SELinux Fix

**Symptoms:** Graphical login loops even with correct contexts

**Possible Causes:**
1. GDM service issue
2. User home directory permission problem
3. X server configuration issue

**Diagnosis:**
```bash
# Check GDM logs
journalctl -u gdm -b -0

# Check X server logs
cat /var/log/Xorg.0.log | grep -i error

# Check user session logs
journalctl --user -b -0
```

### Issue: Containers Don't Start at Boot

**Symptoms:** `podman ps` shows no containers after reboot

**Diagnosis:**
```bash
# Check service status
systemctl --user status kwaainet-compose.service

# Check service logs
journalctl --user -u kwaainet-compose.service

# Check user lingering
loginctl show-user metro | grep Linger
```

**Fix:**
```bash
# Enable user lingering
sudo loginctl enable-linger metro

# Restart service
systemctl --user restart kwaainet-compose.service
```

### Issue: GPU Not Accessible in Containers

**Symptoms:** `podman exec kwaainet-node ls /dev/nvidia*` returns error

**Diagnosis:**
```bash
# Check nvidia-persistenced
systemctl status nvidia-persistenced.service

# Check host GPU access
ls -la /dev/nvidia*

# Check CDI configuration
podman info | grep -i cdi
```

**Fix:**
```bash
# Restart nvidia-persistenced
sudo systemctl restart nvidia-persistenced.service

# Recreate containers
podman compose -f ~/compose.yml down
podman compose -f ~/compose.yml up -d
```

---

## Pre-Reboot Testing Checklist

Before rebooting, verify these items to ensure smooth recovery:

- [ ] nvidia-selinux-fix.service enabled: `systemctl is-enabled nvidia-selinux-fix.service`
- [ ] nvidia-persistenced.service enabled: `systemctl is-enabled nvidia-persistenced.service`
- [ ] kwaainet-compose.service enabled: `systemctl --user is-enabled kwaainet-compose.service`
- [ ] User lingering enabled: `loginctl show-user metro | grep Linger=yes`
- [ ] Compose file valid: `podman compose -f ~/compose.yml config`
- [ ] Model cache exists: `du -sh ~/.cache/huggingface`
- [ ] All changes committed: `git status` (clean working directory)

---

## Quick Reference Command Summary

```bash
# Fix login immediately (console)
sudo restorecon -v /dev/nvidia-modeset /dev/nvidia-uvm /dev/nvidia-uvm-tools
sudo systemctl restart gdm

# Check all services
systemctl status nvidia-selinux-fix.service
systemctl status nvidia-persistenced.service
systemctl --user status kwaainet-compose.service

# Verify containers
podman ps
podman logs kwaainet-node | tail -20
podman exec kwaainet-node ls /dev/nvidia*

# Check SELinux
ls -laZ /dev/nvidia* | grep xserver_misc_device_t
journalctl -b -0 | grep -i "selinux.*denied.*nvidia" | wc -l

# Restart services if needed
sudo systemctl restart nvidia-selinux-fix.service
systemctl --user restart kwaainet-compose.service
```

---

## System Architecture Overview

```
Boot Sequence:
├─ 1. Kernel loads NVIDIA modules (nvidia, nvidia-modeset, nvidia-uvm)
├─ 2. nvidia-persistenced.service starts
│     └─ Creates /dev/nvidia-uvm devices
├─ 3. nvidia-selinux-fix.service runs (after #2)
│     └─ Applies xserver_misc_device_t contexts
├─ 4. GDM starts (can now access GPU)
├─ 5. User login triggers systemd user session
│     └─ kwaainet-compose.service starts
│           └─ Containers start with GPU access via CDI
└─ 6. GNOME Shell loads (GPU accessible, no crashes)
```

**Critical Dependencies:**
- nvidia-selinux-fix MUST run after nvidia-persistenced
- GDM MUST start after SELinux contexts applied
- User services MUST have lingering enabled to survive logout

---

## Recovery Contacts & Resources

**If this guide doesn't solve the issue:**

1. Check system logs: `journalctl -b -0 --no-pager | less`
2. Review CLAUDE.md session history for context
3. Check NVIDIA driver version: `cat /proc/driver/nvidia/version`
4. Verify SELinux mode: `getenforce` (should be "Enforcing")
5. Test in permissive mode: `sudo setenforce 0` (TEMPORARY - for diagnosis only)

**Useful diagnostic commands:**
```bash
# Full boot log
journalctl -b -0 --no-pager > /tmp/boot.log

# SELinux audit log
sudo ausearch -m avc -ts recent

# NVIDIA driver info
nvidia-smi
lsmod | grep nvidia
dmesg | grep -i nvidia

# Container logs
podman logs kwaainet-node > /tmp/node.log
podman logs kwaainet-api > /tmp/api.log
```
