# Post-Reboot Recovery Quick Reference

## Status Before Reboot (2025-10-14)
- ✅ SELinux NVIDIA fix installed: `/etc/udev/rules.d/71-nvidia-selinux.rules`
- ✅ Docker containers configured with CDI GPU access
- ✅ Systemd auto-start enabled: `~/.config/systemd/user/kwaainet-compose.service`
- ✅ 6 commits ready to push to origin/main

## Quick Verification (Run These Commands)

```bash
# 1. Check SELinux contexts (should be xserver_misc_device_t)
ls -laZ /dev/nvidia* | grep -E "(nvidia-modeset|nvidia-uvm)"

# 2. Check containers running
podman ps

# 3. Check for SELinux denials (should be 0 or minimal)
journalctl -b -0 --no-pager | grep -i "selinux.*denied.*nvidia" | wc -l

# 4. Check systemd service
systemctl --user status kwaainet-compose.service

# 5. Verify GPU in container
podman exec kwaainet-node ls /dev/nvidia0

# 6. Check network connectivity
podman logs kwaainet-node | grep Announced
```

## Expected Results
- All `/dev/nvidia-*` devices have `xserver_misc_device_t` context
- Both containers (kwaainet-api, kwaainet-node) are "Up"
- Zero or very few SELinux denials
- Systemd service shows "active (exited)" with container processes
- GPU devices accessible inside container
- Node announced blocks to network

## If Something Failed

### Login Failed
```bash
# Boot to recovery mode, then:
sudo restorecon -v /dev/nvidia-modeset /dev/nvidia-uvm /dev/nvidia-uvm-tools
reboot
```

### Containers Not Running
```bash
# Check user lingering
loginctl show-user metro | grep Linger

# Manual start
podman compose -f ~/compose.yml up -d
```

### GPU Not Accessible
```bash
# Check nvidia-persistenced
systemctl status nvidia-persistenced.service

# If not running:
sudo systemctl start nvidia-persistenced.service
```

## Next Steps After Successful Reboot
1. Document results in CLAUDE.md
2. Push all 6 commits: `git push origin main`
3. Mark reboot test as ✅ PASSED

---
See CLAUDE.md for detailed session history and troubleshooting.
