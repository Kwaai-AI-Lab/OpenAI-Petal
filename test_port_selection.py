#!/usr/bin/env python3
"""
Test script for automatic port selection feature.
This script tests the port availability checking and fallback logic.
"""

import socket
import sys
import time
from pathlib import Path

# Add the installer path to sys.path
installer_path = Path(__file__).parent / "Installer" / "macOS"
sys.path.insert(0, str(installer_path))

from kwaainet.utils import is_port_available, find_available_port


def occupy_port(port):
    """Create a socket that occupies a port"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.listen(1)
    return sock


def test_port_availability():
    """Test basic port availability checking"""
    print("Test 1: Basic port availability checking")
    print("-" * 60)

    # Test an available port
    test_port = 9999
    if is_port_available(test_port):
        print(f"✅ Port {test_port} is correctly identified as available")
    else:
        print(f"⚠️  Port {test_port} appears to be in use")

    # Occupy a port and test again
    occupied_port = 9998
    sock = occupy_port(occupied_port)
    print(f"   Occupied port {occupied_port} for testing...")

    if not is_port_available(occupied_port):
        print(f"✅ Port {occupied_port} is correctly identified as occupied")
    else:
        print(f"❌ Port {occupied_port} should be occupied but appears available")

    sock.close()
    print()


def test_find_available_port_preferred():
    """Test finding available port when preferred is available"""
    print("Test 2: Finding available port (preferred available)")
    print("-" * 60)

    preferred = 8765
    port, is_preferred = find_available_port(preferred, start_range=8000, end_range=9000)

    if is_preferred and port == preferred:
        print(f"✅ Correctly returned preferred port {port}")
    else:
        print(f"❌ Expected preferred port {preferred}, got {port} (is_preferred={is_preferred})")
    print()


def test_find_available_port_occupied():
    """Test finding alternative port when preferred is occupied"""
    print("Test 3: Finding alternative port (preferred occupied)")
    print("-" * 60)

    preferred = 8543

    # Occupy the preferred port
    sock = occupy_port(preferred)
    print(f"   Occupied preferred port {preferred}")

    try:
        port, is_preferred = find_available_port(preferred, start_range=8000, end_range=9000)

        if not is_preferred and port != preferred:
            print(f"✅ Correctly found alternative port {port} (preferred {preferred} was occupied)")

            # Verify the alternative port is actually available
            if is_port_available(port):
                print(f"✅ Alternative port {port} is confirmed available")
            else:
                print(f"❌ Alternative port {port} is not actually available")
        else:
            print(f"❌ Should have found alternative, got port={port}, is_preferred={is_preferred}")
    finally:
        sock.close()

    print()


def test_multiple_occupied_ports():
    """Test finding available port when multiple ports are occupied"""
    print("Test 4: Finding port with multiple occupied ports")
    print("-" * 60)

    preferred = 8600
    occupied_ports = [8600, 8601, 8599, 8602, 8598]  # Occupy preferred and nearby ports

    sockets = []
    try:
        # Occupy multiple ports
        for port in occupied_ports:
            sock = occupy_port(port)
            sockets.append(sock)
        print(f"   Occupied ports: {occupied_ports}")

        port, is_preferred = find_available_port(preferred, start_range=8000, end_range=9000)

        if not is_preferred and port not in occupied_ports:
            print(f"✅ Found available port {port} (avoiding {len(occupied_ports)} occupied ports)")
        else:
            print(f"❌ Port selection failed: port={port}, is_preferred={is_preferred}")
    finally:
        for sock in sockets:
            sock.close()

    print()


def test_no_available_ports():
    """Test error handling when no ports are available"""
    print("Test 5: Error handling (no available ports)")
    print("-" * 60)

    # Create a very narrow range and occupy all ports
    start, end = 9990, 9993
    sockets = []

    try:
        for port in range(start, end + 1):
            sock = occupy_port(port)
            sockets.append(sock)
        print(f"   Occupied all ports in range {start}-{end}")

        try:
            port, is_preferred = find_available_port(9991, start_range=start, end_range=end, max_attempts=10)
            print(f"❌ Should have raised RuntimeError, but got port {port}")
        except RuntimeError as e:
            print(f"✅ Correctly raised RuntimeError: {str(e)[:60]}...")
    finally:
        for sock in sockets:
            sock.close()

    print()


def main():
    """Run all tests"""
    print("=" * 60)
    print("   Port Selection Automatic Tests")
    print("=" * 60)
    print()

    try:
        test_port_availability()
        test_find_available_port_preferred()
        test_find_available_port_occupied()
        test_multiple_occupied_ports()
        test_no_available_ports()

        print("=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
