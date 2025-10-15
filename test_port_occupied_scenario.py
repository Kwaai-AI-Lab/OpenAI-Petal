#!/usr/bin/env python3
"""
Practical test: Occupy port 8080 and verify kwaainet finds an alternative
"""

import socket
import subprocess
import time
import sys

def occupy_port(port):
    """Create a socket that occupies a port"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.listen(1)
    return sock

def main():
    print("=" * 70)
    print("  Practical Test: Port 8080 Occupied Scenario")
    print("=" * 70)
    print()

    test_port = 8090
    print(f"Step 1: Occupying port {test_port} to simulate conflict...")
    sock = occupy_port(test_port)
    print(f"✅ Port {test_port} is now occupied")
    print()

    print(f"Step 2: Checking if port {test_port} is actually unavailable...")
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        test_sock.bind(('0.0.0.0', test_port))
        print(f"❌ Port {test_port} should be occupied but appears available")
        test_sock.close()
        sock.close()
        return 1
    except OSError:
        print(f"✅ Confirmed: Port {test_port} is not available")
        test_sock.close()
    print()

    print("Step 3: Simulating kwaainet trying to use this port...")
    print(f"   (kwaainet should detect port {test_port} is occupied and find alternative)")
    print()

    # Import and test the port selection logic directly
    sys.path.insert(0, 'Installer/macOS')
    from kwaainet.utils import find_available_port

    try:
        selected_port, is_preferred = find_available_port(test_port, start_range=8000, end_range=9000)

        if not is_preferred and selected_port != test_port:
            print(f"✅ SUCCESS: kwaainet would use alternative port {selected_port}")
            print(f"   (Original port {test_port} was occupied)")
            print()
            print("   The user would see:")
            print(f"   ⚠️  Port {test_port} was not available, using alternate port {selected_port}")
            print(f"   💡 To use this port permanently, update config: kwaainet config --set port {selected_port}")
        else:
            print(f"❌ FAILED: Expected alternative port, got {selected_port} (is_preferred={is_preferred})")
            sock.close()
            return 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sock.close()
        return 1

    sock.close()
    print()
    print("=" * 70)
    print("✅ Practical test completed successfully!")
    print("   Port conflict detection and fallback working correctly.")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
