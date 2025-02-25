#!/bin/bash

# Ensure KWAAI_SECRET_KEY is set (contains the Base64 private key)
if [ -z "$KWAAI_SECRET_KEY" ]; then
    echo "ERROR: Environment variable KWAAI_SECRET_KEY is not set!"
    exit 1
fi

echo "✅ Retrieving private key from secret..."

# Extract public key from the secret
#PUBLIC_KEY=$(python3 /app/check_identity.py get-public-secret "$KWAAI_SECRET_KEY")
#if [ $? -ne 0 ]; then
#    echo "ERROR: Failed to extract public key from secret"
#    exit 1
#fi

#echo "✅ Multihash Public Key (Base58): $PUBLIC_KEY"

# Decode the private key from Base64 and write to a temporary file
echo "$KWAAI_SECRET_KEY" | base64 -d > /tmp/private_key.bin
chmod 400 /tmp/private_key.bin

# Start Petals DHT daemon using the in-memory key
exec python -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/8000 /ip6/::/tcp/8000 --identity_path "/tmp/private_key.bin"
