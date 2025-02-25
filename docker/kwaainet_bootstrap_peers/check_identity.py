import os
import base64
import base58
import multihash
import hashlib
from hivemind.proto import crypto_pb2 
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


class RSAPrivateKey:
    def __init__(self):
        """Generate a new RSA private key."""
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def get_public_key(self):
        """Returns the corresponding public key."""
        return RSAPublicKey(self._private_key.public_key())

    def to_bytes(self) -> bytes:
        """Serializes the private key to bytes using DER format (for protobuf storage)."""
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @classmethod
    def from_bytes(cls, key_bytes: bytes):
        """Loads a private key from bytes."""
        private_key = serialization.load_der_private_key(key_bytes, password=None)
        instance = cls.__new__(cls)
        instance._private_key = private_key
        return instance


class RSAPublicKey:
    def __init__(self, public_key: rsa.RSAPublicKey):
        """Initialize with an existing RSA public key."""
        self._public_key = public_key

    def to_bytes(self) -> bytes:
        """Serializes the public key in DER SubjectPublicKeyInfo format."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    
    def to_multihash_base58(self) -> str:
        """Returns the public key as a Multihash Base58-encoded string (Libp2p Peer ID format)."""
        encoded_public_key = self.to_bytes()

        # Wrap in Protobuf PublicKey message
        protobuf_public_key = crypto_pb2.PublicKey(
            key_type=crypto_pb2.KeyType.RSA, data=encoded_public_key
        ).SerializeToString()

        # Compute SHA2-256 hash of the protobuf-wrapped public key
        sha256_hash = hashlib.sha256(protobuf_public_key).digest()

        # Encode the hash using multihash
        encoded_digest = multihash.encode(sha256_hash, multihash.coerce_code("sha2-256"))

        # Convert to Base58
        return base58.b58encode(encoded_digest).decode()


def encode_private_key(identity_path: str) -> str:
    """Encodes a binary protobuf private key file into Base64 for cloud storage."""
    try:
        with open(identity_path, "rb") as f:
            private_key_data = f.read()

        return base64.b64encode(private_key_data).decode()

    except FileNotFoundError:
        raise FileNotFoundError(f"Private key file `{identity_path}` not found.")


def decode_private_key(base64_key: str) -> bytes:
    """Decodes a Base64-encoded protobuf private key back into binary format."""
    return base64.b64decode(base64_key)


def get_public_key_from_secret(secret_value: str) -> str:
    """Loads a Base64-encoded private key from a secret and extracts the public key."""
    try:
        private_key_data = decode_private_key(secret_value)

        # Parse the protobuf
        protobuf = crypto_pb2.PrivateKey()
        protobuf.ParseFromString(private_key_data)

        if protobuf.key_type != crypto_pb2.KeyType.RSA:
            raise ValueError("Invalid key type in protobuf data.")

        # Load private key from the protobuf data
        private_key = RSAPrivateKey.from_bytes(protobuf.data)
        public_key = private_key.get_public_key()

        # Get public key as a Multihash Base58-encoded string
        return public_key.to_multihash_base58()

    except ValueError as e:
        raise ValueError(f"Failed to load private key from secret: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Handle protobuf-encoded private keys securely in cloud secrets.")
    parser.add_argument("action", choices=["encode", "get-public-secret"], help="Action to perform")
    parser.add_argument("data", help="Path to private key file or Base64 secret value")

    args = parser.parse_args()

    if args.action == "encode":
        result = encode_private_key(args.data)        
        print(result)

    elif args.action == "get-public-secret":
        result = get_public_key_from_secret(args.data)        
        print(result)
