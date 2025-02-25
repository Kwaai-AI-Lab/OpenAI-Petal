import os
import base58
import hashlib
import multihash
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


def generate_identity(identity_path: str) -> None:
    """Generates an RSA private key and stores it in a protobuf format."""
    private_key = RSAPrivateKey()
    
    # Create protobuf object
    protobuf = crypto_pb2.PrivateKey(
        key_type=crypto_pb2.KeyType.RSA,
        data=private_key.to_bytes()
    )

    try:
        with open(identity_path, "wb") as f:
            f.write(protobuf.SerializeToString())

        # Secure the file permissions (readable only by owner)
        os.chmod(identity_path, 0o400)
        print(f"Private key successfully saved to {identity_path} in protobuf format")

    except IOError as e:
        raise IOError(f"Failed to write private key to `{identity_path}`: {e}")


def get_public_key(identity_path: str) -> str:
    """Loads a private key from a protobuf file and returns the public key as a string."""
    try:
        with open(identity_path, "rb") as f:
            private_key_data = f.read()

        # Parse the protobuf
        protobuf = crypto_pb2.PrivateKey()
        protobuf.ParseFromString(private_key_data)

        if protobuf.key_type != crypto_pb2.KeyType.RSA:
            raise ValueError("Invalid key type in protobuf data.")

        # Load private key from the protobuf data
        private_key = RSAPrivateKey.from_bytes(protobuf.data)
        public_key = private_key.get_public_key()

        # Get public key as a Multihash Base58-encoded string
        multihash_b58 = public_key.to_multihash_base58()

        return  f"{multihash_b58}"

    except FileNotFoundError:
        raise FileNotFoundError(f"Private key file `{identity_path}` not found.")
    except ValueError as e:
        raise ValueError(f"Failed to load private key from protobuf: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and retrieve RSA keys in protobuf format with Multihash.")
    parser.add_argument("action", choices=["generate", "get-public"], help="Action to perform")
    parser.add_argument("path", help="Path to the private key file")

    args = parser.parse_args()

    if args.action == "generate":
        generate_identity(args.path)
    elif args.action == "get-public":
        result = get_public_key(args.path)
        print(result)
