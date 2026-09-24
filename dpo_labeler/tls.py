import ssl
import subprocess
from pathlib import Path


def context(directory: Path) -> ssl.SSLContext:
    directory.mkdir(parents=True, exist_ok=True)
    certificate, key = directory / "certificate.pem", directory / "private-key.pem"
    if not certificate.exists() or not key.exists():
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(certificate), "-days", "3650",
            "-subj", "/CN=DPO Labeler local development",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ], check=True, capture_output=True)
        key.chmod(0o600)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certificate, key)
    return ctx
