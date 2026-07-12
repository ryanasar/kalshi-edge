"""
src/pipeline/auth.py

Kalshi API authentication via RSA-PSS request signing.

Every request to Kalshi is cryptographically signed. There is no bearer token.
Kalshi stores only the PUBLIC half of your key, so a breach of their database
cannot yield anything that impersonates you. The private half never leaves this
machine.

The protocol, in full:

    canonical string = timestamp_ms + METHOD + path
    signature        = base64( RSA-PSS-SHA256( canonical_string, private_key ) )

    sent as three headers:
        KALSHI-ACCESS-KEY        -> who you claim to be (key ID, not secret)
        KALSHI-ACCESS-TIMESTAMP  -> when you claim it (also inside the signature)
        KALSHI-ACCESS-SIGNATURE  -> proof

The server looks up your public key by key ID, rebuilds the same canonical
string from the timestamp header and the request line, and verifies. Any byte
of disagreement -> a bare 401 with no explanation.

Setup:
    pip install cryptography requests
    export KALSHI_API_KEY_ID="your-key-id-uuid"
    export KALSHI_PRIVATE_KEY_PATH="./secrets/kalshi_private_key.pem"

Run the credential smoke test:
    python auth.py
"""

from __future__ import annotations

import base64
import os
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from dotenv import load_dotenv

load_dotenv()

# --- Config -------------------------------------------------------------------

API_PREFIX = "/trade-api/v2"

# Kalshi's docs are genuinely inconsistent about the base host across guides.
# Rather than trust one, the smoke test below probes these in order and reports
# which actually authenticates. Once you know, hardcode the winner and delete
# the rest.
CANDIDATE_HOSTS = [
    "https://api.elections.kalshi.com",
    "https://trading-api.kalshi.com",
    "https://api.kalshi.com",
]

DEMO_HOST = "https://demo-api.kalshi.co"  # sandbox; separate credentials


class KalshiAuthError(RuntimeError):
    pass


# --- Client -------------------------------------------------------------------


class KalshiClient:
    """Signs and sends authenticated requests to Kalshi."""
    def __init__(self, key_id: str, private_key: rsa.RSAPrivateKey, host: str):
        self.key_id = key_id
        self.private_key = private_key
        self.host = host
        self.session = requests.Session()

    @classmethod
    def from_env(cls, host: str) -> "KalshiClient":
        # Bracket access, not os.getenv(): raise KeyError loudly at startup if a
        # credential is missing, rather than returning None and detonating later
        # somewhere confusing. Fail at the boundary.
        key_id = os.environ["KALSHI_API_KEY_ID"]
        key_path = os.environ["KALSHI_PRIVATE_KEY_PATH"]

        # "rb" — the PEM parser wants bytes, not a decoded str.
        # password=None — the key is not encrypted at rest. That's a real
        # tradeoff: an encrypted key needs a passphrase at load time, which for
        # an unattended pipeline just relocates the secret rather than removing
        # it. Filesystem permissions + .gitignore is the proportionate answer here.
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)

        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise KalshiAuthError(
                f"Expected an RSA private key, got {type(private_key).__name__}. "
                "Kalshi issues RSA keys; an EC or Ed25519 key means you grabbed "
                "the wrong file."
            )

        return cls(key_id, private_key, host)

    # --- The signer: the load-bearing part ------------------------------------

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        # Kalshi signs the PATH ONLY — the query string is excluded. Get this
        # wrong and auth works fine on /portfolio/balance (no query string to
        # chop) then breaks the moment you add pagination to /markets.
        # split("?")[0] returns the string untouched when there's no "?", so one
        # line covers both cases.
        path_to_sign = path.split("?")[0]

        # The canonical string. No separators — the pieces are jammed directly
        # together, because that is exactly what the server does on its end.
        # .upper() because "get" and "GET" are different bytes and therefore
        # hash differently; the server always builds its string with the
        # uppercase verb. This is canonicalization: normalize so both sides agree.
        # .encode() because crypto primitives operate on bytes, not str.
        message = f"{timestamp_ms}{method.upper()}{path_to_sign}".encode()

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                # THE silent-401 line. Kalshi expects 32 bytes of salt (= the
                # SHA-256 digest length). cryptography's other constant,
                # MAX_LENGTH, produces a perfectly VALID signature that Kalshi
                # REJECTS. Two plausible constants, one right, no error message
                # telling you which.
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            # You sign the HASH, not the message: RSA can only operate on numbers
            # smaller than the key modulus, so a 4KB body could never be signed
            # directly. Hash to 32 bytes, sign that.
            hashes.SHA256(),
        )

        # Raw signature bytes are arbitrary and non-printable — they cannot go in
        # an HTTP header. Base64 maps them onto a safe ASCII alphabet; .decode()
        # turns those base64 bytes back into a str, because requests wants str
        # header values. bytes -> base64 bytes -> str.
        return base64.b64encode(signature).decode()

    # --- Request wrapper ------------------------------------------------------

    def request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json: dict | None = None,
        timeout: int = 10,
    ) -> requests.Response:
        """
        `endpoint` is the part AFTER the API prefix, e.g. "/portfolio/balance".

        The full path is assembled HERE, in one place, so that the path we sign
        and the path we send are physically incapable of drifting apart. That
        drift is the most common auth bug in the wild, and this design makes it
        unrepresentable.
        """
        path = API_PREFIX + endpoint

        # Milliseconds, not seconds. Forgetting the *1000 gives a timestamp a
        # thousand times too small, which reads as 1970 to Kalshi's clock, which
        # is wildly outside their freshness window -> silent 401. The #1 auth bug.
        timestamp_ms = str(int(time.time() * 1000))

        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp_ms, method, path),
            "Content-Type": "application/json",
        }

        # The timestamp is both SIGNED and SENT. Signed so it can't be tampered
        # with; sent so the server knows which value to rebuild the string with.
        return self.session.request(
            method.upper(),
            self.host + path,
            headers=headers,
            params=params,
            json=json,
            timeout=timeout,
        )


# --- Credential smoke test ----------------------------------------------------


def smoke_test() -> None:
    """
    Probes each candidate host with an AUTHENTICATED endpoint.

    /portfolio/balance specifically, NOT /markets. /markets is public — it
    returns 200 even with a garbage signature, so it proves nothing. Only an
    endpoint that REQUIRES auth can tell you your signature was accepted.

    A balance of 0 is a pass. You're testing the handshake, not your account.
    """
    print("Probing hosts with GET /portfolio/balance ...\n")

    for host in CANDIDATE_HOSTS:
        try:
            client = KalshiClient.from_env(host)
            resp = client.request("GET", "/portfolio/balance")
        except requests.exceptions.RequestException as e:
            print(f"  {host:<40} CONNECTION FAILED ({type(e).__name__})")
            continue

        print(f"  {host:<40} HTTP {resp.status_code}")

        if resp.status_code == 200:
            print(f"\n{'=' * 70}")
            print("AUTHENTICATED. Signature accepted.")
            print(f"  Working host: {host}")
            print(f"  Response:     {resp.text}")
            print(f"{'=' * 70}")
            print("\nHardcode that host, delete the other candidates, and move on")
            print("to GET /markets?series_ticker=KXFED to confirm live field names")
            print("and the dollar-string price format.")
            return

        if resp.status_code == 401:
            print(f"      -> reachable, but signature REJECTED: {resp.text[:200]}")
        elif resp.status_code == 404:
            print("      -> wrong host or wrong API prefix (not a crypto problem)")

    print(f"\n{'=' * 70}")
    print("NO HOST AUTHENTICATED.")
    print(f"{'=' * 70}")
    print(
        "\nDistinguish the two failure classes first:\n"
        "  - 404 / connection error  = the HOST is wrong. Your crypto is untested.\n"
        "  - 401                     = you reached Kalshi; your SIGNATURE was rejected.\n"
        "\nIf you got a 401, check in this order:\n"
        "  1. Timestamp in MILLISECONDS (time.time() * 1000), not seconds.\n"
        "  2. Signed path includes '/trade-api/v2' and EXCLUDES the '?query'.\n"
        "  3. salt_length is PSS.DIGEST_LENGTH, not PSS.MAX_LENGTH.\n"
        "  4. KALSHI_API_KEY_ID matches the key whose PRIVATE half you loaded.\n"
        "     (Mismatched pairs are common if you've regenerated keys.)\n"
        "  5. System clock isn't skewed. The timestamp is inside the signature and\n"
        "     Kalshi compares it to THEIR clock — a few seconds of drift and every\n"
        "     signature you produce is rejected as stale, even with perfect code.\n"
        "\nIf every host 404'd, check docs.kalshi.com for the current base URL.\n"
    )


if __name__ == "__main__":
    smoke_test()