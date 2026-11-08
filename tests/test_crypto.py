import os
import stat

import pytest

from genesis import crypto

# RFC 8032, section 7.1, tests 1-3.
VECTORS = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe"
     "24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302a"
     "eeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed2"
     "8dc027beceea1ec40a"),
]


@pytest.mark.parametrize("seed,pub,msg,sig", VECTORS)
def test_rfc8032_vectors(seed, pub, msg, sig):
    seed, pub, msg, sig = (bytes.fromhex(x) for x in (seed, pub, msg, sig))
    assert crypto.public_key(seed) == pub
    assert crypto.sign(seed, msg) == sig
    assert crypto.verify(pub, msg, sig)


def test_tampering_is_rejected():
    seed, pub, msg, sig = (bytes.fromhex(x) for x in VECTORS[2])
    assert not crypto.verify(pub, msg + b"\x00", sig)
    assert not crypto.verify(pub, msg, sig[:-1] + bytes([sig[-1] ^ 1]))
    assert not crypto.verify(bytes.fromhex(VECTORS[0][1]), msg, sig)
    # S + L is the same scalar mod L but non-canonical: malleability must be refused.
    s = int.from_bytes(sig[32:], "little") + crypto.L
    assert not crypto.verify(pub, msg, sig[:32] + s.to_bytes(32, "little"))
    assert not crypto.verify(pub, msg, b"short")


def test_statements_are_bound_to_their_purpose_and_content():
    seed = bytes(range(32))
    pub = crypto.public_key(seed)
    signed = crypto.sign_statement(seed, "policy", {"serial": 1, "roles": {"admin": ["*"]}})
    assert crypto.verify_statement(pub, signed, "policy")
    assert not crypto.verify_statement(pub, signed, "audit-seal")
    assert not crypto.verify_statement(pub, {**signed, "purpose": "audit-seal"}, "audit-seal")
    assert not crypto.verify_statement(pub, {**signed, "statement": {"serial": 2, "roles": {"admin": ["*"]}}},
                                       "policy")
    assert not crypto.verify_statement(pub, {**signed, "signature": "zz"}, "policy")
    assert not crypto.verify_statement(crypto.public_key(bytes(32)), signed, "policy")
    assert not crypto.verify_statement(pub, None, "policy")


def test_key_files(tmp_path):
    priv, pub = str(tmp_path / "keys" / "admin.key"), str(tmp_path / "keys" / "admin.pub")
    kid = crypto.generate_keypair(priv, pub)
    assert stat.S_IMODE(os.stat(priv).st_mode) == 0o600
    seed = crypto.load_private(priv)
    assert crypto.key_id(crypto.load_public(pub)) == kid == crypto.key_id(crypto.public_key(seed))
    with pytest.raises(FileExistsError):
        crypto.generate_keypair(priv, pub)
    os.chmod(priv, 0o644)
    with pytest.raises(PermissionError):
        crypto.load_private(priv)
    with pytest.raises(ValueError):
        crypto.load_public("02" + "00" * 31)  # y = 2 is not on the curve
