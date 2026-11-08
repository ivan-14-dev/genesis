"""Administration commands: ``python -m genesis {admin,audit,corpus,lm} ...``.

Trust model: whoever holds the admin private key (default ``~/.genesis/admin.key``, mode 0600) sets the rules.
Its public key is pinned in each memory; policies, corpus approvals, model activations, version deployments and
audit seals are statements signed with it, so write access to the database alone cannot forge them.
"""
from __future__ import annotations

import json
import os
import sys

DEFAULT_KEY_DIR = os.path.join(os.path.expanduser("~"), ".genesis")


def _key_paths(args) -> tuple:
    d = args.key_dir
    return os.path.join(d, "admin.key"), os.path.join(d, "admin.pub")


def _seed(args) -> bytes:
    from . import crypto
    return crypto.load_private(args.key or _key_paths(args)[0])


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _open(args, create: bool = False):
    from .memory import LongTermMemory
    if not create and not os.path.exists(args.db):
        raise FileNotFoundError(f"no memory at {args.db}")
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    return LongTermMemory(args.db)


def _guard(ltm, actor: str):
    from . import governance
    return governance.Guard(governance.load_policy(ltm), ltm, actor or "cli", "admin")


def add_parsers(sub) -> None:
    def common(p, key: bool = False, db: bool = True):
        if db:
            p.add_argument("--db", default="runs/demo.db")
        if key:
            p.add_argument("--key", help="admin private key file (default: <key-dir>/admin.key)")
        p.add_argument("--key-dir", default=DEFAULT_KEY_DIR)
        return p

    adm = sub.add_parser("admin", help="admin key, access tokens and signed policy").add_subparsers(
        dest="sub", required=True)
    common(adm.add_parser("keygen", help="create the admin key pair (never overwrites)"), db=False)
    pin = common(adm.add_parser("pin", help="pin the admin public key in a memory (created if missing)"))
    pin.add_argument("--pub", help="public key file or hex (default: <key-dir>/admin.pub)")
    tok = adm.add_parser("token", help="new access token + the user entry to paste into the policy")
    tok.add_argument("--name", required=True)
    tok.add_argument("--role", default="user")
    tpl = adm.add_parser("policy-template", help="print the default policy as an editable JSON file")
    tpl.add_argument("--serial", type=int, default=1)
    inst = common(adm.add_parser("policy-install", help="validate, sign and install a policy file"), key=True)
    inst.add_argument("--file", required=True)
    common(adm.add_parser("policy-show", help="installed policy (signature checked)"))

    aud = sub.add_parser("audit", help="tamper-evident audit log").add_subparsers(dest="sub", required=True)
    v = common(aud.add_parser("verify", help="check the hash chain, seals, corpus and models"))
    v.add_argument("--pub", help="expected admin public key (file or hex)")
    common(aud.add_parser("seal", help="sign the current head of the audit chain"), key=True)
    lg = common(aud.add_parser("log", help="print audit entries"))
    lg.add_argument("--since", type=int, default=0)
    lg.add_argument("--limit", type=int, default=50)

    cor = sub.add_parser("corpus", help="documents the in-house language model may learn from").add_subparsers(
        dest="sub", required=True)
    add = common(cor.add_parser("add", help="submit text files (pending until the admin approves them)"))
    add.add_argument("files", nargs="+", help="UTF-8 text files ('-' for stdin)")
    add.add_argument("--source", default="local")
    add.add_argument("--license", default="unspecified")
    add.add_argument("--by", default="cli")
    ls = common(cor.add_parser("list"))
    ls.add_argument("--status", choices=("pending", "approved", "rejected"))
    for name in ("approve", "reject"):
        p = common(cor.add_parser(name, help=f"{name} documents (signed with the admin key)"), key=True)
        p.add_argument("ids", nargs="+", type=int)
        p.add_argument("--by", required=True, help="name of the deciding human")
    common(cor.add_parser("verify", help="detect approved documents altered after approval"))

    lm = sub.add_parser("lm", help="in-house language models (trained here, nothing pretrained)").add_subparsers(
        dest="sub", required=True)
    tr = common(lm.add_parser("train", help="train on approved train documents"))
    tr.add_argument("--kind", choices=("ngram", "gpt"), default="ngram")
    tr.add_argument("--by", default="cli")
    for opt in ("order", "vocab-size", "dim", "n-layers", "n-heads", "block-size", "steps", "batch-size", "seed"):
        tr.add_argument(f"--{opt}", type=int)
    tr.add_argument("--lr", type=float)
    common(lm.add_parser("list"))
    apv = common(lm.add_parser("approve", help="sign and activate a model"), key=True)
    apv.add_argument("id", type=int)
    apv.add_argument("--by", required=True)
    apv.add_argument("--force", action="store_true", help="activate even without beating the unigram baseline")
    dea = common(lm.add_parser("deactivate", help="turn text generation off"), key=True)
    dea.add_argument("--by", required=True)
    ev = common(lm.add_parser("eval", help="final measurement on the sealed test split (audited)"))
    ev.add_argument("id", type=int)
    gen = common(lm.add_parser("generate", help="continue a prompt with the active model"))
    gen.add_argument("prompt")
    gen.add_argument("--max-tokens", type=int, default=100)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--temperature", type=float, default=0.8)


def _admin(args) -> int:
    from . import crypto, governance
    priv, pub_path = _key_paths(args) if hasattr(args, "key_dir") else (None, None)
    if args.sub == "keygen":
        kid = crypto.generate_keypair(priv, pub_path)
        _out({"key_id": kid, "private_key": priv, "public_key": pub_path,
              "next": f"python -m genesis admin pin --db <memory> --pub {pub_path}"})
    elif args.sub == "pin":
        ltm = _open(args, create=True)
        try:
            _out({"pinned": governance.pin_admin_key(ltm, crypto.load_public(args.pub or pub_path)),
                  "memory_id": governance.memory_id(ltm)})
        finally:
            ltm.close()
    elif args.sub == "token":
        token, digest = governance.new_token()
        _out({"token": token, "give_to_user": f"http://127.0.0.1:8766/?token={token}",
              "policy_users_entry": {"name": args.name, "role": args.role, "token_sha256": digest},
              "note": "only the hash goes in the policy; the token is shown once"})
    elif args.sub == "policy-template":
        doc = {**governance.DEFAULT_POLICY, "serial": args.serial, "name": "my policy"}
        _out(doc)
    elif args.sub == "policy-install":
        with open(args.file, encoding="utf-8") as fh:
            doc = json.load(fh)
        signed = governance.sign_policy(_seed(args), doc)
        ltm = _open(args)
        try:
            _out(governance.install_policy(ltm, signed).summary())
        finally:
            ltm.close()
    elif args.sub == "policy-show":
        ltm = _open(args)
        try:
            _out(governance.load_policy(ltm).summary())
        finally:
            ltm.close()
    return 0


def _audit(args) -> int:
    from . import corpus, crypto, governance
    from .lm import registry
    ltm = _open(args)
    try:
        if args.sub == "verify":
            pub = crypto.load_public(args.pub) if args.pub else None
            if pub is not None and governance.pinned_key(ltm) != pub:
                _out({"ok": False, "error": "the pinned admin key is not the expected one"})
                return 2
            rep = {"audit": governance.verify_audit(ltm, pub), "corpus": corpus.verify(ltm),
                   "language_models": registry.verify(ltm)}
            try:
                governance.load_policy(ltm, pub)
                rep["policy"] = {"ok": True}
            except governance.PolicyIntegrityError as exc:
                rep["policy"] = {"ok": False, "error": str(exc)}
            rep["ok"] = all(part["ok"] for part in rep.values())
            _out(rep)
            return 0 if rep["ok"] else 2
        if args.sub == "seal":
            _out({**governance.seal_audit(ltm, _seed(args)),
                  "note": "keep this head hash outside the database: seals inside it can be deleted"})
        elif args.sub == "log":
            _out(ltm.audit_entries(args.since, max(1, min(args.limit, 10000))))
    finally:
        ltm.close()
    return 0


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _corpus(args) -> int:
    from . import corpus
    ltm = _open(args, create=args.sub == "add")
    try:
        if args.sub == "add":
            guard, results, code = _guard(ltm, args.by), [], 0
            for path in args.files:
                try:
                    res = corpus.submit(ltm, guard, os.path.basename(path) if path != "-" else "stdin",
                                        _read(path), args.source, args.license)
                    results.append({"file": path, **res})
                except (ValueError, PermissionError, OSError) as exc:
                    results.append({"file": path, "refused": str(exc)})
                    code = 1
            _out(results)
            return code
        if args.sub == "list":
            _out(corpus.documents(ltm, args.status))
        elif args.sub in ("approve", "reject"):
            seed = _seed(args)
            _out([corpus.review(ltm, seed, i, args.sub == "approve", args.by) for i in args.ids])
        elif args.sub == "verify":
            rep = corpus.verify(ltm)
            _out(rep)
            return 0 if rep["ok"] else 2
    finally:
        ltm.close()
    return 0


def _lm(args) -> int:
    from .lm import registry
    ltm = _open(args)
    try:
        if args.sub == "train":
            cfg = {k: getattr(args, k) for k in ("order", "vocab_size", "dim", "n_layers", "n_heads", "block_size",
                                                 "steps", "batch_size", "seed", "lr")}
            cfg = {k: v for k, v in cfg.items() if v is not None and k in registry.DEFAULTS[args.kind]}
            log = (lambda r: print(f"step {r['step']}: loss {r['loss']:.4f}", file=sys.stderr)
                   if r["step"] % 25 == 0 else None) if args.kind == "gpt" else None
            res = registry.train(ltm, _guard(ltm, args.by), args.kind, log=log, **cfg)
            _out({k: v for k, v in res.items() if k != "provenance"} | {"provenance": {
                k: v for k, v in res["provenance"].items() if k != "corpus"}})
        elif args.sub == "list":
            _out(registry.models(ltm))
        elif args.sub == "approve":
            _out(registry.approve(ltm, _seed(args), args.id, args.by, force=args.force))
        elif args.sub == "deactivate":
            _out({"deactivated": registry.deactivate(ltm, _seed(args), args.by)})
        elif args.sub == "eval":
            _out(registry.evaluate(ltm, args.id))
        elif args.sub == "generate":
            from . import governance
            model = registry.load_active(ltm)
            if model is None:
                print("no active language model", file=sys.stderr)
                return 1
            limit = governance.load_policy(ltm).limit("max_generate_tokens")
            out = model.generate(args.prompt, max(1, min(args.max_tokens, limit)), args.seed, args.temperature)
            _out({"model": model.describe(), "unverified": True, **out})
    finally:
        ltm.close()
    return 0


def run(args) -> int:
    from . import governance
    from .lm.registry import LMIntegrityError
    handlers = {"admin": _admin, "audit": _audit, "corpus": _corpus, "lm": _lm}
    try:
        return handlers[args.cmd](args)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError, KeyError,
            governance.PolicyIntegrityError, LMIntegrityError) as exc:
        print(f"{args.cmd} {args.sub}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
