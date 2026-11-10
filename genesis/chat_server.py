"""ChatGPT-style web entry point to GENESIS (stdlib only).

``python -m genesis chat --db runs/demo.db`` serves a single page and a small JSON API:

    GET  /api/status                         stage and memory counts
    GET  /api/whoami                         identity, role and permissions under the admin policy
    POST /api/login                          {"user": "...", "password": "..."} -> session cookie
    POST /api/logout                         clears the session cookie
    GET  /api/conversations                  the caller's conversations (most recent first)
    POST /api/conversations                  {"lang": "fr"|"en"} -> new conversation
    GET  /api/conversations/<id>             messages of one of the caller's conversations
    POST /api/conversations/<id>/messages    {"text": "...", "lang": "fr"|"en"} -> agent reply

Unlike the read-only dashboard, this interface *does* let the agent act and learn, so it holds one
``Genesis`` instance for the life of the server; requests are served on threads but the engine is used by one
at a time. Defences, in the order a request meets them:

* **Host allowlist** (127.0.0.1 / localhost / [::1] on the bound port, plus ``--allow-host``): defeats DNS
  rebinding, where a hostile page re-points its own name at 127.0.0.1.
* **Same-origin writes**: POSTs must be JSON, and Origin / Sec-Fetch-Site must not be cross-site.
* **Identity** from the admin policy (``governance.py``): a bearer token (``Authorization`` header, or
  ``/?token=…`` once, which sets an HttpOnly SameSite=Strict cookie), or the login form (user name + password,
  the password being that user's token; ``genesis init`` generates one for the administrator). Without a token
  the policy's default role applies, unless the policy requires authentication, in which case the page shows
  the login form. Login attempts are rate-limited per client address and audited.
* **Rate limit** per identity, bounded body size and socket timeouts, a per-message time budget in the engine.
* **Policy guard** inside the engine: default deny, every refusal and write in the audit chain.
* Responses carry a strict CSP (hash-pinned inline script/style), no-store and anti-framing headers; internal
  errors return an opaque id, details go to the server log only.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qs, urlparse

from . import governance
from .agent import Genesis
from .chat import ChatEngine
from .metacognition import Metacognition

MAX_BODY = 16 * 1024
SOCKET_TIMEOUT = 10
COOKIE = "genesis_token"
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,128}")
LOGINS_PER_MINUTE = 10


def _cookie(token: str) -> str:
    return f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/" + ("" if token else "; Max-Age=0")


def status(engine: ChatEngine) -> dict:
    a = engine.agent
    st = Metacognition.current_stage(a.assessments())
    return {"stage": st["stage"], "stage_name": st["name"], "interactions": a.interactions,
            "knowledge": a.ltm.count("knowledge"),
            "validated": len(a.ltm.knowledge(status="validated")), "episodes": a.ltm.count("episodes"),
            "session": a.session}


def _hash_source(tag: str) -> str:
    body = PAGE.split(f"<{tag}>", 1)[1].split(f"</{tag}>", 1)[0]
    return "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"


def allowed_hosts(server_address, extra: Iterable[str] = ()) -> set:
    host, port = server_address[0], server_address[1]
    names = {"127.0.0.1", "localhost", "[::1]"}
    if host not in ("", "0.0.0.0", "::"):
        names.add(f"[{host}]" if ":" in host else host)
    out = set()
    for n in names | {h.lower() for h in extra}:
        if re.fullmatch(r"(\[[0-9a-f:]+\]|[^:\[\]]+):\d+", n):
            out.add(n)
            continue
        out.add(f"{n}:{port}")
        if port == 80:
            out.add(n)
    return out


class PolicyCache:
    """Re-verifies the signed policy only when it changes in memory (an admin may install one while serving)."""

    def __init__(self, ltm, expected_pub: Optional[bytes] = None):
        self.ltm, self.expected_pub, self.raw, self.policy = ltm, expected_pub, object(), None

    def get(self) -> governance.Policy:
        raw = (self.ltm.db.execute("SELECT value FROM meta WHERE key='policy'").fetchone() or [None])[0], \
            self.ltm.get_meta("admin_pubkey")
        if raw != self.raw:
            self.policy = governance.load_policy(self.ltm, self.expected_pub)  # raises if tampered
            self.raw = raw
        return self.policy


class RateLimiter:
    def __init__(self):
        self.hits, self.lock = {}, threading.Lock()

    def allow(self, who: str, per_minute: int) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits.setdefault(who, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= per_minute:
                return False
            q.append(now)
            return True


def make_handler(engine: ChatEngine, allow_hosts: Iterable[str] = (), expected_pub: Optional[bytes] = None):
    lock = threading.RLock()
    policies = PolicyCache(engine.ltm, expected_pub)
    limiter = RateLimiter()
    extra_hosts = tuple(allow_hosts)
    csp = ("default-src 'none'; script-src " + _hash_source("script") + "; style-src " + _hash_source("style")
           + "; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
           "frame-ancestors 'none'")

    class Handler(BaseHTTPRequestHandler):
        timeout = SOCKET_TIMEOUT
        server_version = "GENESIS"
        sys_version = ""

        def log_message(self, *a):
            pass

        def _send(self, code: int, data, ctype: str = "application/json", headers: Optional[dict] = None):
            body = data if isinstance(data, bytes) else json.dumps(data, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", csp)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, exc: BaseException):
            ref = secrets.token_hex(4)
            print(f"[genesis chat] error {ref}: " + "".join(traceback.format_exception(exc)), file=sys.stderr,
                  flush=True)
            return self._send(500, {"error": f"internal error (ref {ref})"})

        # ------------------------------------------------------------ checks
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").strip().lower()
            if host in allowed_hosts(self.server.server_address, extra_hosts):
                return True
            self._send(421, {"error": "unrecognised Host header"})
            return False

        def _token(self) -> Optional[str]:
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                return auth[7:].strip() or None
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE and v:
                    return v
            return None

        def _identify(self):
            """Returns a Guard for the caller, or sends the error response and returns None."""
            try:
                with lock:
                    policy = policies.get()
                    actor, role = policy.identify(self._token())
            except governance.PolicyIntegrityError as exc:
                print(f"[genesis chat] policy integrity failure: {exc}", file=sys.stderr, flush=True)
                self._send(503, {"error": "policy integrity check failed; refusing to serve"})
                return None
            except governance.AuthError as exc:
                self._send(401, {"error": str(exc)}, headers={"WWW-Authenticate": "Bearer"})
                return None
            if not limiter.allow(actor, policy.limit("requests_per_minute")):
                self._send(429, {"error": "too many requests"}, headers={"Retry-After": "60"})
                return None
            return governance.Guard(policy, engine.ltm, actor, role)

        def _call(self, guard, fn, *args):
            with lock:
                engine.guard = guard
                return fn(*args)

        # -------------------------------------------------------------- GET
        def do_GET(self):
            if not self._host_ok():
                return
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            if path in ("/", "/index.html"):
                token = (query.get("token") or [None])[0]
                if token:  # move the token out of the URL (history, logs) into an HttpOnly cookie
                    if not TOKEN_RE.fullmatch(token):
                        return self._send(401, {"error": "unknown token"})
                    try:
                        with lock:
                            policies.get().identify(token)
                    except (governance.AuthError, governance.PolicyIntegrityError):
                        return self._send(401, {"error": "unknown token"})
                    return self._send(303, b"", "text/plain", {"Location": "/", "Set-Cookie": _cookie(token)})
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if not path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            guard = self._identify()
            if guard is None:
                return

            def checked_status():
                guard.require("chat.read")
                return status(engine)
            try:
                if path == "/api/status":
                    return self._send(200, self._call(guard, checked_status))
                if path == "/api/whoami":
                    p = guard.policy
                    return self._send(200, {"user": guard.actor, "role": guard.role, "policy": p.doc["name"],
                                            "serial": p.serial, "signed": p.signed,
                                            "permissions": sorted(a for a in governance.ACTIONS if guard.allowed(a))})
                if path == "/api/conversations":
                    return self._send(200, self._call(guard, engine.conversations))
                m = re.fullmatch(r"/api/conversations/(\d{1,12})", path)
                if m:
                    def read():
                        conv = engine.conversation(int(m.group(1)))
                        return conv and {"conversation": conv, "messages": engine.messages(conv["id"])}
                    data = self._call(guard, read)
                    return self._send(200, data) if data else self._send(404, {"error": "not found"})
            except governance.PolicyDenied as exc:
                return self._send(403, {"error": str(exc)})
            except Exception as exc:
                return self._fail(exc)
            self._send(404, {"error": "not found"})

        # ------------------------------------------------------------- POST
        def _body(self) -> Optional[dict]:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._send(415, {"error": "JSON required"})
                return None
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
                self._send(403, {"error": "cross-origin request refused"})
                return None
            if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
                self._send(403, {"error": "cross-site request refused"})
                return None
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                self._send(400, {"error": "invalid Content-Length"})
                return None
            if n > MAX_BODY:
                self._send(413, {"error": "message too large"})
                return None
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, OSError):
                self._send(400, {"error": "invalid JSON"})
                return None
            if not isinstance(data, dict):
                self._send(400, {"error": "JSON object required"})
                return None
            return data

        def _login(self, data: dict):
            if not limiter.allow("login:" + self.client_address[0], LOGINS_PER_MINUTE):
                return self._send(429, {"error": "too many login attempts"}, headers={"Retry-After": "60"})
            name, password = data.get("user"), data.get("password")
            try:
                with lock:
                    try:
                        if not (isinstance(name, str) and isinstance(password, str) and len(name) <= 64
                                and TOKEN_RE.fullmatch(password)):
                            raise governance.AuthError("invalid user name or password")
                        actor, role = policies.get().authenticate(name.strip(), password)
                    except governance.AuthError as exc:
                        engine.ltm.append_audit("auth.login", "deny", "anonymous", None, None,
                                                {"reason": "invalid credentials"})
                        engine.ltm.commit()
                        return self._send(401, {"error": str(exc)})
                    engine.ltm.append_audit("auth.login", "allow", actor, role)
                    engine.ltm.commit()
            except governance.PolicyIntegrityError as exc:
                print(f"[genesis chat] policy integrity failure: {exc}", file=sys.stderr, flush=True)
                return self._send(503, {"error": "policy integrity check failed; refusing to serve"})
            return self._send(200, {"user": actor, "role": role}, headers={"Set-Cookie": _cookie(password)})

        def do_POST(self):
            if not self._host_ok():
                return
            url = urlparse(self.path)
            data = self._body()
            if data is None:
                return
            if url.path == "/api/login":
                return self._login(data)
            if url.path == "/api/logout":
                return self._send(200, {"ok": True}, headers={"Set-Cookie": _cookie("")})
            guard = self._identify()
            if guard is None:
                return
            path = url.path
            try:
                if path == "/api/conversations":
                    return self._send(200, self._call(guard, engine.new_conversation, data.get("lang")))
                m = re.fullmatch(r"/api/conversations/(\d{1,12})/messages", path)
                if m:
                    text = data.get("text")
                    if not isinstance(text, str) or not text.strip():
                        return self._send(400, {"error": "text required"})
                    return self._send(200, self._call(guard, engine.send, int(m.group(1)), text, data.get("lang")))
            except KeyError:
                return self._send(404, {"error": "conversation not found"})
            except governance.PolicyDenied as exc:
                return self._send(403, {"error": str(exc)})
            except Exception as exc:  # report, never crash the server, never leak internals
                return self._fail(exc)
            self._send(404, {"error": "not found"})

    return Handler


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8766,
          world_factory: Optional[Callable] = None, object_factory: Optional[Callable] = None,
          allow_hosts: Iterable[str] = (), expected_pub: Optional[bytes] = None,
          on_ready: Optional[Callable[[str], None]] = None) -> None:
    agent = Genesis(db_path, check_same_thread=False)  # used by one request thread at a time (engine lock)
    try:
        governance.load_policy(agent.ltm, expected_pub)  # refuse to start under a tampered policy
        engine = ChatEngine(agent, world_factory, object_factory)
        srv = ThreadingHTTPServer((host, port), make_handler(engine, allow_hosts, expected_pub))
    except BaseException:
        agent.close()
        raise
    srv.daemon_threads = True
    shown = "127.0.0.1" if host in ("", "0.0.0.0", "::") else f"[{host}]" if ":" in host else host
    url = f"http://{shown}:{srv.server_port}/"
    print(f"GENESIS chat for {db_path} on {url}  (session {agent.session}; Ctrl+C to stop)", flush=True)
    if on_ready is not None:
        on_ready(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        agent.close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GENESIS</title>
<style>
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{display:flex;background:#212121;color:#ececec;font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
button{font:inherit;color:inherit;cursor:pointer}
#side{width:260px;flex:none;background:#171717;display:flex;flex-direction:column;padding:8px;transition:margin .2s}
#side.hidden{margin-left:-260px}
.side-top{display:flex;align-items:center;justify-content:space-between;padding:6px 6px 10px}
.icon-btn{background:none;border:0;border-radius:8px;padding:6px;display:flex;color:#b4b4b4}
.icon-btn:hover{background:#2f2f2f;color:#fff}
#new{display:flex;align-items:center;gap:10px;width:100%;background:none;border:0;border-radius:8px;padding:8px 10px;font-size:14px;text-align:left}
#new:hover{background:#2f2f2f}
.logo{width:26px;height:26px;border-radius:50%;flex:none;display:grid;place-items:center;font-weight:700;font-size:13px;color:#fff;
 background:conic-gradient(from 200deg,#10a37f,#3b82f6,#a855f7,#10a37f)}
.hist-label{font-size:12px;color:#8e8e8e;padding:16px 10px 6px;font-weight:600}
#convs{flex:1;overflow-y:auto}
.conv{display:block;width:100%;background:none;border:0;border-radius:8px;padding:8px 10px;font-size:14px;text-align:left;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ececec}
.conv:hover{background:#212121}.conv.active{background:#2f2f2f}
#status{border-top:1px solid #2f2f2f;padding:12px 10px 6px;font-size:12px;color:#b4b4b4}
#status b{color:#ececec;font-weight:600}
.badge{display:inline-block;margin-top:6px;padding:2px 8px;border:1px solid #3a3a3a;border-radius:999px;font-size:11px;color:#9b9b9b}
#main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
header{height:56px;flex:none;display:flex;align-items:center;gap:8px;padding:0 12px}
.model{font-size:18px;font-weight:600;padding:6px 10px;border-radius:8px}
.model span{color:#9b9b9b;font-weight:400;font-size:15px;margin-left:6px}
#thread{flex:1;overflow-y:auto}
.col{max-width:48rem;margin:0 auto;padding:0 24px}
.msg{padding:14px 0}
.msg.user{display:flex;justify-content:flex-end}
.bubble{background:#303030;border-radius:24px;padding:10px 20px;max-width:75%;white-space:pre-wrap;word-wrap:break-word}
.msg.genesis{display:flex;gap:16px;align-items:flex-start}
.msg.genesis .body{flex:1;min-width:0;padding-top:1px}
.body p{margin:0 0 12px}.body p:last-child{margin-bottom:0}
.body ul{margin:4px 0 12px;padding-left:22px}.body li{margin:2px 0}
.body code{background:#2f2f2f;border-radius:6px;padding:2px 6px;font:14px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.body em{color:#a3a3a3;font-size:14px}
.body a.kid{color:#7ab7ff;text-decoration:none;cursor:pointer}.body a.kid:hover{text-decoration:underline}
.actions{display:flex;gap:4px;margin-top:6px;align-items:center;opacity:.0;transition:opacity .15s}
.msg.genesis:hover .actions,.msg.genesis:last-child .actions{opacity:1}
.choices{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.choice{background:transparent;color:#ececec;border:1px solid #424242;border-radius:999px;padding:6px 14px;font-size:14px;cursor:pointer}
.choice:hover{background:#2f2f2f}
.chip{font-size:11px;color:#8e8e8e;border:1px solid #333;border-radius:999px;padding:1px 8px;margin-left:4px}
.dots span{display:inline-block;width:8px;height:8px;margin-right:4px;border-radius:50%;background:#b4b4b4;animation:b 1.2s infinite}
.dots span:nth-child(2){animation-delay:.2s}.dots span:nth-child(3){animation-delay:.4s}
@keyframes b{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
#empty{position:absolute;inset:56px 0 150px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;text-align:center}
#empty h1{font-size:28px;font-weight:600;margin:0 0 8px}
#empty p{color:#9b9b9b;margin:0 0 28px;max-width:34rem;font-size:15px}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;max-width:40rem;width:100%}
.card{background:none;border:1px solid #383838;border-radius:16px;padding:12px 14px;text-align:left;font-size:14px}
.card:hover{background:#2a2a2a}.card small{display:block;color:#8e8e8e}
#composer{flex:none;padding:0 0 10px}
.box{display:flex;align-items:flex-end;gap:8px;background:#303030;border-radius:28px;padding:10px 10px 10px 22px;box-shadow:0 0 0 1px #3a3a3a}
textarea{flex:1;resize:none;border:0;outline:0;background:none;color:#ececec;font:inherit;max-height:200px;padding:6px 0;line-height:1.5}
textarea::placeholder{color:#9b9b9b}
#send{width:36px;height:36px;border-radius:50%;border:0;background:#fff;color:#000;display:grid;place-items:center;flex:none}
#send:disabled{background:#676767;color:#2f2f2f;cursor:default}
.note{font-size:12px;color:#9b9b9b;text-align:center;padding-top:8px}
#menu{display:none}
#me{padding:2px 10px 8px;font-size:12px;color:#b4b4b4;display:flex;align-items:center;justify-content:space-between;gap:8px}
#me b{color:#ececec}
#me button{background:none;border:1px solid #3a3a3a;border-radius:999px;padding:2px 10px;font-size:12px;color:#b4b4b4}
#me button:hover{background:#2f2f2f;color:#fff}
#login{position:fixed;inset:0;z-index:10;background:#212121;display:flex;align-items:center;justify-content:center;padding:24px}
#login[hidden]{display:none}
#login form{width:100%;max-width:24rem;display:flex;flex-direction:column;gap:12px;text-align:center}
#login .logo{width:48px;height:48px;font-size:22px;margin:0 auto 4px}
#login h1{font-size:24px;font-weight:600;margin:0}
#login p{color:#9b9b9b;font-size:14px;margin:0}
#login label{display:flex;flex-direction:column;gap:4px;text-align:left;font-size:13px;color:#b4b4b4}
#login input{background:#303030;border:1px solid #424242;border-radius:12px;padding:10px 14px;color:#ececec;font:inherit;outline:0}
#login input:focus{border-color:#8e8e8e}
#login button{background:#fff;color:#000;border:0;border-radius:999px;padding:10px;font-weight:600;margin-top:4px}
#login button:disabled{background:#676767}
#login-err{color:#f87171;font-size:14px;min-height:1.2em}
#login .hint{font-size:12px;color:#8e8e8e;line-height:1.5}
#login code{background:#2f2f2f;border-radius:6px;padding:1px 5px;font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (max-width:768px){#side{position:fixed;z-index:5;height:100%}#side.hidden{margin-left:-260px}.cards{grid-template-columns:1fr}.bubble{max-width:90%}}
</style></head><body>
<nav id="side">
  <div class="side-top"><div class="logo">G</div>
    <button class="icon-btn" id="close" title="Close sidebar"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/></svg></button></div>
  <button id="new"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg><span data-i="new"></span></button>
  <div class="hist-label" data-i="history"></div>
  <div id="convs"></div>
  <div id="status"></div>
  <div id="me" hidden></div>
</nav>
<main id="main">
  <header><button class="icon-btn" id="menu" title="Menu"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/></svg></button>
    <div class="model">GENESIS<span data-i="model"></span></div></header>
  <div id="thread"><div class="col" id="msgs"></div></div>
  <div id="empty"><h1 data-i="hello"></h1><p data-i="sub"></p><div class="cards" id="cards"></div></div>
  <div id="composer"><div class="col">
    <div class="box"><textarea id="input" rows="1"></textarea>
      <button id="send" disabled title="Send"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg></button></div>
    <div class="note" data-i="note"></div></div></div>
</main>
<div id="login" hidden><form id="login-form" autocomplete="on">
  <div class="logo">G</div>
  <h1 data-i="loginTitle"></h1>
  <p data-i="loginSub"></p>
  <label><span data-i="userLabel"></span><input id="login-user" name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required></label>
  <label><span data-i="passLabel"></span><input id="login-pass" name="password" type="password" autocomplete="current-password" required></label>
  <div id="login-err" role="alert"></div>
  <button type="submit" id="login-btn" data-i="loginBtn"></button>
  <p class="hint" id="login-hint"></p>
</form></div>
<script>
const FR = (navigator.language||'').toLowerCase().startsWith('fr');
const LANG = FR ? 'fr' : 'en';
document.documentElement.lang = LANG;
const T = FR ? {
  new:'Nouvelle discussion', history:'Historique', model:'· aucun LLM externe', hello:'Que dois-je apprendre ?',
  sub:"Je ne sais que ce que j'ai appris par expérience. Donne-moi une suite de nombres : je prédirai chaque terme avant de le lire, puis je te dirai quelle règle j'ai trouvée, et avec quelles preuves.",
  placeholder:'Enseigne un fait, pose une question ou envoie une suite…',
  note:"GENESIS n'utilise aucun modèle de langage externe ni pré-entraîné : il comprend quelques commandes, apprend les mots que tu lui enseignes et n'affirme que ce que sa mémoire justifie. Le texte de son petit modèle interne (« complète : … ») est marqué non vérifié.",
  stage:'Stade', beliefs:'croyances', validated:'validées', episodes:'épisodes', copy:'Copier', copied:'Copié',
  untitled:'Nouvelle discussion', error:'Erreur : ', explain:'explique ',
  cards:[['3, 7, 11, 15, 19, 23, 27','une suite arithmétique'],['atelier objets','apprendre des mots à partir des objets'],
         ['Nala est un chat; Les chats sont des animaux','enseigner deux faits, sans les précharger'],['Nala est-elle un animal ?','déduire à partir des enseignements']],
  status:{validated:'validée',candidate:'candidate',unpredictable:'imprévisible',unknown:'inconnue'},
  learnedWord:'mots appris',
  intents:{clarify:'vérification',vocabulary:'vocabulaire','teach-request':'apprentissage de mots','yes-no':'oui/non',thanks:'merci',continue:'apprentissage','too-short':'trop court',sequence:'apprentissage',explain:'preuves',knowledge:'mémoire',explore:'exploration',stage:'stade',limitations:'limites',questions:'curiosité',predict:'prédiction','not-understood':'non compris',help:'aide',intro:'présentation',refused:'refusé',define:'définition',languages:'langues',smalltalk:'conversation',bye:'au revoir',declined:'refus','unknown-topic':'hors de mes connaissances'}
} : {
  new:'New chat', history:'History', model:'· no external LLM', hello:'What should I learn?',
  sub:"I only know what I learned from experience. Give me a number sequence: I will predict each term before reading it, then tell you which rule I found and on what evidence.",
  placeholder:'Teach a fact, ask a question or send a sequence…',
  note:'GENESIS uses no external or pretrained language model: it understands a few commands, learns the words you teach it, and only claims what its memory justifies. Text from its small in-house model ("complete: …") is labelled unverified.',
  stage:'Stage', beliefs:'beliefs', validated:'validated', episodes:'episodes', copy:'Copy', copied:'Copied',
  untitled:'New chat', error:'Error: ', explain:'explain ',
  cards:[['3, 7, 11, 15, 19, 23, 27','an arithmetic sequence'],['object workshop','learn words from objects'],
         ['Nala is a cat; All cats are animals','teach two facts, not preloaded'],['Is Nala an animal?','deduce from teachings']],
  learnedWord:'learned words',
  intents:{clarify:'check',vocabulary:'vocabulary','teach-request':'word learning','yes-no':'yes/no',thanks:'thanks',continue:'learning','too-short':'too short',sequence:'learning',explain:'evidence',knowledge:'memory',explore:'exploration',stage:'stage',limitations:'limitations',questions:'curiosity',predict:'prediction','not-understood':'not understood',help:'help',intro:'intro',refused:'refused',define:'definition',languages:'languages',smalltalk:'small talk',bye:'goodbye',declined:'declined','unknown-topic':'outside my knowledge'}
};
Object.assign(T, FR ? {
  loginTitle:'Connexion à GENESIS', loginSub:"Cette mémoire est protégée. Connecte-toi avec les identifiants créés à l'installation.",
  userLabel:"Nom d'utilisateur", passLabel:'Mot de passe', loginBtn:'Se connecter',
  loginHint:"Ils ont été affichés par l'installateur et enregistrés dans <code>credentials.txt</code>, dans le dossier de GENESIS (par défaut <code>~/.genesis</code>). Perdus ? <code>genesis init --reset-password</code>",
  badLogin:"Nom d'utilisateur ou mot de passe incorrect.", tooMany:'Trop de tentatives : réessaie dans une minute.',
  signedIn:'Connecté :', logout:'Se déconnecter'
} : {
  loginTitle:'Sign in to GENESIS', loginSub:'This memory is protected. Sign in with the credentials created at installation.',
  userLabel:'User name', passLabel:'Password', loginBtn:'Sign in',
  loginHint:'The installer printed them and saved them in <code>credentials.txt</code>, in the GENESIS folder (<code>~/.genesis</code> by default). Lost them? <code>genesis init --reset-password</code>',
  badLogin:'Wrong user name or password.', tooMany:'Too many attempts: try again in a minute.',
  signedIn:'Signed in:', logout:'Sign out'
});
Object.assign(T.intents, FR ? {
  'policy-denied':'refusé par la politique', 'lm-generate':'texte généré (non vérifié)'
} : {
  'policy-denied':'refused by policy', 'lm-generate':'generated text (unverified)'
});
Object.assign(T.intents, FR ? {
  'semantic-teach':'enseignement', 'semantic-query':'raisonnement', 'semantic-clarify':'précision',
  'semantic-explain':'prémisses', 'semantic-recall':'rappel', 'semantic-retract':'rétractation'
} : {
  'semantic-teach':'teaching', 'semantic-query':'reasoning', 'semantic-clarify':'clarification',
  'semantic-explain':'premises', 'semantic-recall':'recall', 'semantic-retract':'retraction'
});
Object.assign(T.intents, FR ? {
  'object-workshop':'atelier objets', 'object-explore':'expériences', 'object-observe':'objet observé',
  'object-test':'manipulation', 'object-select':'référence composée', 'object-word-teach':'sens appris',
  'object-word-query':'interprétation', 'object-word-explain':'origine du sens',
  'construction-teach':'construction apprise', 'construction-uncertain':'construction incertaine'
} : {
  'object-workshop':'object workshop', 'object-explore':'experiments', 'object-observe':'observed object',
  'object-test':'manipulation', 'object-select':'composed reference', 'object-word-teach':'learned meaning',
  'object-word-query':'interpretation', 'object-word-explain':'meaning evidence',
  'construction-teach':'learned construction', 'construction-uncertain':'uncertain construction'
});
document.querySelectorAll('[data-i]').forEach(e => e.textContent = T[e.dataset.i]);
const $ = id => document.getElementById(id);
const input = $('input'), send = $('send'), msgs = $('msgs'), thread = $('thread');
input.placeholder = T.placeholder;
let current = null, busy = false;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const d = await r.json();
  if (!r.ok) {
    if (r.status === 401 && path !== '/api/login') showLogin();
    const err = new Error(d.error || r.status); err.status = r.status; throw err;
  }
  return d;
}
function showLogin() {
  if (!$('login').hidden) return;
  $('login').hidden = false;
  $('login-user').focus();
}
$('login-hint').innerHTML = T.loginHint;
$('login-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = $('login-btn');
  btn.disabled = true; $('login-err').textContent = '';
  try {
    await api('/api/login', {user: $('login-user').value.trim(), password: $('login-pass').value.trim()});
    location.reload();
  } catch (err) {
    $('login-err').textContent = err.status === 429 ? T.tooMany : T.badLogin;
    $('login-pass').value = ''; $('login-pass').focus();
  }
  btn.disabled = false;
});
function showMe(me) {
  const box = $('me');
  box.innerHTML = '<span>' + esc(T.signedIn) + ' <b>' + esc(me.user) + '</b></span>';
  const b = document.createElement('button'); b.textContent = T.logout;
  b.onclick = async () => { try { await api('/api/logout', {}); } catch (e) {} location.reload(); };
  box.appendChild(b); box.hidden = false;
}
const esc = s => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
  return esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\s)_([^_]+)_(?=\s|$|[.,;:])/g, '$1<em>$2</em>')
    .replace(/(^|[^&\w])#(\d+)/g, '$1<a class="kid" data-k="$2">#$2</a>')
    .replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + esc(codes[+i]) + '</code>');
}
function md(text) {
  let html = '', list = null;
  const flush = () => { if (list) { html += '<ul>' + list.join('') + '</ul>'; list = null; } };
  for (const block of text.split(/\n{2,}/)) {
    flush();
    let para = [];
    for (const line of block.split('\n')) {
      const m = line.match(/^\s*- (.*)$/);
      if (m) { if (para.length) { html += '<p>' + para.join('<br>') + '</p>'; para = []; } (list = list || []).push('<li>' + inline(m[1]) + '</li>'); }
      else { flush(); const it = line.match(/^_(.+)_$/); para.push(it ? '<em>' + inline(it[1]) + '</em>' : inline(line)); }
    }
    flush();
    if (para.length) html += '<p>' + para.join('<br>') + '</p>';
  }
  return html;
}
function scrollDown() { thread.scrollTop = thread.scrollHeight; }
function addMsg(m) {
  $('empty').style.display = 'none';
  msgs.querySelectorAll('.choices').forEach(e => e.remove());
  const div = document.createElement('div');
  div.className = 'msg ' + m.role;
  if (m.role === 'user') {
    div.innerHTML = '<div class="bubble"></div>';
    div.firstChild.textContent = m.text;
  } else {
    const meta = m.meta || {}, intent = T.intents[meta.intent] || meta.intent || '';
    const st = meta.status ? ' · ' + ((T.status||{})[meta.status] || meta.status) : '';
    div.innerHTML = '<div class="logo">G</div><div class="body">' + md(m.text) +
      '<div class="actions"><button class="icon-btn copy" title="' + T.copy + '"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg></button>' +
      (intent ? '<span class="chip">' + esc(intent + st) + '</span>' : '') +
      (meta.learned_word || meta.interpreted ? '<span class="chip">' + esc(T.learnedWord) + '</span>' : '') + '</div></div>';
    if (Array.isArray(meta.choices) && meta.choices.length) {
      const box = document.createElement('div'); box.className = 'choices';
      for (const c of meta.choices) {
        const b = document.createElement('button'); b.className = 'choice'; b.textContent = c;
        b.onclick = () => submit(c); box.appendChild(b);
      }
      div.querySelector('.body').insertBefore(box, div.querySelector('.actions'));
    }
    div.querySelector('.copy').onclick = e => { navigator.clipboard && navigator.clipboard.writeText(m.text); e.currentTarget.title = T.copied; };
  }
  msgs.appendChild(div);
  scrollDown();
  return div;
}
function typing() {
  const div = document.createElement('div');
  div.className = 'msg genesis';
  div.innerHTML = '<div class="logo">G</div><div class="body"><div class="dots"><span></span><span></span><span></span></div></div>';
  msgs.appendChild(div); scrollDown();
  return div;
}
function showEmpty() {
  msgs.innerHTML = '';
  $('empty').style.display = 'flex';
}
async function loadConvs() {
  const list = await api('/api/conversations');
  const box = $('convs'); box.innerHTML = '';
  for (const c of list) {
    const b = document.createElement('button');
    b.className = 'conv' + (c.id === current ? ' active' : '');
    b.textContent = c.title || T.untitled;
    b.onclick = () => openConv(c.id);
    box.appendChild(b);
  }
}
async function loadStatus() {
  try {
    const s = await api('/api/status');
    $('status').innerHTML = '<b>' + T.stage + ' ' + s.stage + '</b> — ' + esc(s.stage_name) + '<br>' +
      s.knowledge + ' ' + T.beliefs + ' (' + s.validated + ' ' + T.validated + ') · ' + s.episodes + ' ' + T.episodes +
      '<br><span class="badge">no LLM · stdlib · local</span>';
  } catch (e) {}
}
async function openConv(id) {
  current = id; location.hash = 'c=' + id;
  const d = await api('/api/conversations/' + id);
  msgs.innerHTML = '';
  if (!d.messages.length) showEmpty();
  d.messages.forEach(addMsg);
  loadConvs();
  if (innerWidth <= 768) $('side').classList.add('hidden');
}
function newChat() {
  current = null; history.replaceState(null, '', location.pathname);
  showEmpty(); loadConvs(); input.focus();
  if (innerWidth <= 768) $('side').classList.add('hidden');
}
async function submit(text) {
  text = (text ?? input.value).trim();
  if (!text || busy) return;
  busy = true; input.value = ''; resize(); send.disabled = true;
  addMsg({role:'user', text});
  const t = typing();
  try {
    if (current === null) { const c = await api('/api/conversations', {lang: LANG}); current = c.id; location.hash = 'c=' + current; }
    const r = await api('/api/conversations/' + current + '/messages', {text, lang: LANG});
    t.remove(); addMsg(r.reply);
    loadConvs(); loadStatus();
  } catch (e) {
    t.remove(); addMsg({role:'genesis', text: T.error + e.message, meta:{}});
  }
  busy = false; input.focus();
}
function resize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 200) + 'px'; send.disabled = !input.value.trim(); }
input.addEventListener('input', resize);
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); } });
send.onclick = () => submit();
$('new').onclick = newChat;
$('close').onclick = () => $('side').classList.add('hidden');
$('menu').onclick = () => $('side').classList.toggle('hidden');
new MutationObserver(() => { $('menu').style.display = $('side').classList.contains('hidden') ? 'flex' : 'none'; })
  .observe($('side'), {attributes:true});
msgs.addEventListener('click', e => { const a = e.target.closest('a.kid'); if (a) submit(T.explain + a.dataset.k); });
for (const [q, sub] of T.cards) {
  const b = document.createElement('button'); b.className = 'card';
  b.innerHTML = esc(q) + '<small>' + esc(sub) + '</small>';
  b.onclick = () => submit(q);
  $('cards').appendChild(b);
}
if (innerWidth <= 768) $('side').classList.add('hidden');
async function boot() {
  let me = null;
  try { me = await api('/api/whoami'); } catch (e) { if (e.status === 401) return; }
  if (me && me.user !== 'local') showMe(me);
  const m = location.hash.match(/c=(\d+)/);
  loadStatus();
  if (m) openConv(+m[1]).catch(newChat); else newChat();
  input.focus();
}
boot();
</script>
</body></html>
"""
