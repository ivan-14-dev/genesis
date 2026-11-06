"""ChatGPT-style web entry point to GENESIS (stdlib only).

``python -m genesis chat --db runs/demo.db`` serves a single page and a small JSON API:

    GET  /api/status                         stage and memory counts
    GET  /api/conversations                  conversation list (most recent first)
    POST /api/conversations                  {"lang": "fr"|"en"} -> new conversation
    GET  /api/conversations/<id>             messages of a conversation
    POST /api/conversations/<id>/messages    {"text": "...", "lang": "fr"|"en"} -> agent reply

Unlike the read-only dashboard, this interface *does* let the agent act and learn, so it holds one
``Genesis`` instance (one session) for the life of the server. Requests are served one at a time on the
thread that owns the SQLite connection. It binds to localhost by default; POSTs must be JSON and, when the
browser sends an Origin header, come from the same origin.
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional
from urllib.parse import urlparse

from .agent import Genesis
from .chat import ChatEngine
from .metacognition import Metacognition

MAX_BODY = 16 * 1024


def status(engine: ChatEngine) -> dict:
    a = engine.agent
    st = Metacognition.current_stage(a.assessments())
    return {"stage": st["stage"], "stage_name": st["name"], "interactions": a.interactions,
            "knowledge": a.ltm.count("knowledge"),
            "validated": len(a.ltm.knowledge(status="validated")), "episodes": a.ltm.count("episodes"),
            "session": a.session}


def make_handler(engine: ChatEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, data, ctype: str = "application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._send(200, status(engine))
            if path == "/api/conversations":
                return self._send(200, engine.conversations())
            m = re.fullmatch(r"/api/conversations/(\d+)", path)
            if m:
                conv = engine.ltm.conversation(int(m.group(1)))
                if conv is None:
                    return self._send(404, {"error": "not found"})
                return self._send(200, {"conversation": conv, "messages": engine.messages(conv["id"])})
            self._send(404, {"error": "not found"})

        def _body(self) -> Optional[dict]:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._send(415, {"error": "JSON required"})
                return None
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self._send(403, {"error": "cross-origin request refused"})
                return None
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._send(413, {"error": "message too large"})
                return None
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self._send(400, {"error": "invalid JSON"})
                return None
            if not isinstance(data, dict):
                self._send(400, {"error": "JSON object required"})
                return None
            return data

        def do_POST(self):
            path = urlparse(self.path).path
            data = self._body()
            if data is None:
                return
            try:
                if path == "/api/conversations":
                    return self._send(200, engine.new_conversation(data.get("lang")))
                m = re.fullmatch(r"/api/conversations/(\d+)/messages", path)
                if m:
                    text = data.get("text")
                    if not isinstance(text, str) or not text.strip():
                        return self._send(400, {"error": "text required"})
                    return self._send(200, engine.send(int(m.group(1)), text, data.get("lang")))
            except KeyError:
                return self._send(404, {"error": "conversation not found"})
            except Exception as e:  # report, never crash the server
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, {"error": "not found"})

    return Handler


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8766,
          world_factory: Optional[Callable] = None, object_factory: Optional[Callable] = None) -> None:
    agent = Genesis(db_path)
    engine = ChatEngine(agent, world_factory, object_factory)
    srv = HTTPServer((host, port), make_handler(engine))
    print(f"GENESIS chat for {db_path} on http://{host}:{port}  (session {agent.session}; Ctrl+C to stop)",
          flush=True)
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
@media (max-width:768px){#side{position:fixed;z-index:5;height:100%}#side.hidden{margin-left:-260px}.cards{grid-template-columns:1fr}.bubble{max-width:90%}}
</style></head><body>
<nav id="side">
  <div class="side-top"><div class="logo">G</div>
    <button class="icon-btn" id="close" title="Close sidebar"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/></svg></button></div>
  <button id="new"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg><span data-i="new"></span></button>
  <div class="hist-label" data-i="history"></div>
  <div id="convs"></div>
  <div id="status"></div>
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
<script>
const FR = (navigator.language||'').toLowerCase().startsWith('fr');
const LANG = FR ? 'fr' : 'en';
document.documentElement.lang = LANG;
const T = FR ? {
  new:'Nouvelle discussion', history:'Historique', model:'· sans modèle de langage', hello:'Que dois-je apprendre ?',
  sub:"Je ne sais que ce que j'ai appris par expérience. Donne-moi une suite de nombres : je prédirai chaque terme avant de le lire, puis je te dirai quelle règle j'ai trouvée, et avec quelles preuves.",
  placeholder:'Enseigne un fait, pose une question ou envoie une suite…',
  note:"GENESIS n'utilise aucun modèle de langage : il comprend quelques commandes, apprend les mots que tu lui enseignes et n'affirme que ce que sa mémoire justifie.",
  stage:'Stade', beliefs:'croyances', validated:'validées', episodes:'épisodes', copy:'Copier', copied:'Copié',
  untitled:'Nouvelle discussion', error:'Erreur : ', explain:'explique ',
  cards:[['3, 7, 11, 15, 19, 23, 27','une suite arithmétique'],['atelier objets','apprendre des mots à partir des objets'],
         ['Nala est un chat; Les chats sont des animaux','enseigner deux faits, sans les précharger'],['Nala est-elle un animal ?','déduire à partir des enseignements']],
  status:{validated:'validée',candidate:'candidate',unpredictable:'imprévisible',unknown:'inconnue'},
  learnedWord:'mots appris',
  intents:{clarify:'vérification',vocabulary:'vocabulaire','teach-request':'apprentissage de mots','yes-no':'oui/non',thanks:'merci',continue:'apprentissage','too-short':'trop court',sequence:'apprentissage',explain:'preuves',knowledge:'mémoire',explore:'exploration',stage:'stade',limitations:'limites',questions:'curiosité',predict:'prédiction','not-understood':'non compris',help:'aide',intro:'présentation',refused:'refusé',define:'définition',languages:'langues',smalltalk:'conversation',bye:'au revoir',declined:'refus','unknown-topic':'hors de mes connaissances'}
} : {
  new:'New chat', history:'History', model:'· no language model', hello:'What should I learn?',
  sub:"I only know what I learned from experience. Give me a number sequence: I will predict each term before reading it, then tell you which rule I found and on what evidence.",
  placeholder:'Teach a fact, ask a question or send a sequence…',
  note:'GENESIS uses no language model: it understands a few commands, learns the words you teach it, and only claims what its memory justifies.',
  stage:'Stage', beliefs:'beliefs', validated:'validated', episodes:'episodes', copy:'Copy', copied:'Copied',
  untitled:'New chat', error:'Error: ', explain:'explain ',
  cards:[['3, 7, 11, 15, 19, 23, 27','an arithmetic sequence'],['object workshop','learn words from objects'],
         ['Nala is a cat; All cats are animals','teach two facts, not preloaded'],['Is Nala an animal?','deduce from teachings']],
  learnedWord:'learned words',
  intents:{clarify:'check',vocabulary:'vocabulary','teach-request':'word learning','yes-no':'yes/no',thanks:'thanks',continue:'learning','too-short':'too short',sequence:'learning',explain:'evidence',knowledge:'memory',explore:'exploration',stage:'stage',limitations:'limitations',questions:'curiosity',predict:'prediction','not-understood':'not understood',help:'help',intro:'intro',refused:'refused',define:'definition',languages:'languages',smalltalk:'small talk',bye:'goodbye',declined:'declined','unknown-topic':'outside my knowledge'}
};
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
  if (!r.ok) throw new Error(d.error || r.status);
  return d;
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
const m = location.hash.match(/c=(\d+)/);
loadStatus();
if (m) openConv(+m[1]).catch(newChat); else newChat();
input.focus();
</script>
</body></html>
"""
