"""Read-only monitoring dashboard (stdlib only).

``python -m genesis dashboard --db runs/demo.db`` serves a single page and a
small JSON API. The memory database is opened with SQLite's ``mode=ro`` for
every request, so observing the agent can never change it (nor does the
dashboard instantiate an agent, which would start a new session).
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .memory import LongTermMemory
from .metacognition import CURRICULUM, Metacognition


def _j(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except ValueError:
        return s


def _episode(r: dict) -> dict:
    return {k: _j(v) if k in ("action", "observation", "outcome", "predictions") else v for k, v in r.items()}


# ------------------------------------------------------------------ queries
def summary(ltm: LongTermMemory) -> dict:
    a = ltm.assessments()
    gates = [{"stage": g.stage, "name": g.name,
              "criteria": [{"metric": m, "required": f"{op} {t}", "value": a.get(m)} for m, op, t in g.criteria]}
             for g in CURRICULUM]
    counts = {t: ltm.count(t) for t in ("episodes", "knowledge", "evidence", "procedures", "associations",
                                        "metacog", "objectives", "model_versions")}
    by_status = {r[0]: r[1] for r in ltm.db.execute("SELECT status, COUNT(*) FROM knowledge GROUP BY status")}
    return {"stage": Metacognition.current_stage(a), "gates": gates, "session": ltm.get_meta("session"),
            "interactions": ltm.get_meta("interactions"), "counts": counts, "knowledge_by_status": by_status,
            "bytes": ltm.size_bytes(), "active_config": ltm.get_meta("active_config") or {},
            "innate": ltm.get_meta("innate_manifest"), "stage0_audit": ltm.get_meta("stage0_audit"),
            "innate_language": ltm.get_meta("innate_language")}


def knowledge(ltm: LongTermMemory, status=None) -> list:
    return [vars(k) for k in ltm.knowledge(status=status)]


def episodes(ltm: LongTermMemory, limit: int = 50, env=None) -> list:
    q, args = "SELECT * FROM episodes", []
    if env:
        q += " WHERE env_id=?"
        args.append(env)
    rows = ltm.db.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
    return [_episode(dict(r)) for r in rows]


def failures(ltm: LongTermMemory, limit: int = 50) -> dict:
    """Confident predictions that turned out wrong, and refuted beliefs."""
    wrong = []
    for r in ltm.db.execute("SELECT * FROM episodes WHERE predictions LIKE '%confidence%' ORDER BY id DESC"):
        preds = _j(r["predictions"]) or {}
        bad = {k: p for k, p in preds.items()
               if p and p.get("predicted") is not None and p["predicted"] != p.get("actual")
               and (p.get("confidence") or 0) >= 0.5}
        if bad:
            e = _episode(dict(r))
            e["wrong"] = bad
            wrong.append(e)
            if len(wrong) >= limit:
                break
    refuted = []
    for k in ltm.knowledge(status="refuted"):
        hist = ltm.db.execute("SELECT reason FROM knowledge_history WHERE knowledge_id=? AND new_status='refuted' "
                              "ORDER BY ts DESC LIMIT 1", (k.id,)).fetchone()
        refuted.append({"id": k.id, "key": k.key, "statement": k.statement, "reason": hist[0] if hist else None})
    return {"confident_wrong_predictions": wrong, "refuted_beliefs": refuted}


def performance(ltm: LongTermMemory, bucket: int = 25) -> dict:
    """Prediction accuracy over the agent's lifetime (from episodic memory), per bucket of episodes."""
    hist, ok, n, start, env = [], 0, 0, None, None
    for r in ltm.db.execute("SELECT id, session, env_id, predictions FROM episodes ORDER BY id"):
        preds = _j(r["predictions"]) or {}
        scored = [p for p in preds.values() if p and p.get("predicted") is not None]
        if not scored:
            continue
        start = start if start is not None else r["id"]
        env = env or r["env_id"]
        ok += sum(p["predicted"] == p["actual"] for p in scored)
        n += len(scored)
        if n >= bucket:
            hist.append({"first_episode": start, "last_episode": r["id"], "session": r["session"], "env": env,
                         "accuracy": round(ok / n, 4), "n": n})
            ok = n = 0
            start = env = None
    assess = [{"metric": m["metric"], "value": m["value"], "session": m["session"], "ts": m["ts"]}
              for m in ltm.metrics() if m["context"] and '"assessment"' in m["context"]]
    calib = [{"session": m["session"], "ece": m["value"], "ts": m["ts"]} for m in ltm.metrics("calibration")]
    activity = [dict(r) for r in ltm.db.execute(
        "SELECT session, env_id, provenance, COUNT(*) AS n FROM episodes GROUP BY session, env_id, provenance "
        "ORDER BY session, env_id")]
    return {"accuracy_history": hist, "assessments": assess, "calibration": calib, "activity": activity}


ROUTES = {
    "summary": lambda ltm, q: summary(ltm),
    "knowledge": lambda ltm, q: knowledge(ltm, q.get("status")),
    "episodes": lambda ltm, q: episodes(ltm, int(q.get("limit", 50)), q.get("env")),
    "failures": lambda ltm, q: failures(ltm),
    "performance": lambda ltm, q: performance(ltm),
    "objectives": lambda ltm, q: ltm.objectives(q.get("status")),
    "versions": lambda ltm, q: ltm.versions(),
    "procedures": lambda ltm, q: [dict(r) for r in ltm.db.execute("SELECT * FROM procedures ORDER BY successes DESC")],
    "associations": lambda ltm, q: ltm.associations(),
}


def api(db_path: str, path: str, query: dict):
    ltm = LongTermMemory(db_path, read_only=True)
    try:
        m = re.fullmatch(r"/api/explain/(\d+)", path)
        if m:
            ex = ltm.explain(int(m.group(1)))
            for rel in ex.get("evidence", {}).values():
                rel["examples"] = [_episode(e) for e in rel["examples"]]
            return ex or None
        name = path[len("/api/"):]
        return ROUTES[name](ltm, query) if name in ROUTES else None
    finally:
        ltm.close()


def make_handler(db_path: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path.startswith("/api/"):
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                try:
                    data = api(db_path, u.path, q)
                except Exception as e:  # report, never crash the server
                    return self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
                if data is None:
                    return self._send(404, b'{"error": "not found"}', "application/json")
                return self._send(200, json.dumps(data, default=str).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            self._send(405, b"read-only dashboard", "text/plain")

    return Handler


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(db_path))
    print(f"GENESIS dashboard (read-only) for {db_path} on http://{host}:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>GENESIS monitor</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;margin:0;background:#0f1419;color:#d8dee9}
header{padding:12px 20px;background:#1a2129;border-bottom:1px solid #2c3540;display:flex;gap:24px;align-items:baseline}
h1{font-size:18px;margin:0;color:#88c0d0} h2{font-size:15px;color:#88c0d0;margin:0 0 8px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px;padding:14px}
section{background:#1a2129;border:1px solid #2c3540;border-radius:6px;padding:12px;overflow:auto;max-height:440px}
table{border-collapse:collapse;width:100%;font-size:12px} td,th{padding:3px 6px;border-bottom:1px solid #2c3540;text-align:left;vertical-align:top}
th{color:#81a1c1;position:sticky;top:0;background:#1a2129}
.ok{color:#a3be8c}.bad{color:#bf616a}.warn{color:#ebcb8b}.dim{color:#6b7785}
.tag{display:inline-block;padding:0 6px;border-radius:8px;font-size:11px;background:#2c3540}
a{color:#88c0d0;cursor:pointer} code{font-size:12px;color:#ebcb8b}
#explain{position:fixed;inset:5% 10%;background:#11161c;border:1px solid #88c0d0;border-radius:6px;padding:16px;overflow:auto;display:none;z-index:9}
select{background:#0f1419;color:#d8dee9;border:1px solid #2c3540}
</style></head><body>
<header><h1>GENESIS</h1><span id="stage"></span><span id="meta" class="dim"></span>
<span style="margin-left:auto" class="dim">read-only · auto-refresh 10 s</span></header>
<main>
<section><h2>Developmental stages (measured gates)</h2><div id="gates"></div></section>
<section><h2>Learning progress (prediction accuracy over lifetime)</h2><div id="chart"></div><div id="activity"></div></section>
<section><h2>Acquired knowledge <select id="kstatus"><option value="">all</option><option>validated</option><option>candidate</option><option>refuted</option><option>unpredictable</option></select></h2><div id="knowledge"></div></section>
<section><h2>Memory statistics</h2><div id="memory"></div></section>
<section><h2>Recent experiences: predictions vs outcomes</h2><div id="episodes"></div></section>
<section><h2>Failed predictions &amp; refuted beliefs</h2><div id="failures"></div></section>
<section><h2>Learning objectives (self-generated)</h2><div id="objectives"></div></section>
<section><h2>Model versions (self-improvement)</h2><div id="versions"></div></section>
<section><h2>Assessment history</h2><div id="assess"></div></section>
<section><h2>Procedural memory &amp; associations</h2><div id="procs"></div></section>
</main>
<div id="explain"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const get=p=>fetch("/api/"+p).then(r=>r.json());
const num=v=>v==null?"<span class=dim>n/a</span>":(typeof v=="number"?(+v.toFixed(4)):esc(v));
const table=(cols,rows)=>"<table><tr>"+cols.map(c=>"<th>"+c[0]+"</th>").join("")+"</tr>"+
  rows.map(r=>"<tr>"+cols.map(c=>"<td>"+c[1](r)+"</td>").join("")+"</tr>").join("")+"</table>";
const cls={validated:"ok",achieved:"ok",deployed:"ok",refuted:"bad",abandoned:"warn",unpredictable:"warn","rejected-trial":"dim","rejected-holdout":"bad","pending-approval":"warn"};
const tag=s=>`<span class="tag ${cls[s]||""}">${esc(s)}</span>`;
function chart(h){ if(!h.length) return "<p class=dim>no predictions yet</p>";
  const W=440,H=150,n=h.length,x=i=>20+i*(W-30)/Math.max(n-1,1),y=v=>H-15-v*(H-30);
  let envs=[...new Set(h.map(p=>p.env))],col=["#88c0d0","#a3be8c","#ebcb8b","#b48ead","#d08770","#5e81ac"];
  let pts=h.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p.accuracy)}" r="2.5" fill="${col[envs.indexOf(p.env)%6]}"><title>${esc(p.env)} episodes ${p.first_episode}-${p.last_episode}: ${(p.accuracy*100).toFixed(0)}%</title></circle>`).join("");
  let line=h.map((p,i)=>`${x(i)},${y(p.accuracy)}`).join(" ");
  return `<svg width="${W}" height="${H}"><line x1=20 x2=${W-10} y1=${y(0)} y2=${y(0)} stroke="#2c3540"/><line x1=20 x2=${W-10} y1=${y(1)} y2=${y(1)} stroke="#2c3540"/>
  <text x=0 y=${y(1)+4} fill="#6b7785" font-size=10>1</text><text x=0 y=${y(0)+4} fill="#6b7785" font-size=10>0</text>
  <polyline points="${line}" fill="none" stroke="#4c566a"/>${pts}</svg><div>`+envs.map((e,i)=>`<span style="color:${col[i%6]}">● ${esc(e)}</span> `).join("")+"</div>";}
async function explain(id){ const e=await get("explain/"+id), it=e.item; let h=`<a onclick="$('explain').style.display='none'">✕ close</a>
  <h2>#${it.id} ${esc(it.key)}: <code>${esc(it.statement)}</code> ${tag(it.status)}</h2>
  <p>confidence ${num(it.confidence)} · provenance <b>${esc(it.provenance)}</b> · support ${it.n_support} · contradictions ${it.n_contra} · forward predictions ${it.forward_ok} ok / ${it.forward_bad} wrong</p>
  <h2>Status history</h2>`+table([["time",r=>new Date(r.ts*1000).toLocaleTimeString()],["from",r=>esc(r.old_status)],["to",r=>tag(r.new_status)],["reason",r=>esc(r.reason)]],e.history);
  for(const [rel,v] of Object.entries(e.evidence)) if(v.total) h+=`<h2>${rel} (${v.total} episodes, first ${v.examples.length} shown)</h2>`+
    table([["episode",r=>r.id],["session",r=>r.session],["provenance",r=>esc(r.provenance)],["action",r=>"<code>"+esc(JSON.stringify(r.action))+"</code>"],["outcome",r=>"<code>"+esc(JSON.stringify(r.outcome))+"</code>"]],v.examples);
  $("explain").innerHTML=h; $("explain").style.display="block";}
async function refresh(){
  const s=await get("summary"), st=s.stage;
  $("stage").innerHTML=`stage <b>${st.stage}</b> — ${esc(st.name)}`;
  $("meta").textContent=`session ${s.session} · ${s.interactions} interactions · ${(s.bytes/1024).toFixed(0)} KiB`;
  $("gates").innerHTML=table([["stage",g=>(st.passed.includes(g.stage)?"<span class=ok>✔</span> ":g.stage==st.stage?"<span class=warn>▶</span> ":"<span class=dim>·</span> ")+g.stage],["name",g=>esc(g.name)],
    ["criteria (value / required)",g=>g.criteria.map(c=>`${esc(c.metric)} = ${num(c.value)} <span class=dim>${esc(c.required)}</span>`).join("<br>")]],s.gates);
  $("memory").innerHTML=table([["store",r=>r[0]],["items",r=>r[1]]],Object.entries(s.counts))+
    "<p>knowledge by status: "+Object.entries(s.knowledge_by_status).map(([k,v])=>tag(k)+" "+v).join(" ")+"</p>"+
    `<p>active config (approved self-improvements): <code>${esc(JSON.stringify(s.active_config))}</code></p>`+
    `<p class=dim>innate: ${esc(Object.keys(s.innate?.arithmetic_primitives||{}).join(", "))}; domain knowledge: ${(s.innate?.domain_knowledge||[]).length?esc(s.innate.domain_knowledge.join(", ")):"none (empty at birth)"}; innate language faculty: ${s.innate_language?esc(s.innate_language.languages.join("/")+" — "+s.innate_language.n_forms+" word forms about itself, no world knowledge, no pretrained model (v"+s.innate_language.version+", installed session "+s.innate_language.installed_session+")"):"none"}</p>`;
  const k=await get("knowledge"+($("kstatus").value?"?status="+$("kstatus").value:""));
  $("knowledge").innerHTML=table([["id",r=>`<a onclick="explain(${r.id})">${r.id}</a>`],["key",r=>esc(r.key)],["belief",r=>"<code>"+esc(r.statement)+"</code>"],["status",r=>tag(r.status)],["conf",r=>num(r.confidence)],["source",r=>esc(r.provenance)]],k.slice().reverse());
  const p=await get("performance");
  $("chart").innerHTML=chart(p.accuracy_history);
  $("activity").innerHTML=table([["session",r=>r.session],["environment",r=>esc(r.env_id)],["mode",r=>esc(r.provenance)],["episodes",r=>r.n]],p.activity);
  $("assess").innerHTML=table([["session",r=>r.session],["metric",r=>esc(r.metric)],["value",r=>num(r.value)]],p.assessments.slice().reverse());
  const ep=await get("episodes?limit=40");
  $("episodes").innerHTML=table([["#",r=>r.id],["context",r=>esc(r.context)],["predicted → actual",r=>Object.entries(r.predictions||{}).map(([q,v])=>`<span class=${v.predicted===v.actual?"ok":"bad"}>${esc(v.predicted)} → ${esc(v.actual)}</span> <span class=dim>(${num(v.confidence)})</span>`).slice(0,4).join("<br>")||"<span class=dim>no prediction (warm-up)</span>"],["surprise",r=>num(r.surprise)]],ep);
  const f=await get("failures");
  $("failures").innerHTML="<h2>refuted beliefs</h2>"+table([["id",r=>`<a onclick="explain(${r.id})">${r.id}</a>`],["key",r=>esc(r.key)],["belief",r=>"<code>"+esc(r.statement)+"</code>"],["why",r=>esc(r.reason)]],f.refuted_beliefs)+
    "<h2>confident predictions that failed</h2>"+table([["#",r=>r.id],["context",r=>esc(r.context)],["wrong",r=>Object.entries(r.wrong).map(([q,v])=>`${esc(q.split("/").slice(1).join("/"))}: ${esc(v.predicted)} ≠ ${esc(v.actual)} (${num(v.confidence)})`).join("<br>")]],f.confident_wrong_predictions);
  const o=await get("objectives");
  $("objectives").innerHTML=table([["status",r=>tag(r.status)],["objective",r=>esc(r.text)],["outcome",r=>"<code>"+esc(r.outcome||"")+"</code>"]],o.slice().reverse());
  const v=await get("versions");
  $("versions").innerHTML=v.length?table([["id",r=>r.id],["change",r=>esc(r.rationale)],["status",r=>tag(r.status)],["trial",r=>num((r.trial_metrics?.variant||r.trial_metrics)?.score)],
    ["holdout variant / base",r=>r.holdout_metrics?num(r.holdout_metrics.variant.score)+" / "+num(r.holdout_metrics.baseline.score):""],["approved by",r=>esc(r.approved_by||"")]],v.slice().reverse()):"<p class=dim>no self-improvement cycle yet</p>";
  const pr=await get("procedures"), as=await get("associations");
  $("procs").innerHTML=table([["rule shape",r=>"<code>"+esc(r.shape)+"</code>"],["successes",r=>r.successes],["attempts",r=>r.attempts]],pr)+
    table([["a",r=>esc(r.a)],["b",r=>esc(r.b)],["kind",r=>tag(r.kind)],["weight",r=>num(r.weight)]],as);
}
$("kstatus").onchange=refresh; refresh(); setInterval(refresh,10000);
</script></body></html>
"""
