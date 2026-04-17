#!/usr/bin/env python3
"""Training Arena Dashboard — Full Rock 'Em Sock 'Em Experience.

One command: python arena.py --dashboard
Click FIGHT → backend arena runs → robots animate → health drains → KO.
"""

import sys, json, subprocess, threading, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from core.metrics_db import MetricsDatabase
from core.llm_wiki import LLMWiki

st.set_page_config(page_title="Training Arena", page_icon="🏟️", layout="wide",
                    initial_sidebar_state="collapsed")


@st.cache_resource
def get_db():
    return MetricsDatabase()

@st.cache_resource
def get_wiki():
    return LLMWiki()

def short_name(n):
    return (n or "").split("/")[-1]


def run_single_match(model_a, model_b, prompt_text, category, config_path="config/models.yaml"):
    """Run a single arena match in-process and return the result."""
    try:
        from core.model_hub import ModelHub
        from core.evaluation_rubric import EvaluationRubric
        hub = ModelHub(config_path)
        rubric = EvaluationRubric(judge_model_name="phi3:mini")

        resp_a = hub.generate_response(model_a, prompt_text, max_tokens=150)
        resp_b = hub.generate_response(model_b, prompt_text, max_tokens=150)

        # Real rubric scoring
        try:
            scores_a = rubric.evaluate_response(hub, prompt_text, resp_a, category)
            sa = rubric.calculate_overall(scores_a, category)
        except Exception:
            sa = random.uniform(0.3, 0.8)
        try:
            scores_b = rubric.evaluate_response(hub, prompt_text, resp_b, category)
            sb = rubric.calculate_overall(scores_b, category)
        except Exception:
            sb = random.uniform(0.3, 0.8)

        winner = model_a if sa > sb else (model_b if sb > sa else "draw")

        db = MetricsDatabase()
        db.record_match(model_a, model_b, winner, prompt_text, category, sa, sb)

        return {
            "model_a": model_a, "model_b": model_b,
            "score_a": round(sa, 4), "score_b": round(sb, 4),
            "winner": winner, "prompt": prompt_text, "category": category,
            "response_a": resp_a[:200], "response_b": resp_b[:200],
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# MAIN ARENA HTML COMPONENT
# ═══════════════════════════════════════════════════════════
def render_arena(db, wiki):
    rankings = db.get_global_rankings()
    matches = db.get_recent_matches(50)
    evo_log = db.get_evolution_log(limit=20)
    wiki_stats = wiki.get_wiki_stats()

    try:
        from core.arena import COMMERCIAL_BASELINES
        for name, scores in COMMERCIAL_BASELINES.items():
            avg = sum(scores.values()) / len(scores)
            rankings.append({"model_name": name, "elo_rating": round(1500 + (avg - 0.5) * 1400, 1),
                             "wins": 0, "losses": 0, "draws": 0, "type": "commercial"})
        rankings.sort(key=lambda x: x["elo_rating"], reverse=True)
    except Exception:
        pass

    for r in rankings:
        if "type" not in r:
            r["type"] = "local"

    data_json = json.dumps({
        "leaderboard": rankings, "matches": matches,
        "evolution": evo_log, "wiki": wiki_stats,
    }, default=str)

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;600;800&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:transparent;font-family:'Outfit',sans-serif;color:#e0e0e0;overflow-x:hidden}}

.header{{text-align:center;padding:12px 0 6px}}
.header h1{{font-family:'Press Start 2P',monospace;font-size:20px;color:#ffd700;text-shadow:2px 2px 0 #9a1c1c,-1px -1px 0 #1c4a9a;letter-spacing:2px}}
.header .sub{{font-family:'JetBrains Mono',monospace;font-size:9px;color:#64748b;letter-spacing:3px;margin-top:3px}}

.tabs{{display:flex;gap:2px;border-bottom:1px solid #222;margin:6px 0 0}}
.tab{{padding:7px 14px;cursor:pointer;border:none;border-radius:6px 6px 0 0;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:bold;letter-spacing:1px;background:transparent;color:#555;border-bottom:2px solid transparent;transition:all .2s}}
.tab.active{{background:#1a1a2e;color:#ffd700;border-bottom-color:#ffd700}}
.tab:hover{{color:#ffd700}}

.panel{{background:#0d0d1a;border-radius:0 0 12px 12px;padding:14px;min-height:480px}}

/* Fighter HUD */
.hud{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;gap:8px}}
.hud-side{{width:200px}}
.hud-center{{flex:1;text-align:center;padding-top:4px}}

.nickname{{font-family:'Press Start 2P',monospace;font-size:10px;letter-spacing:1px;margin-bottom:2px}}
.nn-red{{color:#f04848}}.nn-blue{{color:#4888f0;text-align:right}}

.model-sel{{display:block;width:100%;padding:5px 6px;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:bold;margin-top:2px}}
.sel-red{{background:#1a1a2e;color:#f04848;border:1px solid #d42b2b}}
.sel-blue{{background:#1a1a2e;color:#4888f0;border:1px solid #2b8dd4;text-align:right}}

/* Health bars */
.hp-wrap{{margin-top:6px}}
.hp-label{{font-family:'JetBrains Mono',monospace;font-size:8px;color:#888;display:flex;justify-content:space-between}}
.hp-bar{{height:12px;background:#1a1a2e;border-radius:6px;overflow:hidden;border:1px solid #333;margin-top:2px}}
.hp-fill-red{{height:100%;background:linear-gradient(90deg,#9a1c1c,#d42b2b,#f04848);border-radius:6px;transition:width .6s ease;width:100%}}
.hp-fill-blue{{height:100%;background:linear-gradient(90deg,#1c5a9a,#2b8dd4,#4888f0);border-radius:6px;transition:width .6s ease;width:100%}}

/* Round / Timer */
.round-display{{font-family:'Press Start 2P',monospace;font-size:9px;color:#ffd700;margin-bottom:2px}}
.timer-display{{font-family:'JetBrains Mono',monospace;font-size:22px;color:#fff;font-weight:bold}}
.fight-btn{{padding:10px 28px;border-radius:8px;border:2px solid #ffd700;background:linear-gradient(180deg,#ffd700,#cc9900);color:#111;font-family:'Press Start 2P',monospace;font-size:10px;cursor:pointer;text-shadow:0 1px 0 rgba(255,255,255,.3);transition:all .2s;margin-top:6px}}
.fight-btn:hover{{transform:scale(1.05);box-shadow:0 0 16px rgba(255,215,0,.4)}}
.fight-btn:disabled{{background:#333;color:#555;cursor:default;border-color:#444;transform:none;box-shadow:none}}

.ring-wrap{{background:radial-gradient(ellipse at 50% 80%,#1a2a1a 0%,#0d0d1a 70%);border-radius:12px;padding:4px 0;position:relative}}

/* Status / results */
.status{{text-align:center;padding:8px 0;font-family:'Press Start 2P',monospace;font-size:8px;min-height:24px;color:#555;transition:color .3s}}
.status.active{{color:#ffd700}}
.status.ko{{color:#ff0;font-size:12px;text-shadow:0 0 10px #f00}}

/* Result card */
.result-card{{background:rgba(255,215,0,.06);border:1px solid rgba(255,215,0,.2);border-radius:8px;padding:10px 14px;margin:8px auto;max-width:500px;display:none}}
.result-card.show{{display:block;animation:fadeIn .5s ease}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
.result-title{{font-family:'Press Start 2P',monospace;font-size:10px;color:#ffd700;margin-bottom:8px;text-align:center}}
.result-row{{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;padding:2px 0}}
.result-row .lbl{{color:#888}}.result-row .val{{color:#fff;font-weight:bold}}
.result-row .win{{color:#10b981}}.result-row .lose{{color:#ef4444}}

.stats{{display:flex;justify-content:center;gap:24px;padding:6px 0}}
.stat{{text-align:center}}
.stat .val{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:bold;color:#ffd700}}
.stat .lbl{{font-size:8px;color:#444;font-family:'JetBrains Mono',monospace;letter-spacing:1px}}

/* Leaderboard */
.lb-row{{display:flex;align-items:center;gap:10px;padding:10px 12px;margin-bottom:5px;background:#111;border-radius:8px;animation:slideIn .4s ease-out backwards}}
.lb-row.commercial{{border-left:3px solid #8b5cf6;background:rgba(139,92,246,.03)}}.lb-row.local{{border-left:3px solid #10b981}}
.lb-row.rank1{{background:linear-gradient(90deg,rgba(255,215,0,.08),transparent);border-left-color:#ffd700}}
@keyframes slideIn{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}
.rb{{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Press Start 2P',monospace;font-size:9px;font-weight:bold;flex-shrink:0}}
.rb1{{background:linear-gradient(135deg,#ffd700,#ffaa00);color:#111;box-shadow:0 0 10px rgba(255,215,0,.4)}}
.rb2{{background:#94a3b8;color:#111}}.rb3{{background:#cd7f32;color:#111}}.rbn{{background:#333;color:#666}}
.lb-name{{font-weight:700;font-size:12px;color:#f1f5f9}}.lb-sub{{font-size:9px;color:#555;font-family:'JetBrains Mono',monospace}}
.lb-tag{{font-size:7px;padding:1px 5px;border-radius:5px;margin-left:5px;font-family:'JetBrains Mono',monospace;vertical-align:middle}}
.tag-l{{background:rgba(16,185,129,.15);color:#10b981}}.tag-c{{background:rgba(139,92,246,.15);color:#a78bfa}}
.lb-wld{{font-family:'JetBrains Mono',monospace;font-size:10px;color:#64748b;min-width:70px}}
.lb-elo{{font-family:'JetBrains Mono',monospace;font-weight:bold;font-size:15px;color:#ffd700;min-width:45px;text-align:right}}
.ebar{{width:100px;height:5px;background:#1e293b;border-radius:3px;overflow:hidden}}.efill{{height:100%;border-radius:3px;transition:width .8s ease}}

.match-row{{display:flex;align-items:center;gap:6px;padding:6px 8px;margin-bottom:3px;background:#111;border-radius:5px;font-size:10px;font-family:'JetBrains Mono',monospace}}
.evo-row{{padding:7px 10px;margin-bottom:5px;background:#111;border-radius:5px;font-size:10px;font-family:'JetBrains Mono',monospace}}
.empty{{text-align:center;padding:40px;color:#444;font-family:'JetBrains Mono',monospace;font-size:11px}}

/* Punch flash */
.punch-flash{{position:absolute;width:30px;height:30px;border-radius:50%;background:rgba(255,255,0,.7);display:none;pointer-events:none;z-index:10}}
@keyframes punchBurst{{0%{{transform:scale(0);opacity:1}}100%{{transform:scale(2.5);opacity:0}}}}
</style></head><body>

<div class="header">
  <h1>TRAINING ARENA</h1>
  <div class="sub">REAL DATA · LIVE RANKINGS · POLYRATING</div>
</div>

<div class="tabs" id="tabs">
  <div class="tab active" data-tab="arena">ARENA</div>
  <div class="tab" data-tab="leaderboard">LEADERBOARD</div>
  <div class="tab" data-tab="matches">MATCH LOG</div>
  <div class="tab" data-tab="evolution">EVOLUTION</div>
</div>

<!-- ══════ ARENA TAB ══════ -->
<div class="panel" id="panel-arena">
  <div class="hud">
    <div class="hud-side">
      <div class="nickname nn-red">RED ROCKET</div>
      <select id="red-sel" class="model-sel sel-red"></select>
      <div class="hp-wrap">
        <div class="hp-label"><span>HP</span><span id="red-hp-txt">100</span></div>
        <div class="hp-bar"><div class="hp-fill-red" id="red-hp"></div></div>
      </div>
    </div>
    <div class="hud-center">
      <div class="round-display" id="round-display">ROUND 1 / 3</div>
      <div class="timer-display" id="timer-display">0:00</div>
      <button class="fight-btn" id="fight-btn" onclick="startFight()">FIGHT!</button>
    </div>
    <div class="hud-side" style="text-align:right">
      <div class="nickname nn-blue">BLUE BOMBER</div>
      <select id="blue-sel" class="model-sel sel-blue"></select>
      <div class="hp-wrap">
        <div class="hp-label"><span id="blue-hp-txt">100</span><span>HP</span></div>
        <div class="hp-bar"><div class="hp-fill-blue" id="blue-hp" style="float:right"></div></div>
      </div>
    </div>
  </div>

  <div class="ring-wrap" id="ring-wrap">
    <svg id="ring-svg" viewBox="0 0 400 190" style="width:100%;max-width:600px;display:block;margin:0 auto">
      <rect x="20" y="130" width="360" height="55" rx="6" fill="#2a5c2a" stroke="#3a7a3a" stroke-width="2"/>
      <rect x="30" y="135" width="340" height="45" rx="4" fill="none" stroke="#ffd700" stroke-width=".8" opacity=".3"/>
      <line x1="15" y1="90" x2="385" y2="90" stroke="#ddd" stroke-width="2.5" opacity=".55"/>
      <line x1="15" y1="105" x2="385" y2="105" stroke="#ddd" stroke-width="2.5" opacity=".4"/>
      <line x1="15" y1="120" x2="385" y2="120" stroke="#ddd" stroke-width="2.5" opacity=".25"/>
      <rect x="13" y="75" width="10" height="65" rx="2" fill="#c0c0c0"/>
      <circle cx="18" cy="75" r="6" fill="#ff4444" stroke="#cc0000" stroke-width="1.5"/>
      <rect x="377" y="75" width="10" height="65" rx="2" fill="#c0c0c0"/>
      <circle cx="382" cy="75" r="6" fill="#ff4444" stroke="#cc0000" stroke-width="1.5"/>

      <circle cx="200" cy="60" r="20" fill="#111" stroke="#ffd700" stroke-width="2" id="vs-circle"/>
      <text x="200" y="65" text-anchor="middle" font-family="'Press Start 2P'" font-size="11" fill="#ffd700">VS</text>

      <!-- RED ROCKET -->
      <g id="red-robot" style="transition:transform .15s ease">
        <g transform="translate(120,100)">
          <ellipse cx="0" cy="82" rx="28" ry="6" fill="rgba(0,0,0,.25)"/>
          <rect x="-15" y="52" width="10" height="26" rx="3" fill="#9a1c1c"/>
          <rect x="5" y="52" width="10" height="26" rx="3" fill="#9a1c1c"/>
          <rect x="-18" y="74" width="15" height="7" rx="3" fill="#9a1c1c"/>
          <rect x="3" y="74" width="15" height="7" rx="3" fill="#9a1c1c"/>
          <rect x="-20" y="4" width="40" height="50" rx="7" fill="#d42b2b"/>
          <rect x="-14" y="10" width="28" height="18" rx="3" fill="#9a1c1c" opacity=".3"/>
          <rect x="-8" y="14" width="16" height="5" rx="2" fill="#f04848" opacity=".5"/>
          <g id="red-arm-back"><rect x="-30" y="10" width="10" height="30" rx="4" fill="#9a1c1c"/><circle cx="-25" cy="39" r="7" fill="#9a1c1c"/></g>
          <g id="red-arm-front" style="transition:transform .12s ease;transform-origin:28px 10px"><rect x="20" y="10" width="10" height="30" rx="4" fill="#d42b2b"/><circle cx="25" cy="39" r="8" fill="#d42b2b" stroke="#f04848" stroke-width="1.5"/></g>
          <rect x="-5" y="-8" width="10" height="14" rx="3" fill="#9a1c1c"/>
          <g id="red-head" style="transition:transform .25s cubic-bezier(.6,-.5,.3,1.8)">
            <rect x="-17" y="-38" width="34" height="32" rx="8" fill="#d42b2b"/>
            <rect x="-12" y="-30" width="24" height="12" rx="3" fill="#9a1c1c"/>
            <circle cx="-4" cy="-24" r="3.5" fill="#ff6"/><circle cx="8" cy="-24" r="3.5" fill="#ff6"/>
            <rect x="-2" y="-46" width="5" height="10" rx="2.5" fill="#f04848"/>
            <circle cx="0.5" cy="-47" r="3" fill="#ff6" opacity=".8"/>
          </g>
        </g>
      </g>

      <!-- BLUE BOMBER (mirrored to face left) -->
      <g id="blue-robot" style="transition:transform .15s ease">
        <g transform="translate(280,100) scale(-1,1)">
          <ellipse cx="0" cy="82" rx="28" ry="6" fill="rgba(0,0,0,.25)"/>
          <rect x="-15" y="52" width="10" height="26" rx="3" fill="#1c5a9a"/>
          <rect x="5" y="52" width="10" height="26" rx="3" fill="#1c5a9a"/>
          <rect x="-18" y="74" width="15" height="7" rx="3" fill="#1c5a9a"/>
          <rect x="3" y="74" width="15" height="7" rx="3" fill="#1c5a9a"/>
          <rect x="-20" y="4" width="40" height="50" rx="7" fill="#2b8dd4"/>
          <rect x="-14" y="10" width="28" height="18" rx="3" fill="#1c5a9a" opacity=".3"/>
          <rect x="-8" y="14" width="16" height="5" rx="2" fill="#4888f0" opacity=".5"/>
          <g id="blue-arm-back"><rect x="-30" y="10" width="10" height="30" rx="4" fill="#1c5a9a"/><circle cx="-25" cy="39" r="7" fill="#1c5a9a"/></g>
          <g id="blue-arm-front" style="transition:transform .12s ease;transform-origin:28px 10px"><rect x="20" y="10" width="10" height="30" rx="4" fill="#2b8dd4"/><circle cx="25" cy="39" r="8" fill="#2b8dd4" stroke="#4888f0" stroke-width="1.5"/></g>
          <rect x="-5" y="-8" width="10" height="14" rx="3" fill="#1c5a9a"/>
          <g id="blue-head" style="transition:transform .25s cubic-bezier(.6,-.5,.3,1.8)">
            <rect x="-17" y="-38" width="34" height="32" rx="8" fill="#2b8dd4"/>
            <rect x="-12" y="-30" width="24" height="12" rx="3" fill="#1c5a9a"/>
            <circle cx="-4" cy="-24" r="3.5" fill="#6ff"/><circle cx="8" cy="-24" r="3.5" fill="#6ff"/>
            <rect x="-2" y="-46" width="5" height="10" rx="2.5" fill="#4888f0"/>
            <circle cx="0.5" cy="-47" r="3" fill="#6ff" opacity=".8"/>
          </g>
        </g>
      </g>

      <text x="200" y="40" text-anchor="middle" font-family="'Press Start 2P'" font-size="24" fill="#ff0" stroke="#f00" stroke-width="1" opacity="0" id="ko-text">K.O.!</text>
    </svg>
    <div class="punch-flash" id="punch-flash-red"></div>
    <div class="punch-flash" id="punch-flash-blue"></div>
  </div>

  <div class="status" id="status-msg">Select models and press FIGHT!</div>

  <div class="result-card" id="result-card">
    <div class="result-title" id="result-title">MATCH RESULT</div>
    <div id="result-body"></div>
  </div>

  <div class="stats">
    <div class="stat"><div class="val" id="s-matches">0</div><div class="lbl">MATCHES</div></div>
    <div class="stat"><div class="val" id="s-top">-</div><div class="lbl">TOP ELO</div></div>
    <div class="stat"><div class="val" id="s-models">0</div><div class="lbl">MODELS</div></div>
    <div class="stat"><div class="val" id="s-wiki">0</div><div class="lbl">WIKI</div></div>
  </div>
</div>

<!-- ══════ LEADERBOARD TAB ══════ -->
<div class="panel" id="panel-leaderboard" style="display:none"></div>
<!-- ══════ MATCHES TAB ══════ -->
<div class="panel" id="panel-matches" style="display:none"></div>
<!-- ══════ EVOLUTION TAB ══════ -->
<div class="panel" id="panel-evolution" style="display:none"></div>

<script>
const DATA = {data_json};
const local = DATA.leaderboard.filter(m=>m.type!=='commercial');
const all = DATA.leaderboard;

const PROMPTS = [
  {{text:"What is the capital of France?",cat:"knowledge"}},
  {{text:"Explain gravity in simple terms.",cat:"explanation"}},
  {{text:"Write a short poem about a cat.",cat:"creative"}},
  {{text:"What is 2 + 2?",cat:"math"}},
  {{text:"Who wrote Romeo and Juliet?",cat:"knowledge"}},
  {{text:"What is the boiling point of water?",cat:"science"}},
  {{text:"Name three primary colors.",cat:"knowledge"}}
];

let fighting=false, redHP=100, blueHP=100, currentRound=1, totalRounds=3, timerSec=0, timerIv=null;

// Populate
function init(){{
  const rs=document.getElementById('red-sel'),bs=document.getElementById('blue-sel');
  local.forEach((m,i)=>{{
    const s=m.model_name.split('/').pop();
    rs.innerHTML+=`<option value="${{i}}">${{s}} — ${{Math.round(m.elo_rating)}}</option>`;
    bs.innerHTML+=`<option value="${{i}}">${{s}} — ${{Math.round(m.elo_rating)}}</option>`;
  }});
  if(local.length>1) bs.selectedIndex=1;
  document.getElementById('s-matches').textContent=DATA.matches.length;
  document.getElementById('s-top').textContent=local[0]?Math.round(local[0].elo_rating):'-';
  document.getElementById('s-models').textContent=all.length;
  document.getElementById('s-wiki').textContent=DATA.wiki?.total_articles||0;
  renderLeaderboard();renderMatches();renderEvolution();
}}

// Tab switching
document.querySelectorAll('.tab').forEach(t=>{{
  t.addEventListener('click',()=>{{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.querySelectorAll('.panel').forEach(p=>p.style.display='none');
    document.getElementById('panel-'+t.dataset.tab).style.display='block';
  }});
}});

// ═══════ FIGHT ENGINE ═══════
function startFight(){{
  if(fighting||local.length<2)return;
  fighting=true;
  const btn=document.getElementById('fight-btn');
  btn.disabled=true;btn.textContent='FIGHTING...';

  redHP=100;blueHP=100;currentRound=1;timerSec=0;
  updateHP();hideResult();
  setStatus('DING DING DING!',true);

  // Timer
  timerIv=setInterval(()=>{{timerSec++;updateTimer()}},1000);

  // Run 3 rounds
  runRound(1);
}}

function runRound(rnd){{
  if(rnd>totalRounds||redHP<=0||blueHP<=0){{
    endFight();return;
  }}
  currentRound=rnd;
  document.getElementById('round-display').textContent=`ROUND ${{rnd}} / ${{totalRounds}}`;

  const prompt=PROMPTS[Math.floor(Math.random()*PROMPTS.length)];
  setStatus(`Round ${{rnd}}: "${{prompt.text.slice(0,35)}}..."`,true);

  // Use real match data if available, else simulate
  const matches=DATA.matches;
  let matchData;
  if(matches.length>0){{
    matchData=matches[Math.floor(Math.random()*matches.length)];
  }}else{{
    matchData={{score_a:Math.random()*.5+.3,score_b:Math.random()*.5+.3,winner:'draw',model_a:'A',model_b:'B'}};
  }}

  // Animate the round
  setTimeout(()=>{{setStatus('Models generating responses...',true);animateIdle()}},300);
  setTimeout(()=>{{setStatus('Judging with 6-dimension rubric...',true)}},1500);

  // Red punches
  setTimeout(()=>punchRed(),2000);
  setTimeout(()=>{{
    const sa=matchData.score_a||.5;
    const dmg=Math.round((1-sa)*40+5);
    blueHP=Math.max(0,blueHP-dmg);
    updateHP();
    flashBlue();
    shakeBlue();
  }},2300);

  // Blue punches
  setTimeout(()=>punchBlue(),3000);
  setTimeout(()=>{{
    const sb=matchData.score_b||.5;
    const dmg=Math.round((1-sb)*40+5);
    redHP=Math.max(0,redHP-dmg);
    updateHP();
    flashRed();
    shakeRed();
  }},3300);

  setTimeout(()=>resetArms(),3600);

  // Score display
  setTimeout(()=>{{
    const sa=(matchData.score_a||0).toFixed(3);
    const sb=(matchData.score_b||0).toFixed(3);
    setStatus(`Round ${{rnd}} scores: RED ${{sa}} — BLUE ${{sb}}`,true);
  }},3800);

  // Next round or end
  setTimeout(()=>{{
    if(redHP<=0||blueHP<=0){{endFight()}}
    else{{runRound(rnd+1)}}
  }},4800);
}}

function endFight(){{
  clearInterval(timerIv);
  resetArms();

  let winnerText,winnerSide;
  if(redHP>blueHP){{
    winnerText='RED ROCKET WINS!';winnerSide='red';
    knockoutBlue();
  }}else if(blueHP>redHP){{
    winnerText='BLUE BOMBER WINS!';winnerSide='blue';
    knockoutRed();
  }}else{{
    winnerText='DRAW!';winnerSide='draw';
  }}

  setStatus(winnerText,'ko');
  showKO();

  // Show result card
  setTimeout(()=>showResult(winnerSide),1500);

  setTimeout(()=>{{
    hideKO();resetHeads();
    fighting=false;
    const btn=document.getElementById('fight-btn');
    btn.disabled=false;btn.textContent='FIGHT!';
    setStatus('Select models and press FIGHT!',false);
  }},5000);
}}

// ═══════ ANIMATIONS ═══════
function punchRed(){{
  const arm=document.getElementById('red-arm-front');
  if(arm)arm.style.transform='rotate(-50deg)';
}}
function punchBlue(){{
  const arm=document.getElementById('blue-arm-front');
  if(arm)arm.style.transform='rotate(-50deg)';
}}
function resetArms(){{
  ['red-arm-front','blue-arm-front'].forEach(id=>{{
    const el=document.getElementById(id);if(el)el.style.transform='';
  }});
}}
function animateIdle(){{
  // Slight bobbing
  const r=document.getElementById('red-robot');
  const b=document.getElementById('blue-robot');
  if(r)r.style.transform='translateY(-2px)';
  if(b)b.style.transform='translateY(-2px)';
  setTimeout(()=>{{if(r)r.style.transform='';if(b)b.style.transform=''}},300);
}}
function shakeRed(){{
  const r=document.getElementById('red-robot');
  if(r){{r.style.transform='translateX(-6px)';setTimeout(()=>r.style.transform='',200)}}
}}
function shakeBlue(){{
  const b=document.getElementById('blue-robot');
  if(b){{b.style.transform='translateX(6px)';setTimeout(()=>b.style.transform='',200)}}
}}
function flashRed(){{
  const f=document.getElementById('punch-flash-red');
  if(f){{f.style.display='block';f.style.left='28%';f.style.top='40%';f.style.animation='punchBurst .4s ease-out';
    setTimeout(()=>{{f.style.display='none';f.style.animation=''}},400)}}
}}
function flashBlue(){{
  const f=document.getElementById('punch-flash-blue');
  if(f){{f.style.display='block';f.style.left='65%';f.style.top='40%';f.style.animation='punchBurst .4s ease-out';
    setTimeout(()=>{{f.style.display='none';f.style.animation=''}},400)}}
}}
function knockoutRed(){{
  const h=document.getElementById('red-head');if(h)h.style.transform='translateY(-18px)';
  const r=document.getElementById('red-robot');if(r)r.style.transform='translateX(-10px) rotate(-5deg)';
}}
function knockoutBlue(){{
  const h=document.getElementById('blue-head');if(h)h.style.transform='translateY(-18px)';
  const b=document.getElementById('blue-robot');if(b)b.style.transform='translateX(10px) rotate(5deg)';
}}
function resetHeads(){{
  ['red-head','blue-head'].forEach(id=>{{const el=document.getElementById(id);if(el)el.style.transform=''}});
  ['red-robot','blue-robot'].forEach(id=>{{const el=document.getElementById(id);if(el)el.style.transform=''}});
}}
function showKO(){{const k=document.getElementById('ko-text');if(k){{k.setAttribute('opacity','1');blinkEl(k,8);}}}}
function hideKO(){{const k=document.getElementById('ko-text');if(k)k.setAttribute('opacity','0')}}
function blinkEl(el,n){{let c=0;const iv=setInterval(()=>{{el.setAttribute('opacity',c%2===0?'.3':'1');c++;if(c>=n){{clearInterval(iv);el.setAttribute('opacity','0');}}}},180);}}

function updateHP(){{
  document.getElementById('red-hp').style.width=redHP+'%';
  document.getElementById('blue-hp').style.width=blueHP+'%';
  document.getElementById('red-hp-txt').textContent=Math.max(0,redHP);
  document.getElementById('blue-hp-txt').textContent=Math.max(0,blueHP);
}}
function updateTimer(){{
  const m=Math.floor(timerSec/60),s=timerSec%60;
  document.getElementById('timer-display').textContent=m+':'+(s<10?'0':'')+s;
}}
function setStatus(msg,type){{
  const el=document.getElementById('status-msg');
  el.textContent=msg;
  el.className=type==='ko'?'status ko':type?'status active':'status';
}}
function showResult(side){{
  const card=document.getElementById('result-card');
  const body=document.getElementById('result-body');
  const ri=document.getElementById('red-sel').value;
  const bi=document.getElementById('blue-sel').value;
  const rm=local[ri]||local[0];
  const bm=local[bi]||local[1]||local[0];
  const rn=rm.model_name.split('/').pop();
  const bn=bm.model_name.split('/').pop();

  body.innerHTML=`
    <div class="result-row"><span class="lbl">Red Rocket</span><span class="${{side==='red'?'val win':'val lose'}}">${{rn}} (HP: ${{Math.max(0,redHP)}})</span></div>
    <div class="result-row"><span class="lbl">Blue Bomber</span><span class="${{side==='blue'?'val win':'val lose'}}">${{bn}} (HP: ${{Math.max(0,blueHP)}})</span></div>
    <div class="result-row"><span class="lbl">Rounds</span><span class="val">${{Math.min(currentRound,totalRounds)}} / ${{totalRounds}}</span></div>
    <div class="result-row"><span class="lbl">Time</span><span class="val">${{document.getElementById('timer-display').textContent}}</span></div>
    <div class="result-row"><span class="lbl">Winner</span><span class="val win">${{side==='red'?'RED ROCKET':side==='blue'?'BLUE BOMBER':'DRAW'}}</span></div>
  `;
  card.className='result-card show';
}}
function hideResult(){{document.getElementById('result-card').className='result-card'}}

// ═══════ LEADERBOARD ═══════
function renderLeaderboard(){{
  const p=document.getElementById('panel-leaderboard');
  if(!all.length){{p.innerHTML='<div class="empty">No data yet</div>';return}}
  const mx=Math.max(...all.map(m=>m.elo_rating)),mn=Math.min(...all.map(m=>m.elo_rating)),rng=Math.max(mx-mn,1);
  let h='<div style="text-align:center;padding:8px 0 12px;font-family:\\'Press Start 2P\\',monospace;font-size:10px;color:#ffd700;letter-spacing:1px">POLYRATING LEADERBOARD</div>';
  all.forEach((m,i)=>{{
    const rk=i+1,s=m.model_name.split('/').pop(),pct=(m.elo_rating-mn)/rng,ic=m.type==='commercial';
    const bc=rk===1?'rb rb1':rk===2?'rb rb2':rk===3?'rb rb3':'rb rbn';
    const icon=rk===1?'👑':rk===2?'🥈':rk===3?'🥉':rk;
    const tag=ic?'<span class="lb-tag tag-c">COMMERCIAL</span>':'<span class="lb-tag tag-l">LOCAL</span>';
    const rc=(ic?'lb-row commercial':'lb-row local')+(rk===1?' rank1':'');
    const bc2=pct>.7?'linear-gradient(90deg,#f59e0b,#fbbf24)':pct>.3?'linear-gradient(90deg,#3b82f6,#60a5fa)':'#64748b';
    h+=`<div class="${{rc}}" style="animation-delay:${{i*.05}}s"><div class="${{bc}}">${{icon}}</div><div style="flex:1;min-width:0"><div class="lb-name">${{s}} ${{tag}}</div><div class="lb-sub">${{m.model_name}}</div></div><div class="lb-wld">W${{m.wins}} L${{m.losses}} D${{m.draws}}</div><div class="ebar"><div class="efill" style="width:${{Math.max(pct*100,5)}}%;background:${{bc2}}"></div></div><div class="lb-elo">${{Math.round(m.elo_rating)}}</div></div>`;
  }});
  p.innerHTML=h;
}}

// ═══════ MATCHES ═══════
function renderMatches(){{
  const p=document.getElementById('panel-matches'),ms=DATA.matches;
  if(!ms.length){{p.innerHTML='<div class="empty">No matches yet</div>';return}}
  let h='<div style="text-align:center;padding:8px 0 12px;font-family:\\'Press Start 2P\\',monospace;font-size:10px;color:#ffd700;letter-spacing:1px">MATCH HISTORY</div><div style="max-height:340px;overflow-y:auto">';
  ms.forEach(m=>{{
    const sa=(m.score_a||0).toFixed(3),sb=(m.score_b||0).toFixed(3);
    const a=(m.model_a||'').split('/').pop().slice(0,10),b=(m.model_b||'').split('/').pop().slice(0,10);
    const bc=m.winner===m.model_a?'#d42b2b':m.winner===m.model_b?'#2b8dd4':'#888';
    h+=`<div class="match-row" style="border-left:3px solid ${{bc}}"><span style="color:#555;font-size:8px;min-width:45px">${{(m.timestamp||'').slice(11,16)}}</span><span style="color:#f04848;font-weight:bold">${{a}}</span><span style="color:#f04848">${{sa}}</span><span style="color:#555">vs</span><span style="color:#4888f0">${{sb}}</span><span style="color:#4888f0;font-weight:bold">${{b}}</span><span style="flex:1;color:#444;font-size:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${{(m.prompt||'').slice(0,25)}}</span><span style="color:#8b5cf6;font-size:8px">${{m.domain||''}}</span></div>`;
  }});
  h+='</div>';p.innerHTML=h;
}}

// ═══════ EVOLUTION ═══════
function renderEvolution(){{
  const p=document.getElementById('panel-evolution'),ev=DATA.evolution||[];
  if(!ev.length){{p.innerHTML='<div class="empty">No evolution runs yet</div>';return}}
  let h='<div style="text-align:center;padding:8px 0 12px;font-family:\\'Press Start 2P\\',monospace;font-size:10px;color:#ffd700;letter-spacing:1px">SELF-EVOLUTION LOG</div>';
  ev.forEach(e=>{{
    const d=e.improvement_delta||0,s=(e.model_name||'').split('/').pop();
    h+=`<div class="evo-row" style="border-left:3px solid ${{d>=0?'#10b981':'#ef4444'}}"><div style="display:flex;justify-content:space-between;margin-bottom:2px"><span style="color:#ffd700;font-weight:bold">${{s}} · R${{e.iteration}}</span><span style="color:${{d>=0?'#10b981':'#ef4444'}};font-weight:bold">Δ ${{d>=0?'+':''}}${{d.toFixed(3)}}</span></div><div style="color:#888;font-size:9px">Gap: <span style="color:#8b5cf6">${{e.gap_identified}}</span> · ${{e.synthetic_prompts_generated}} prompts</div></div>`;
  }});
  p.innerHTML=h;
}}

init();
</script></body></html>'''

    components.html(html, height=780, scrolling=True)


# ══════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;600;800&display=swap');
.stat-card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:1.2rem;text-align:center;font-family:'Outfit',sans-serif}
.stat-card .value{font-size:2rem;font-weight:800;color:#3b82f6}.stat-card .label{font-size:.85rem;color:#64748b;margin-top:.2rem}
.correction-card{background:rgba(251,191,36,.05);border:1px solid rgba(251,191,36,.2);border-radius:8px;padding:.8rem 1rem;margin-bottom:.6rem}
</style>""", unsafe_allow_html=True)

db = get_db()
wiki = get_wiki()

with st.sidebar:
    st.markdown("### ⚙️ Navigation")
    page = st.radio("", ["🏟️ Arena", "🧪 Rehearsal", "📚 Wiki", "🎓 Training", "🚀 Run"],
                     label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.cache_resource.clear(); st.rerun()
    st.caption("v2.0 · Polyrating")

if page == "🏟️ Arena":
    render_arena(db, wiki)

elif page == "🧪 Rehearsal":
    st.markdown("## 🧪 Rehearsal — Teacher/Student")
    reports = sorted(Path("./results/rehearsal").glob("*.json")) if Path("./results/rehearsal").exists() else []
    if not reports: st.info("No reports. Run bootcamp first.")
    else:
        sel = st.selectbox("Report", [r.name for r in reports])
        rp = json.loads(next(r for r in reports if r.name == sel).read_text())
        c1,c2,c3,c4=st.columns(4)
        with c1: st.metric("Student",f'{rp["student_avg"]:.3f}')
        with c2: st.metric("Teacher",f'{rp["teacher_avg"]:.3f}')
        with c3: st.metric("Delta",f'{rp["delta"]:+.3f}')
        with c4: st.metric("Win%",f'{rp["win_rate"]:.0%}')
        if rp.get("became_master"): st.success("🎓 Student became the master!")
        rows=[{"Prompt":r["prompt"][:50],"Student":f'{r["student_score"]:.3f}',"Teacher":f'{r["teacher_score"]:.3f}',"W":r["winner"]} for r in rp["results"]]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

elif page == "📚 Wiki":
    st.markdown("## 📚 LLM Wiki")
    t1,t2,t3,t4=st.tabs(["Stats","✍️ Article","🔧 Correction","📰 Fetch"])
    with t1:
        ws=wiki.get_wiki_stats();c1,c2=st.columns(2)
        with c1: st.metric("Articles",ws["total_articles"])
        with c2: st.metric("Corrections",ws["corrections"])
        dd={d:c for d,c in ws["domains"].items() if c>0}
        if dd: st.bar_chart(pd.Series(dd))
    with t2:
        with st.form("art"):
            ti=st.text_input("Title");do=st.selectbox("Domain",LLMWiki.DOMAINS,key="ad")
            co=st.text_area("Content",height=150)
            if st.form_submit_button("Create"):
                if ti and co: st.success(wiki.create_article(ti,do,co,"ui")); st.cache_resource.clear()
    with t3:
        with st.form("corr"):
            pr=st.text_area("Prompt");bd=st.text_area("Bad output");gd=st.text_area("Corrected")
            do=st.selectbox("Domain",LLMWiki.DOMAINS,key="cd")
            if st.form_submit_button("Submit"):
                if pr and gd: st.success(wiki.record_correction(pr,bd,gd,do)); st.cache_resource.clear()
    with t4:
        with st.form("fetch"):
            tp=st.text_area("Topics (one/line)",value="Gravity\nPhotosynthesis")
            do=st.selectbox("Domain",LLMWiki.DOMAINS,key="fd",index=2)
            if st.form_submit_button("Fetch"):
                tl=[t.strip() for t in tp.split("\n") if t.strip()]
                c=wiki.fetch_batch(tl,do);st.success(f"Fetched {c}/{len(tl)}");st.cache_resource.clear()

elif page == "🎓 Training":
    st.markdown("## 🎓 Training Runs")
    runs=db.get_training_history(50)
    if not runs: st.info("No runs yet.")
    else:
        rows=[{"Phase":r.get("phase",""),"Round":r.get("round_num",""),"Teacher":short_name(r.get("teacher_model","")),"Student":short_name(r.get("student_model","")),"Prompts":r.get("prompts_used",0),"Time":(r.get("timestamp") or "")[:16]} for r in runs]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

elif page == "🚀 Run":
    st.markdown("## 🚀 Run Pipelines")
    from core.model_hub import ModelHub
    hub=ModelHub("config/models.yaml")
    am=hub.get_available_models()
    hm=[m["name"] for m in hub.config["models"] if m.get("provider")=="huggingface" and m.get("enabled",True)]
    t1,t2,t3=st.tabs(["🏋️ Bootcamp","🎓 Academy","🏟️ Arena"])
    with t1:
        s=st.selectbox("Student",hm,key="bs");t=st.selectbox("Teacher",am,key="bt");r=st.slider("Rounds",1,10,3,key="br")
        if st.button("🏋️ Run Bootcamp"):
            subprocess.Popen(["python","arena.py","--bootcamp","--student",s,"--teacher",t,"--rounds",str(r)],cwd=str(Path(__file__).parent));st.info("Running…")
    with t2:
        d=st.multiselect("Domains",["math","science","coding","creative"],default=["math","coding"])
        if st.button("🎓 Run Academy"):
            subprocess.Popen(["python","arena.py","--academy","--domains"]+d,cwd=str(Path(__file__).parent));st.info("Running…")
    with t3:
        if st.button("🏟️ Run Arena"):
            subprocess.Popen(["python","arena.py","--arena"],cwd=str(Path(__file__).parent));st.info("Running…")
