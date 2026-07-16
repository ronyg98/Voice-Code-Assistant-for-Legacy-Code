/* SignSpeak frontend — Live Studio (webcam → /api/predict) + Signy chat (/api/chat) */
"use strict";

/* ════════════════════════ helpers ════════════════════════ */

const $ = (id) => document.getElementById(id);

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],        // index
  [5, 9], [9, 10], [10, 11], [11, 12],   // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20],// pinky
  [0, 17],
];

const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

/* ════════════════════════ health / status pill ════════════════════════ */

async function checkHealth() {
  const dot = document.querySelector("#server-status .dot");
  const text = $("server-status-text");
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    if (h.letter_model) {
      dot.className = "dot dot-ok";
      const extras = [];
      if (h.two_hand_model) extras.push("2-hand");
      if (h.providers.length) extras.push(h.providers[0] + " AI");
      text.textContent = "Models ready" + (extras.length ? " · " + extras.join(" · ") : "");
    } else {
      dot.className = "dot dot-bad";
      text.textContent = "Letter model missing";
    }
  } catch {
    dot.className = "dot dot-bad";
    text.textContent = "Server offline";
  }
}
checkHealth();

/* ════════════════════════ vocabulary grid ════════════════════════ */

const letterGrid = $("letter-grid");
for (const letter of LETTERS) {
  const chip = document.createElement("div");
  chip.className = "letter-chip";
  chip.id = "letter-chip-" + letter;
  chip.textContent = letter;
  letterGrid.appendChild(chip);
}

function flashChip(label) {
  const chip = $("letter-chip-" + label);
  if (!chip) return;
  chip.classList.add("flash");
  setTimeout(() => chip.classList.remove("flash"), 900);
}

/* ════════════════════════ live studio ════════════════════════ */

const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
const cctx = captureCanvas.getContext("2d");

const STABLE_N = 8;          // consecutive identical predictions to commit

let stream = null;
let running = false;
let sentence = "";
let holdCount = 0;
let holdKey = null;          // "label|type" currently being held
let lastCommitted = null;

function setSentence(s) {
  sentence = s;
  const box = $("sentence-box");
  if (!sentence) {
    box.innerHTML = '<span class="sentence-placeholder">Signed letters and words appear here…</span>';
  } else {
    box.textContent = sentence;
    const caret = document.createElement("span");
    caret.className = "sentence-caret";
    box.appendChild(caret);
  }
}

function commit(label, type) {
  if (type === "word") {
    let s = sentence;
    if (s && !s.endsWith(" ")) s += " ";
    setSentence(s + label + " ");
  } else {
    setSentence(sentence + label);
  }
  flashChip(label);
}

function updatePredictionUI(label, type, confidence, holding) {
  const labelEl = $("predict-label");
  const typeEl = $("predict-type");
  const ring = $("hold-ring");
  const CIRC = 326.7;

  if (label) {
    labelEl.textContent = label;
    labelEl.classList.toggle("word-label", type === "word");
    typeEl.textContent = type === "word" ? "two-hand word — hold it…" : "fingerspelled letter — hold it…";
  } else {
    labelEl.textContent = "–";
    labelEl.classList.remove("word-label");
    typeEl.textContent = running ? "show a hand to begin" : "camera is off";
  }
  ring.style.strokeDashoffset = CIRC * (1 - Math.min(holding / STABLE_N, 1));
  $("conf-fill").style.width = Math.round(confidence * 100) + "%";
  $("conf-text").textContent = "confidence " + Math.round(confidence * 100) + "%";
}

function drawHands(hands) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const w = overlay.width, h = overlay.height;
  for (const hand of hands) {
    octx.strokeStyle = "rgba(34, 211, 238, 0.9)";
    octx.lineWidth = 2.5;
    octx.shadowColor = "rgba(34, 211, 238, 0.7)";
    octx.shadowBlur = 6;
    for (const [a, b] of HAND_CONNECTIONS) {
      octx.beginPath();
      octx.moveTo(hand[a].x * w, hand[a].y * h);
      octx.lineTo(hand[b].x * w, hand[b].y * h);
      octx.stroke();
    }
    octx.shadowBlur = 0;
    for (const p of hand) {
      octx.beginPath();
      octx.arc(p.x * w, p.y * h, 3.4, 0, Math.PI * 2);
      octx.fillStyle = "#8b5cf6";
      octx.fill();
      octx.strokeStyle = "rgba(255,255,255,0.85)";
      octx.lineWidth = 1;
      octx.stroke();
    }
  }
}

async function predictLoop() {
  if (!running) return;
  const t0 = performance.now();
  try {
    cctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const frame = captureCanvas.toDataURL("image/jpeg", 0.85);
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame }),
    });
    if (!res.ok) throw new Error("predict failed");
    const data = await res.json();

    drawHands(data.hands || []);
    $("hud-hands").textContent = "✋ " + (data.hands ? data.hands.length : 0) +
      (data.hands && data.hands.length === 1 ? " hand" : " hands");
    const dt = performance.now() - t0;
    $("hud-fps").textContent = (1000 / Math.max(dt, 1)).toFixed(1) + " fps";

    // stability / commit logic (mirrors main.py)
    const key = data.label ? data.label + "|" + data.type : null;
    if (key !== holdKey) {
      holdKey = key;
      holdCount = 0;
      lastCommitted = null;
    }
    if (key) {
      holdCount = Math.min(holdCount + 1, STABLE_N);
      if (holdCount === STABLE_N && lastCommitted !== key) {
        commit(data.label, data.type);
        lastCommitted = key;
      }
    }
    updatePredictionUI(data.label, data.type, data.confidence || 0, key ? holdCount : 0);
  } catch {
    /* server hiccup — keep looping */
  }
  setTimeout(predictLoop, 30);
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch {
    alert("Could not access the webcam. Please allow camera permission and try again.");
    return;
  }
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();

  // full camera resolution - downscaling degrades landmark precision enough
  // to break similar letters (U/V) and small two-hand detections
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  $("video-placeholder").style.display = "none";
  $("video-hud").hidden = false;
  $("btn-stop").hidden = false;
  running = true;
  predictLoop();
}

function stopCamera() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $("video-placeholder").style.display = "flex";
  $("video-hud").hidden = true;
  $("btn-stop").hidden = true;
  holdKey = null;
  holdCount = 0;
  updatePredictionUI(null, null, 0, 0);
}

$("btn-start").addEventListener("click", startCamera);
$("btn-stop").addEventListener("click", stopCamera);
$("btn-space").addEventListener("click", () => setSentence(sentence + " "));
$("btn-back").addEventListener("click", () => setSentence(sentence.slice(0, -1)));
$("btn-clear").addEventListener("click", () => setSentence(""));
$("btn-speak").addEventListener("click", () => {
  const text = sentence.trim();
  if (!text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.95;
  const enVoice = speechSynthesis.getVoices().find((v) => v.lang.startsWith("en"));
  if (enVoice) utter.voice = enVoice;
  speechSynthesis.speak(utter);
});

/* ════════════════════════ chat ════════════════════════ */

const chatMessages = $("chat-messages");
const chatInput = $("chat-input");
const chatForm = $("chat-form");
const chatSend = $("chat-send");
const history = []; // {role, content} sent to the server

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* tiny markdown: **bold**, *italic*, `code`, -/* bullets, 1. lists */
function renderMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  let html = "", list = null; // "ul" | "ol" | null

  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const inline = (s) =>
    s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
     .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
     .replace(/`([^`]+)`/g, "<code>$1</code>");

  for (const raw of lines) {
    const line = raw.trim();
    if (/^[-*]\s+/.test(line)) {
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += "<li>" + inline(line.replace(/^[-*]\s+/, "")) + "</li>";
    } else if (/^\d+[.)]\s+/.test(line)) {
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += "<li>" + inline(line.replace(/^\d+[.)]\s+/, "")) + "</li>";
    } else if (line === "") {
      closeList();
    } else {
      closeList();
      html += "<p>" + inline(line) + "</p>";
    }
  }
  closeList();
  return html;
}

function addMessage(role, html, extraClass = "") {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "msg-user" : "msg-bot") + (extraClass ? " " + extraClass : "");
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

async function sendChat(text) {
  text = text.trim();
  if (!text || chatSend.disabled) return;

  addMessage("user", escapeHtml(text));
  history.push({ role: "user", content: text });
  chatInput.value = "";
  chatInput.style.height = "auto";
  chatSend.disabled = true;

  const typingBubble = addMessage("bot", '<span class="typing"><span></span><span></span><span></span></span>');

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "request failed");

    typingBubble.innerHTML = renderMarkdown(data.reply);
    history.push({ role: "assistant", content: data.reply });
    if (data.provider) $("chat-provider").textContent = "sign language expert · powered by " + data.provider;
  } catch (err) {
    typingBubble.parentElement.classList.add("msg-error");
    typingBubble.innerHTML = "⚠️ Couldn't reach the assistant. " + escapeHtml(String(err.message || err));
    history.pop(); // let the user retry the same question
  } finally {
    chatSend.disabled = false;
    chatInput.focus();
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendChat(chatInput.value);
});
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat(chatInput.value);
  }
});
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px";
});
document.querySelectorAll("#chat-suggestions .chip").forEach((chip) =>
  chip.addEventListener("click", () => sendChat(chip.textContent))
);

/* ════════════════════════ floating ASL chart ════════════════════════ */

const chartFab = $("chart-fab");
const chartPopup = $("chart-popup");

function toggleChart(force) {
  const open = force !== undefined ? force : chartPopup.hidden;
  chartPopup.hidden = !open;
  chartFab.classList.toggle("open", open);
  chartFab.setAttribute("aria-expanded", String(open));
  $("chart-fab-icon").textContent = open ? "+" : "✋";
}

chartFab.addEventListener("click", () => toggleChart());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !chartPopup.hidden) toggleChart(false);
});

/* preload speech voices (Chrome loads them lazily) */
if ("speechSynthesis" in window) speechSynthesis.getVoices();
