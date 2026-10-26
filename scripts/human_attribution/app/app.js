(function () {
  "use strict";
  const S = window.STUDY;
  const KEY = "vmem_attr_" + S.version + "_" + S.seed;
  const PANELS = ["earlier", "refs", "keyframe", "clip"];
  const KIND = { character: "角色", prop: "道具", location: "场景" };
  const store = JSON.parse(localStorage.getItem(KEY) || '{"answers":{},"pos":0,"annotator":""}');
  const app = document.getElementById("app");
  let pos = Math.min(store.pos || 0, S.items.length - 1);
  let active = 0;
  let shownAt = Date.now();

  if (!store.annotator) {
    store.annotator = (prompt("请输入你的标注员代号（例如 H1）：") || "H1").trim();
    save();
  }

  function save() {
    store.pos = pos;
    localStorage.setItem(KEY, JSON.stringify(store));
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  }

  function questions(item) {
    const n = esc(item.card.name);
    if (item.target === "must_be_absent") {
      return {
        earlier: `这段<b>之前的片段</b>里，能看到 ${n} 吗？`,
        refs: `三张<b>记忆参考图</b>中，任意一张能看到 ${n} 吗？`,
        keyframe: `<b>关键帧</b>里能看到 ${n} 吗？`,
        clip: `<b>生成的片段</b>里能看到 ${n} 吗？`,
      };
    }
    const cond = item.card.required_state ? "（并且处于要求的状态）" : "";
    return {
      earlier: `这段<b>之前的片段</b>里，能看到 ${n} 吗？`,
      refs: `三张<b>记忆参考图</b>中，任意一张有正确的 ${n}${cond} 吗？`,
      keyframe: `<b>关键帧</b>里有正确的 ${n}${cond} 吗？`,
      clip: `<b>生成的片段</b>里有正确的 ${n}${cond} 吗？`,
    };
  }

  function render() {
    const item = S.items[pos];
    const ans = store.answers[item.display_id] || {};
    const q = questions(item);
    const m = "media/" + item.display_id + "/";
    const c = item.card;
    const absent = item.target === "must_be_absent";
    const scored = S.items.filter((x) => !x.practice);
    const done = scored.filter((x) => isComplete(store.answers[x.display_id])).length;
    document.getElementById("progress").textContent = `第 ${pos + 1} / ${S.items.length} 题　已完成正式题 ${done} / ${scored.length}`;
    const modeEl = document.getElementById("mode");
    modeEl.textContent = item.practice ? "练习题（不计分）" : "正式题";
    modeEl.className = "badge" + (item.practice ? " practice" : "");

    let la = "";
    if (c.lookalike_note) {
      const f = Object.values(c.lookalike_note).map(esc).join("<br>");
      la = `<div class="la">注意：${esc(c.name)} 与 ${esc(c.lookalike_other)} 长得很像，请按下面的区别判断：<br>${f}</div>`;
    }
    const rule = absent
      ? `此刻 ${esc(c.name)} 已经<b>不应该再出现</b>。下面的问题只问“画面里有没有它”，不问对错。`
      : `此刻画面里<b>应该出现</b> ${esc(c.name)}。“正确”指外观与下面的描述、以及之前片段里的样子一致。`;
    const state = c.required_state ? `<div class="desc"><b>要求的状态：</b>${esc(c.required_state)}</div>` : "";

    app.innerHTML = `
      <div class="target ${absent ? "absent" : ""}">
        <h2>${esc(c.name)} <span class="badge">${esc(KIND[c.kind] || c.kind)}</span></h2>
        <div class="rule">${rule}</div>
        <div class="desc"><b>外观：</b>${esc(c.appearance)}</div>
        ${state}${la}
      </div>
      <div class="grid">
        ${panel(0, "① 之前出现的片段", `<video src="${m}earlier.mp4" autoplay loop muted playsinline controls></video>`, "来自同一个故事更早的片段，作为参照。", q.earlier, ans.earlier)}
        ${panel(1, "② 记忆参考图（3 张）", `<div class="refs">${[0, 1, 2].map((i) => `<img src="${m}ref_${i}.jpg" data-zoom>`).join("")}</div>`, "记忆模块交给生成器的参考图，点击可放大。", q.refs, ans.refs)}
        ${panel(2, "③ 合成的关键帧", `<img src="${m}keyframe.jpg" data-zoom>`, "由参考图和文字合成的首帧，点击可放大。", q.keyframe, ans.keyframe)}
        ${panel(3, "④ 生成的片段", `<video src="${m}clip.mp4" autoplay loop muted playsinline controls></video>`, `本段文字：${esc(item.intent)}`, q.clip, ans.clip)}
      </div>
      <div class="footer">
        <input id="note" placeholder="备注（可选，例如：画面太模糊、看不清）" value="${esc(ans.note || "")}">
      </div>`;
    app.querySelectorAll(".q button").forEach((b) => b.addEventListener("click", () => answer(Number(b.dataset.p), b.dataset.v)));
    app.querySelectorAll("[data-zoom]").forEach((img) => img.addEventListener("click", () => zoom(img.src)));
    document.getElementById("note").addEventListener("input", (e) => setAns({ note: e.target.value }));
    active = PANELS.findIndex((p) => !ans[p]);
    if (active < 0) active = 3;
    highlight();
    shownAt = Date.now();
    save();
  }

  function panel(i, title, body, hint, question, val) {
    const btn = (v, label, k) => `<button data-p="${i}" data-v="${v}" class="${val === v ? "sel" : ""}">${label}（${k}）</button>`;
    return `<div class="panel" data-i="${i}"><h3>${title}</h3>${body}<div class="hint">${hint}</div>
      <div>${question}</div><div class="q">${btn("yes", "有", 1)}${btn("no", "没有", 2)}${btn("unsure", "不确定", 3)}</div></div>`;
  }

  function highlight() {
    app.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", Number(p.dataset.i) === active));
  }

  function setAns(patch) {
    const id = S.items[pos].display_id;
    const cur = store.answers[id] || { first_seen: new Date().toISOString(), ms: 0 };
    store.answers[id] = Object.assign(cur, patch, { updated: new Date().toISOString() });
    save();
  }

  function answer(i, v) {
    setAns({ [PANELS[i]]: v });
    app.querySelectorAll(`.q button[data-p="${i}"]`).forEach((b) => b.classList.toggle("sel", b.dataset.v === v));
    active = Math.min(i + 1, 3);
    highlight();
  }

  function isComplete(a) {
    return !!a && PANELS.every((p) => a[p]);
  }

  function go(delta) {
    const id = S.items[pos].display_id;
    if (store.answers[id]) store.answers[id].ms = (store.answers[id].ms || 0) + (Date.now() - shownAt);
    if (delta > 0 && !isComplete(store.answers[id])) {
      if (!confirm("这题还有问题没答，确定跳到下一题吗？")) return;
    }
    pos = Math.max(0, Math.min(S.items.length - 1, pos + delta));
    save();
    render();
    window.scrollTo(0, 0);
    const scored = S.items.filter((x) => !x.practice);
    const done = scored.filter((x) => isComplete(store.answers[x.display_id])).length;
    if (delta > 0 && done > 0 && done % 40 === 0) exportData(true);
  }

  function exportData(auto) {
    const payload = { version: S.version, seed: S.seed, annotator: store.annotator, exported: new Date().toISOString(), answers: store.answers };
    const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `labels_${store.annotator}_${S.version}${auto ? "_autosave" : ""}.json`;
    a.click();
  }

  function zoom(src) {
    const z = document.getElementById("zoom");
    z.querySelector("img").src = src;
    z.style.display = "flex";
  }
  document.getElementById("zoom").addEventListener("click", (e) => (e.currentTarget.style.display = "none"));

  document.getElementById("prev").onclick = () => go(-1);
  document.getElementById("next").onclick = () => go(1);
  document.getElementById("export").onclick = () => exportData(false);
  document.getElementById("help").onclick = () => window.open("INSTRUCTIONS.md", "_blank");

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "1" || e.key === "2" || e.key === "3") answer(active, { 1: "yes", 2: "no", 3: "unsure" }[e.key]);
    else if (e.key === "Enter") go(1);
    else if (e.key === "ArrowLeft") go(-1);
    else if (e.key === "ArrowRight") go(1);
    else if (e.key.toLowerCase() === "r") app.querySelectorAll("video").forEach((v) => { v.currentTime = 0; v.play(); });
    else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); highlight(); }
    else if (e.key === "ArrowDown") { active = Math.min(3, active + 1); highlight(); }
  });

  render();
})();
