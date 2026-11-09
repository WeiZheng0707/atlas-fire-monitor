/* ==========================================================================
 * Atlas 200I DK A2 视觉火情监测 · 前端
 * - SSE /api/stream 实时检测
 * - /api/health 模型与板端状态
 * - /api/detect 单图验证；/api/events 证据
 * ========================================================================== */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var state = {
    es: null,
    lastSeq: -1,
    lastAlarmId: null,
    lastCount: null,
    events: [],
    health: null,
    model: null,
    file: null
  };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  var LABEL_CN = { fire: "火焰", smoke: "烟雾" };

  function cn(label) {
    var key = String(label == null ? "" : label).toLowerCase();
    return LABEL_CN[key] || label || "目标";
  }

  // 分级文案：1=烟雾预警，2=火情告警
  var LEVEL_KEY = { 1: "warn", 2: "fire" };

  function levelName(level) {
    return level === 1 ? "烟雾预警" : "火情告警";
  }


  function nowTime() {
    var d = new Date();
    return [d.getHours(), d.getMinutes(), d.getSeconds()]
      .map(function (v) { return String(v).padStart(2, "0"); }).join(":");
  }

  function fmtMB(mb) {
    if (mb == null) return "--";
    return (mb / 1024).toFixed(1) + " GB";
  }

  function showToast(text, type) {
    var root = $("toastRoot");
    var el = document.createElement("div");
    el.className = "toast " + (type || "");
    el.textContent = text;
    root.appendChild(el);
    setTimeout(function () {
      el.classList.add("out");
      setTimeout(function () { el.remove(); }, 220);
    }, 3200);
  }

  function log(text, type) {
    var list = $("logList");
    var el = document.createElement("div");
    el.className = "log-item " + (type || "");
    el.innerHTML = "<time>" + nowTime() + "</time><span>" + esc(text) + "</span>";
    list.insertBefore(el, list.firstChild);
    while (list.children.length > 60) list.removeChild(list.lastChild);
  }

  function tickClock() {
    var d = new Date();
    $("clockTime").textContent = nowTime();
    $("clockDate").textContent = d.getFullYear() + "/" +
      String(d.getMonth() + 1).padStart(2, "0") + "/" +
      String(d.getDate()).padStart(2, "0");
  }

  /* ------------------------------------------------------------------ */
  /* 健康状态 / 模型 / 板卡                                              */
  /* ------------------------------------------------------------------ */
  function setChip(id, text, kind) {
    var el = $(id);
    el.className = "status-chip " + (kind || "");
    el.querySelector("span").textContent = text;
  }

  function applyHealth(h) {
    state.health = h;
    state.model = h.model || {};
    setChip("serverChip", "服务在线", "ok");

    var m = h.model || {};
    if (m.available) {
      setChip("modelChip", "模型就绪", "ok");
    } else {
      setChip("modelChip", "模型未就绪", "warn");
    }
    var sys = h.system || {};
    if (sys.npu_available) {
      setChip("npuChip", "NPU 可用", "ok");
    } else if (sys.is_atlas) {
      setChip("npuChip", "Atlas / CPU", "warn");
    } else {
      setChip("npuChip", "CPU 调试模式", "warn");
    }

    // 模型卡片
    var hero = $("modelHero");
    hero.classList.toggle("offline", !m.available);
    $("modelName").textContent = m.model || (m.available ? "YOLOv8n" : "模型未就绪");
    $("modelFile").textContent = m.available
      ? "输入 " + (m.input_size || "--") + " · " + (m.labels || []).join(" / ")
      : "把训练后的 .om / .onnx 放入 models/";
    $("modelState").textContent = m.available
      ? (String(m.backend || "").toUpperCase() + " · READY")
      : "MODEL MISSING";
    $("infoBackend").textContent = m.backend || "--";
    $("infoDevice").textContent = m.device || "--";
    $("infoInput").textContent = m.input_size ? m.input_size + " × " + m.input_size : "--";
    $("infoConf").textContent = m.conf_threshold != null ? Number(m.conf_threshold).toFixed(2) : "--";
    var alarmCfg = h.alarm || {};
    var thrMap = alarmCfg.thresholds || {};
    var thrKeys = Object.keys(thrMap);
    $("infoAlarm").textContent = thrKeys.length
      ? thrKeys.map(function (k) { return cn(k) + " " + Number(thrMap[k]).toFixed(2); }).join(" / ")
      : (m.alarm_threshold != null ? Number(m.alarm_threshold).toFixed(2) : "--");
    var cfMap = alarmCfg.confirm_frames || {};
    $("infoConfirm").textContent = cfMap.smoke ? ("烟雾连续 " + cfMap.smoke + " 帧") : "--";

    // 模型缺失遮罩
    var overlay = $("modelOverlay");
    overlay.classList.toggle("show", !m.available);
    if (!m.available) {
      $("modelOverlayText").textContent = m.error ||
        "请先训练 YOLOv8 并导出 ONNX，再拷贝到板端转换为 OM。数据无需上板。";
    }

    // 板卡
    $("boardName").textContent = sys.platform || "未知设备";
    $("boardArch").textContent = sys.arch || "--";
    $("boardCann").textContent = sys.cann_version || (sys.is_atlas ? "未读取" : "PC 无 CANN");
    if (sys.memory) {
      $("boardMem").textContent = fmtMB(sys.memory.used_mb) + " / " + fmtMB(sys.memory.total_mb);
    } else {
      $("boardMem").textContent = "--";
    }
    $("boardChip").textContent = sys.npu_available ? "ASCEND NPU" : (sys.is_atlas ? "ATLAS / CPU" : "HOST CPU");
    $("boardChip").className = "board-chip" + (sys.npu_available ? "" : " cpu");
    $("npuOutput").textContent = sys.npu_output || (sys.npu_available
      ? "npu-smi info 无输出" : "当前为 PC 调试环境，未检测到 npu-smi");

    // 摄像头
    var cam = h.camera || {};
    $("camLabel").textContent = (cam.label || "CAMERA") + " · YOLOv8";
  }

  function pollHealth() {
    fetch("/api/health", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(applyHealth)
      .catch(function () {
        setChip("serverChip", "服务离线", "bad");
        setChip("modelChip", "未知", "bad");
        setChip("npuChip", "未知", "bad");
      });
  }

  function refreshBoard() {
    fetch("/api/board", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.system) {
          state.health.system = j.system;
          applyHealth(state.health);
          showToast("板端信息已刷新", "ok");
        }
      })
      .catch(function () { showToast("刷新失败", "error"); });
  }

  /* ------------------------------------------------------------------ */
  /* SSE                                                                 */
  /* ------------------------------------------------------------------ */
  function connectStream() {
    if (state.es) state.es.close();
    state.es = new EventSource("/api/stream?seq=0");
    state.es.onopen = function () {
      $("streamState").textContent = "推理流已连接";
      setChip("serverChip", "服务在线", "ok");
      log("实时推理流已连接", "safe");
    };
    state.es.onerror = function () {
      $("streamState").textContent = "连接中断，自动重连…";
      setChip("serverChip", "重连中", "warn");
    };
    state.es.onmessage = function (ev) {
      var data;
      try { data = JSON.parse(ev.data); } catch (e) { return; }
      handleFrame(data);
    };
  }

  function handleFrame(frame) {
    if (frame.image) {
      var img = $("liveImage");
      img.src = "data:image/jpeg;base64," + frame.image;
      if (!img.classList.contains("ready")) {
        img.classList.add("ready");
        $("videoPlaceholder").classList.add("hidden");
      }
      if (frame.frame) {
        $("resolutionTag").textContent = frame.frame.width + " × " + frame.frame.height;
      }
    }

    var dets = frame.detections || [];
    var count = dets.length;
    var score = frame.max_score || 0;
    var stats = frame.stats || {};
    var alarm = frame.alarm || {};
    var model = frame.model || {};

    if (model.available === false) {
      $("modelOverlay").classList.add("show");
      setChip("modelChip", "模型未就绪", "warn");
    } else if (model.available) {
      $("modelOverlay").classList.remove("show");
    }

    $("fpsTag").textContent = Number(stats.fps || 0).toFixed(1) + " FPS";

    // 分类计数（直接用检测框统计，兼容后端未下发 counts 的情况）
    var nFire = 0, nSmoke = 0;
    dets.forEach(function (d) {
      var l = String(d.label || "").toLowerCase();
      if (l === "smoke") nSmoke++;
      else nFire++;
    });

    var level = alarm.level;
    if (level == null) level = alarm.active ? 2 : 0;
    var levelKey = alarm.level_key || (level === 2 ? "fire" : level === 1 ? "warn" : "safe");
    var levelName = alarm.level_name || (level === 2 ? "火情告警" : level === 1 ? "烟雾预警" : "安全");

    $("metricFire").textContent = nFire;
    $("metricFire").className = nFire ? "hot" : "";
    $("metricSmoke").textContent = nSmoke;
    $("metricSmoke").className = nSmoke ? "smoke" : "";
    $("metricScore").textContent = count ? (score * 100).toFixed(1) + "%" : "--";
    $("metricScore").className = count ? "hot" : "";
    $("metricLatency").innerHTML = (stats.infer_ms != null ? stats.infer_ms : "--") + "<em>ms</em>";
    $("metricAlarm").textContent = levelName;
    $("metricAlarm").className = levelKey === "safe" ? "safe" : levelKey;
    $("metricAlarmSub").textContent = level === 2 ? "FIRE ALARM"
      : level === 1 ? "SMOKE WARNING" : "NO ALARM";

    var fireLevel = level === 2 || nFire > 0;
    var warnLevel = !fireLevel && (level === 1 || nSmoke > 0);
    $("videoWrap").classList.toggle("alarm", fireLevel);
    $("videoWrap").classList.toggle("warn", warnLevel);

    var parts = [];
    if (nFire) parts.push("火焰 " + nFire + " 处");
    if (nSmoke) parts.push("烟雾 " + nSmoke + " 处");
    $("detectState").textContent = count
      ? parts.join(" · ") + " · 最高 " + (score * 100).toFixed(1) + "%"
      : (model.available ? "画面未见明火 / 烟雾 · 持续监测" : "模型未就绪");
    $("streamState").textContent = "持续推理中 · #" + frame.seq;

    var sig = nFire + "/" + nSmoke;
    if (state.lastCount !== sig) {
      if (count) {
        log("检测到 " + parts.join("、") + "，最高置信度 " + (score * 100).toFixed(1) + "%",
          nFire ? "hot" : "warn");
      } else if (state.lastCount && state.lastCount !== "0/0") {
        log("画面目标消失，恢复监测", "safe");
      }
      state.lastCount = sig;
    }

    if (alarm.new && alarm.id && alarm.id !== state.lastAlarmId) {
      state.lastAlarmId = alarm.id;
      recordAlarm(alarm);
      fetchEvents();
    }
  }

  /* ------------------------------------------------------------------ */
  /* 告警记录：只写入右侧面板，不弹窗、不发声                             */
  /* ------------------------------------------------------------------ */
  function recordAlarm(alarm) {
    var level = alarm.level === 1 ? 1 : 2;
    var counts = alarm.counts || {};
    var detail = [];
    if (counts.fire) detail.push("火焰 " + counts.fire + " 处");
    if (counts.smoke) detail.push("烟雾 " + counts.smoke + " 处");
    var score = (Number(alarm.score || 0) * 100).toFixed(1) + "%";
    log(levelName(level) + " · " + (detail.length ? detail.join("、") : "目标已确认") +
      " · 最高 " + score + " · 证据已存入右侧列表", LEVEL_KEY[level]);
  }

  function clearAlarm(silent) {
    fetch("/api/alarm/clear", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function () {
        $("videoWrap").classList.remove("alarm");
        $("videoWrap").classList.remove("warn");
        $("metricAlarm").textContent = "安全";
        $("metricAlarm").className = "safe";
        $("metricAlarmSub").textContent = "NO ALARM";
        state.lastAlarmId = null;
        if (!silent) {
          showToast("告警已清除，继续监测", "ok");
          log("告警已清除", "safe");
        }
      })
      .catch(function () { showToast("清除失败", "error"); });
  }

  /* ------------------------------------------------------------------ */
  /* 证据                                                                */
  /* ------------------------------------------------------------------ */
  function fetchEvents() {
    fetch("/api/events", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        state.events = (j && j.events) || [];
        renderEvents();
      })
      .catch(function () { });
  }

  function renderEvents() {
    var list = $("eventList");
    var items = state.events;
    $("eventCount").textContent = items.length;
    if (!items.length) {
      list.innerHTML =
        '<div class="empty-state" id="eventEmpty"><div class="empty-icon">◌</div>' +
        '<p>暂无检测记录</p><small>火焰告警或手动截图会出现在这里</small></div>';
      return;
    }
    list.innerHTML = items.map(function (it) {
      var isManual = it.label === "手动截图";
      var score = isManual
        ? "手动截图"
        : ((it.label ? cn(it.label) + " " : "") +
          (Number(it.score) * 100).toFixed(1) + "%");
      var isNew = it.id && it.id === state.lastAlarmId;
      var cls = it.level === 1 ? " warn" : (it.level === 2 ? " fire" : "");
      return '<div class="event-item' + cls + '" data-src="' + esc(it.image) + '" data-time="' +
        esc(it.time) + '" data-score="' + esc(score) + '">' +
        '<img loading="lazy" src="' + esc(it.image) + '" alt="检测证据">' +
        (isNew ? '<span class="event-new">NEW</span>' : "") +
        '<div class="event-info"><span>' + esc((it.time || "").split(" ").pop()) +
        '</span><span class="event-score">' + esc(score) + "</span></div></div>";
    }).join("");
    Array.prototype.forEach.call(list.querySelectorAll(".event-item"), function (el) {
      el.addEventListener("click", function () {
        openImage(el.getAttribute("data-src"), el.getAttribute("data-time"),
          el.getAttribute("data-score"));
      });
    });
  }

  function openImage(src, time, meta) {
    $("modalImage").src = src;
    $("modalCaption").textContent = "检测证据";
    $("modalMeta").textContent = (time || "") + (meta ? " · " + meta : "");
    openModal("imageModal");
  }

  function saveSnapshot() {
    fetch("/api/snapshot", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.ok) {
          showToast("当前画面已保存", "ok");
          log("已保存手动截图", "safe");
          fetchEvents();
        } else {
          showToast(j.error || "保存失败", "error");
        }
      })
      .catch(function () { showToast("保存失败", "error"); });
  }

  /* ------------------------------------------------------------------ */
  /* 单图检测                                                            */
  /* ------------------------------------------------------------------ */
  function openDetect() {
    state.file = null;
    $("detectFile").value = "";
    $("detectPreview").classList.remove("show");
    $("detectResult").classList.remove("show");
    $("detectSummary").textContent = "尚未选择图片";
    $("runDetect").disabled = true;
    openModal("detectModal");
  }

  function onFileChange() {
    var f = $("detectFile").files[0];
    if (!f) return;
    state.file = f;
    var url = URL.createObjectURL(f);
    $("detectPreview").src = url;
    $("detectPreview").classList.add("show");
    $("detectSummary").textContent = f.name + " · " + (f.size / 1024).toFixed(1) + " KB";
    $("runDetect").disabled = false;
  }

  function runDetect() {
    if (!state.file) return;
    var fd = new FormData();
    fd.append("file", state.file, state.file.name);
    $("runDetect").disabled = true;
    $("runDetect").textContent = "推理中…";
    fetch("/api/detect", { method: "POST", body: fd })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (!j.ok) {
          showToast(j.error || "检测失败", "error");
          $("detectSummary").textContent = "检测失败：" + (j.error || "未知错误");
          return;
        }
        var dets = j.detections || [];
        $("detectResultImage").src = "data:image/jpeg;base64," + j.image;
        $("detectResult").classList.add("show");
        $("detectList").innerHTML = dets.map(function (d) {
          return '<span class="detect-pill">' + esc(cn(d.label)) + " " +
            (Number(d.score) * 100).toFixed(1) + "%</span>";
        }).join("") || '<span class="detect-pill">未检测到目标</span>';
        var nF = 0, nS = 0;
        dets.forEach(function (d) {
          if (String(d.label || "").toLowerCase() === "smoke") nS++; else nF++;
        });
        $("detectSummary").innerHTML = "火焰 <b>" + nF + "</b> · 烟雾 <b>" + nS +
          "</b> · 最高 <b>" + (Number(j.max_score || 0) * 100).toFixed(1) + "%</b> · 推理 <b>" +
          (j.stats.infer_ms || 0).toFixed(1) + " ms</b>";
        log("单图检测：火焰 " + nF + " / 烟雾 " + nS, (nF || nS) ? "hot" : "safe");
      })
      .catch(function () {
        showToast("接口请求失败", "error");
        $("detectSummary").textContent = "接口请求失败";
      })
      .finally(function () {
        $("runDetect").disabled = false;
        $("runDetect").textContent = "开始检测";
      });
  }

  /* ------------------------------------------------------------------ */
  /* 弹窗 / 初始化                                                       */
  /* ------------------------------------------------------------------ */
  function openModal(id) {
    var el = $(id);
    if (!el) return;
    el.classList.add("show");
    el.setAttribute("aria-hidden", "false");
  }

  function closeModal(id) {
    var el = $(id);
    if (!el) return;
    el.classList.remove("show");
    el.setAttribute("aria-hidden", "true");
  }

  function bindModals() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-close]"), function (el) {
      el.addEventListener("click", function () { closeModal(el.getAttribute("data-close")); });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        closeModal("helpModal");
        closeModal("imageModal");
        closeModal("detectModal");
      }
    });
  }

  function init() {
    tickClock();
    setInterval(tickClock, 1000);
    bindModals();

    $("uploadNow").addEventListener("click", openDetect);
    $("saveSnapshot").addEventListener("click", saveSnapshot);
    $("clearAlarm").addEventListener("click", function () { clearAlarm(false); });
    $("openHelp").addEventListener("click", function () { openModal("helpModal"); });
    $("overlayHelp").addEventListener("click", function () { openModal("helpModal"); });
    $("refreshBoard").addEventListener("click", refreshBoard);
    $("detectFile").addEventListener("change", onFileChange);
    $("runDetect").addEventListener("click", runDetect);
    $("clearLog").addEventListener("click", function () {
      $("logList").innerHTML = '<div class="log-item"><time>--:--:--</time><span>日志已清空</span></div>';
    });

    log("系统启动，加载模型与视频流…", "safe");
    pollHealth();
    fetchEvents();
    connectStream();
    setInterval(pollHealth, 5000);
  }

  init();
})();
