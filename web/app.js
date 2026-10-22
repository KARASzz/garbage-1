(() => {
  const drawButton = document.querySelector("#drawButton");
  const aiButton = document.querySelector("#aiButton");
  const likeButton = document.querySelector("#likeButton");
  const dislikeButton = document.querySelector("#dislikeButton");
  const message = document.querySelector("#message");
  const screenStatus = document.querySelector("#screenStatus");
  const feedbackPanel = document.querySelector("#feedbackPanel");
  const correctionHint = document.querySelector("#correctionHint");
  const correctionButton = document.querySelector("#correctionButton");
  const restoreButton = document.querySelector("#restoreButton");
  const feedbackStatus = document.querySelector("#feedbackStatus");
  const modelStatus = document.querySelector("#modelStatus");
  const feedForm = document.querySelector("#feedForm");
  const feedInput = document.querySelector("#feedInput");
  const feedButton = document.querySelector("#feedButton");
  const feedStatus = document.querySelector("#feedStatus");
  const styleSelect = document.querySelector("#styleSelect");
  const syncButton = document.querySelector("#syncButton");
  const syncStatus = document.querySelector("#syncStatus");
  const styleStatus = document.querySelector("#styleStatus");
  const styleHelp = document.querySelector("#styleHelp");

  if (
    !drawButton ||
    !aiButton ||
    !likeButton ||
    !dislikeButton ||
    !message ||
    !screenStatus ||
    !feedbackPanel ||
    !correctionHint ||
    !correctionButton ||
    !restoreButton ||
    !feedbackStatus ||
    !modelStatus ||
    !feedForm ||
    !feedInput ||
    !feedButton ||
    !feedStatus ||
    !styleSelect ||
    !syncButton ||
    !syncStatus ||
    !styleStatus ||
    !styleHelp
  ) {
    return;
  }

  const feedbackButtons = [likeButton, dislikeButton];
  const timeoutMs = 30_000;
  const maxCorrectionLength = 800;
  const pendingFeedbackSamples = new Set();
  const pendingCorrectionSamples = new Set();
  const styleDefinitions = [
    { id: "base", name: "原版", english: "Base" },
    { id: "mouthy", name: "嘴臭", english: "Mouthy" },
    { id: "mad", name: "疯子", english: "Mad" },
    { id: "sassy", name: "娘炮", english: "Sassy" },
    { id: "anime", name: "二次元", english: "Anime" },
    { id: "femboy", name: "伪娘", english: "Femboy" },
    { id: "office", name: "社畜", english: "Office" },
    { id: "soldier", name: "兵王", english: "Soldier" },
    { id: "alien", name: "外星人", english: "Alien" },
    { id: "hostess", name: "陪酒女", english: "Hostess" },
  ];
  const styleRecords = new Map(
    styleDefinitions.map((style) => [
      style.id,
      {
        ...style,
        ready: style.id === "base",
        step: null,
        parameters: null,
        stateKnown: false,
      },
    ]),
  );
  let styleCatalogLoaded = false;
  let styleLoadError = "";
  let isGenerating = false;
  let isFeeding = false;
  let activeSampleId = null;
  let activeOriginalText = "";
  let activeOriginalFeedbackStatus = "";
  let activeRating = 0;
  let syncSubmitPending = false;
  let syncInFlight = false;
  let syncPollTimer = null;
  let syncRequestToken = 0;
  let latestBaseStep = null;

  function currentStyleRecord() {
    return styleRecords.get(styleSelect.value) || styleRecords.get("base");
  }

  function normalizeVisibleText(text) {
    return String(text || "").replace(/\r\n?/g, "\n").trim();
  }

  function visibleMessageText() {
    return typeof message.innerText === "string" ? message.innerText : message.textContent || "";
  }

  function isActiveOutputEdited() {
    return activeSampleId !== null && activeOriginalText !== "" &&
      normalizeVisibleText(visibleMessageText()) !== normalizeVisibleText(activeOriginalText);
  }

  function updateMessageEditorState(activeSamplePending) {
    const hasOutput = activeSampleId !== null && activeOriginalText !== "";
    const canEdit = hasOutput && !activeSamplePending;
    message.contentEditable = canEdit ? "plaintext-only" : "false";

    if (hasOutput) {
      message.setAttribute("role", "textbox");
      message.setAttribute("aria-label", "生成结果，可直接编辑");
      message.setAttribute("aria-multiline", "true");
      message.setAttribute("aria-live", "off");
      message.setAttribute("aria-readonly", String(!canEdit));
      message.setAttribute("aria-describedby", "correctionHint");
      message.tabIndex = 0;
    } else {
      message.setAttribute("role", "status");
      message.setAttribute("aria-live", "polite");
      message.setAttribute("aria-atomic", "true");
      message.removeAttribute("aria-label");
      message.removeAttribute("aria-multiline");
      message.removeAttribute("aria-readonly");
      message.removeAttribute("aria-describedby");
      message.removeAttribute("tabindex");
    }
  }

  function updateButtonState() {
    const selectedStyleReady = currentStyleRecord()?.ready === true;
    drawButton.disabled = isGenerating;
    aiButton.disabled = isGenerating || !selectedStyleReady;
    syncButton.disabled = syncSubmitPending || syncInFlight;

    const activeSamplePending =
      isGenerating || (activeSampleId !== null && (
        pendingFeedbackSamples.has(activeSampleId) || pendingCorrectionSamples.has(activeSampleId)
      ));
    const outputEdited = isActiveOutputEdited();
    feedbackButtons.forEach((button) => {
      button.disabled = activeSamplePending || outputEdited || activeSampleId === null;
    });
    correctionButton.disabled = activeSamplePending || activeSampleId === null;
    restoreButton.hidden = !outputEdited;
    restoreButton.disabled = activeSamplePending;
    updateMessageEditorState(activeSamplePending);
    feedButton.disabled = isFeeding || !selectedStyleReady;
    feedInput.disabled = isFeeding || !selectedStyleReady;
  }

  function renderStyleOptions() {
    Array.from(styleSelect.options).forEach((option) => {
      const record = styleRecords.get(option.value);
      if (!record) return;
      const state = record.stateKnown ? (record.ready ? "就绪" : "未就绪") : "状态未知";
      option.textContent = `${record.name} · ${record.english} · ${state}`;
    });
  }

  function styleTrainingDetails(style) {
    const details = [];
    if (Number.isSafeInteger(style.step) && style.step >= 0) details.push(`step ${style.step}`);
    if (style.id !== "base" && Number.isSafeInteger(style.base_step)) details.push(`底座 step ${style.base_step}`);
    if (Number.isSafeInteger(style.parameters) && style.parameters > 0) {
      const size = style.parameters >= 1_000_000
        ? `${Number((style.parameters / 1_000_000).toFixed(1))}M`
        : `${Math.round(style.parameters / 1_000)}K`;
      details.push(`${size} 参数`);
    }
    return details.length ? ` · ${details.join(" · ")}` : "";
  }

  function updateStylePresentation() {
    const style = currentStyleRecord();
    const isBase = style.id === "base";
    const target = isBase ? "原版底座" : `「${style.name}」LoRA`;

    if (styleLoadError) {
      styleStatus.textContent = isBase
        ? `风格状态接口读取失败：${styleLoadError}。原版入口仍可尝试；其他风格状态未知。`
        : `风格状态接口读取失败：${styleLoadError}。当前风格状态未知，AI 与投喂已停用。`;
    } else if (!styleCatalogLoaded) {
      styleStatus.textContent = isBase
        ? "正在读取风格状态；原版入口仍可用。"
        : "正在读取风格状态；当前 LoRA 状态未知，AI 与投喂暂不可用。";
    } else if (style.ready) {
      styleStatus.textContent = `当前投喂目标：${target} · 就绪${styleTrainingDetails(style)}`;
    } else {
      styleStatus.textContent = `当前${isBase ? "底座" : "LoRA 风格"}「${style.name}」未就绪，AI 与投喂已停用。`;
    }

    styleHelp.textContent = isBase
      ? "选原版时投喂到底座训练；选 LoRA 风格后只投喂到对应适配器。"
      : style.ready
        ? `投喂整句只归入「${style.name}」LoRA，帮助它独立学习。`
        : `该风格未就绪，投喂暂停；就绪后整句会只归入「${style.name}」LoRA。`;
    styleHelp.textContent += " 同步会适配全部 9 个 LoRA，目标为点击时的最新底座。";
    updateButtonState();
  }

  function applyStyleCatalog(payload) {
    if (!payload || payload.default !== "base" || !Array.isArray(payload.styles)) {
      throw new Error("风格列表格式不完整。");
    }

    const received = new Map();
    for (const item of payload.styles) {
      if (!item || typeof item.id !== "string") continue;
      const definition = styleDefinitions.find((style) => style.id === item.id);
      if (!definition) continue;
      if (received.has(item.id) || typeof item.name !== "string" || !item.name.trim() || typeof item.ready !== "boolean") {
        throw new Error("风格列表格式不完整。");
      }
      received.set(item.id, {
        ...definition,
        name: item.name.trim(),
        ready: item.ready,
        step: Number.isSafeInteger(item.step) ? item.step : null,
        parameters: Number.isSafeInteger(item.parameters) ? item.parameters : null,
        base_step: Number.isSafeInteger(item.base_step) ? item.base_step : null,
        stateKnown: true,
      });
    }

    if (!received.has("base")) throw new Error("风格列表没有原版底座状态。");
    if (received.size !== styleDefinitions.length) throw new Error("风格列表缺少部分风格状态。");
    styleDefinitions.forEach((definition) => {
      styleRecords.set(
        definition.id,
        received.get(definition.id) || {
          ...definition,
          ready: false,
          step: null,
          parameters: null,
          stateKnown: false,
        },
      );
    });
    styleCatalogLoaded = true;
    styleLoadError = "";
    renderStyleOptions();
    updateStylePresentation();
  }

  function readableError(error, fallback) {
    return error instanceof Error && error.message ? error.message : fallback;
  }

  async function requestJson(url, method, body) {
    const controller = new AbortController();
    let timedOut = false;
    const timeoutId = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    try {
      let response;
      try {
        response = await fetch(url, {
          method,
          headers: {
            Accept: "application/json",
            ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          },
          cache: "no-store",
          signal: controller.signal,
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
      } catch (error) {
        if (timedOut || error?.name === "AbortError") {
          throw new Error("请求超时，请重试。");
        }
        throw new Error("连接失败，请检查网络后重试。");
      }

      let payload;
      try {
        payload = await response.json();
      } catch {
        if (timedOut) throw new Error("请求超时，请重试。");
        throw new Error("服务器响应无法读取，请重试。");
      }

      if (!response.ok) {
        const serverMessage = payload && typeof payload.error === "string" ? payload.error.trim() : "";
        throw new Error(serverMessage || `请求失败（${response.status}），请重试。`);
      }

      return payload;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  function isInteger(value) {
    return Number.isInteger(value);
  }

  const activeSyncStatuses = new Set(["queued", "running"]);
  const terminalSyncStatuses = new Set(["completed", "partial", "failed", "interrupted"]);
  const syncResultStatuses = new Set(["pending", "running", "synced", "skipped", "failed"]);

  function validateSyncJob(job) {
    if (
      !job ||
      typeof job.id !== "string" ||
      !job.id.trim() ||
      !(activeSyncStatuses.has(job.status) || terminalSyncStatuses.has(job.status)) ||
      !(job.target_step === null || (Number.isSafeInteger(job.target_step) && job.target_step >= 0)) ||
      !Number.isSafeInteger(job.completed) ||
      job.completed < 0 ||
      !Number.isSafeInteger(job.total) ||
      job.total < 0 ||
      job.completed > job.total ||
      !Array.isArray(job.results) ||
      !(job.message === undefined || typeof job.message === "string")
    ) {
      throw new Error("同步任务状态格式不完整。");
    }

    for (const result of job.results) {
      if (
        !result ||
        typeof result.style !== "string" ||
        !result.style.trim() ||
        typeof result.name !== "string" ||
        !result.name.trim() ||
        !syncResultStatuses.has(result.status) ||
        !(result.message === undefined || typeof result.message === "string")
      ) {
        throw new Error("同步任务结果格式不完整。");
      }
    }
    return job;
  }

  function validateSyncSnapshot(payload) {
    if (
      !payload ||
      !Object.prototype.hasOwnProperty.call(payload, "job") ||
      !(payload.job === null || typeof payload.job === "object") ||
      !(payload.latest_base_step === null || (
        Number.isSafeInteger(payload.latest_base_step) && payload.latest_base_step >= 0
      ))
    ) {
      throw new Error("同步状态响应格式不完整。");
    }
    return payload;
  }

  function stopSyncPolling() {
    if (syncPollTimer !== null) {
      window.clearTimeout(syncPollTimer);
      syncPollTimer = null;
    }
  }

  function syncResultLabel(status) {
    return {
      pending: "等待中",
      running: "迁移中",
      synced: "已同步",
      skipped: "无需更新",
      failed: "失败",
    }[status] || status;
  }

  function renderSyncJob(job) {
    const counts = Object.fromEntries(
      ["synced", "skipped", "failed", "running", "pending"].map((status) => [
        status,
        job.results.filter((result) => result.status === status).length,
      ]),
    );
    const target = job.target_step === null ? "目标底座 step 未知" : `点击时目标底座 step ${job.target_step}`;
    const progress = `${job.completed}/${job.total}`;
    let summary;

    switch (job.status) {
      case "queued":
        summary = `同步任务已排队（${progress}）`;
        break;
      case "running":
        summary = `正在后台同步全部 LoRA（${progress}；已同步 ${counts.synced}，跳过 ${counts.skipped}，失败 ${counts.failed}）`;
        break;
      case "completed":
        summary = counts.synced === 0 && counts.failed === 0
          ? `检查完成，没有需要切换的 LoRA（${progress}）`
          : `同步完成（${progress}；已同步 ${counts.synced}，跳过 ${counts.skipped}，失败 ${counts.failed}）`;
        break;
      case "partial":
        summary = `同步部分完成（${progress}；已同步 ${counts.synced}，跳过 ${counts.skipped}，失败 ${counts.failed}）`;
        break;
      case "failed":
        summary = `同步失败（${progress}；已同步 ${counts.synced}，失败 ${counts.failed}）`;
        break;
      case "interrupted":
        summary = `同步任务已中断（${progress}；已同步 ${counts.synced}，失败 ${counts.failed}）`;
        break;
      default:
        summary = `同步状态：${job.status}（${progress}）`;
    }

    const details = job.results
      .filter((result) => result.status === "failed" || (
        (result.status === "skipped" || result.status === "running") && result.message?.trim()
      ))
      .map((result) => {
        const note = typeof result.message === "string" ? result.message.trim() : "";
        return `${result.name}：${syncResultLabel(result.status)}${note ? `（${note}）` : ""}`;
      });
    const notes = [summary, target, job.message?.trim(), ...details].filter(Boolean);
    if (Number.isSafeInteger(latestBaseStep) && Number.isSafeInteger(job.target_step) && latestBaseStep > job.target_step) {
      notes.push(`当前底座已到 step ${latestBaseStep}，可在本次结束后再次同步`);
    }
    syncStatus.textContent = notes.join("；");
  }

  function scheduleSyncPoll(token) {
    stopSyncPolling();
    syncPollTimer = window.setTimeout(() => {
      syncPollTimer = null;
      void loadSyncSnapshot(token);
    }, 2_000);
  }

  function applySyncJob(job, token) {
    const validJob = validateSyncJob(job);
    syncInFlight = activeSyncStatuses.has(validJob.status);
    renderSyncJob(validJob);
    updateButtonState();

    if (syncInFlight) {
      scheduleSyncPoll(token);
    } else {
      stopSyncPolling();
      void loadStyles();
      void loadModelStatus();
    }
  }

  async function loadSyncSnapshot(token) {
    try {
      const payload = validateSyncSnapshot(await requestJson("/api/lora/sync", "GET"));
      if (token !== syncRequestToken) return;
      latestBaseStep = payload.latest_base_step;

      if (payload.job === null) {
        const wasActive = syncInFlight;
        syncInFlight = false;
        stopSyncPolling();
        syncStatus.textContent = wasActive
          ? "后台未返回同步任务记录，无法确认同步结果。已刷新当前风格状态。"
          : latestBaseStep === null
            ? "目前没有同步任务；暂时无法读取最新底座 step。"
            : `目前没有同步任务；最新底座 step ${latestBaseStep}。`;
        updateButtonState();
        if (wasActive) {
          void loadStyles();
          void loadModelStatus();
        }
        return;
      }

      applySyncJob(payload.job, token);
    } catch (error) {
      if (token !== syncRequestToken) return;
      const detail = readableError(error, "无法读取同步状态");
      syncStatus.textContent = syncInFlight
        ? `同步状态暂时无法读取：${detail}；任务仍在后台，2 秒后重试。`
        : `无法读取同步状态：${detail}。`;
      updateButtonState();
      if (syncInFlight) scheduleSyncPoll(token);
    }
  }

  async function startLoraSync() {
    if (syncSubmitPending || syncInFlight) return;
    stopSyncPolling();
    const token = ++syncRequestToken;
    syncSubmitPending = true;
    syncStatus.textContent = "正在提交全部 9 个 LoRA 的同步任务…";
    updateButtonState();

    try {
      const payload = await requestJson("/api/lora/sync", "POST", {});
      if (token !== syncRequestToken) return;
      if (!payload || !payload.job) throw new Error("服务器没有返回同步任务状态。");
      applySyncJob(payload.job, token);
    } catch (error) {
      if (token !== syncRequestToken) return;
      const detail = readableError(error, "提交同步任务失败");
      syncStatus.textContent = `未能确认同步任务是否已提交：${detail}。正在检查后台状态…`;
      await loadSyncSnapshot(token);
    } finally {
      if (token === syncRequestToken) {
        syncSubmitPending = false;
        updateButtonState();
      }
    }
  }

  function restoreSyncStatus() {
    const token = ++syncRequestToken;
    void loadSyncSnapshot(token);
  }

  function validateModel(model) {
    return Boolean(
      model &&
        model.name === "garbage-1" &&
        Number.isSafeInteger(model.parameters) &&
        model.parameters > 0 &&
        model.dtype === "float32" &&
        model.temperature === 0,
    );
  }

  function validateGeneration(result, source, requestedStyle) {
    return Boolean(
      result &&
        typeof result.text === "string" &&
        result.text.trim() !== "" &&
        typeof result.id === "string" &&
        typeof result.theme === "string" &&
        isInteger(result.block_count) &&
        typeof result.sample_id === "string" &&
        result.sample_id.trim() !== "" &&
        result.source === source &&
        result.rating === 0 &&
        validateModel(result.model) &&
        (source !== "garbage-1" || (
          result.style === requestedStyle &&
          typeof result.style_name === "string" &&
          result.style_name.trim() !== ""
        )) &&
        (source === "garbage-1" ? result.block_count === 0 : result.block_count >= 0),
    );
  }

  function paintRating(rating) {
    likeButton.setAttribute("aria-pressed", String(rating === 1));
    dislikeButton.setAttribute("aria-pressed", String(rating === -1));
  }

  function applyModel(model) {
    if (!validateModel(model)) return;
    const backend = { mlx: "MLX", cuda: "CUDA", "torch-cpu": "CPU" }[model.backend] || "MLX";
    const size = model.parameters >= 1_000_000
      ? `${Number((model.parameters / 1_000_000).toFixed(1))}M`
      : `${Math.round(model.parameters / 1_000)}K`;
    modelStatus.textContent = `${model.name.toUpperCase()} · ${size} · FP32 · ZH · ${backend}`;
  }

  function installOutput(result) {
    activeOriginalText = result.text.trim();
    message.textContent = activeOriginalText;
    activeSampleId = result.sample_id;
    activeRating = result.rating;
    activeOriginalFeedbackStatus = typeof result.style_name === "string" && result.style_name.trim()
      ? `本句归属风格：${result.style_name}`
      : "";
    feedbackStatus.textContent = activeOriginalFeedbackStatus;
    correctionHint.textContent = "直接修改屏幕上的句子，再提交改写";
    paintRating(activeRating);
    feedbackPanel.hidden = false;
    applyModel(result.model);
    updateButtonState();
  }

  async function generate(source) {
    if (isGenerating) return;
    const requestedStyle = source === "garbage-1" ? styleSelect.value : null;
    const requestedStyleRecord = requestedStyle ? styleRecords.get(requestedStyle) : null;
    if (source === "garbage-1" && requestedStyleRecord?.ready !== true) {
      screenStatus.textContent = "STYLE OFFLINE · 风格未就绪";
      message.textContent = `「${requestedStyleRecord?.name || "当前"}」风格未就绪；可切换到原版或使用规则抽卡。`;
      return;
    }

    isGenerating = true;
    updateButtonState();
    screenStatus.textContent = source === "rule" ? "规则抽卡中…" : "GARBAGE-1 · 正在生成…";

    try {
      const endpoint = source === "rule" ? "/api/draw" : "/api/ai";
      const body = source === "rule" ? {} : { style: requestedStyle };
      const result = await requestJson(endpoint, "POST", body);

      if (!validateGeneration(result, source, requestedStyle)) {
        throw new Error("生成结果格式不完整，请重试。");
      }

      installOutput(result);
      screenStatus.textContent = source === "rule"
        ? "规则抽卡 · 完成"
        : `${result.style_name} · GARBAGE-1 · 贪心输出`;
    } catch (error) {
      screenStatus.textContent = "SIGNAL LOST · 再试一次";
      if (!activeSampleId) {
        message.textContent = readableError(error, "这次没接上。再按一次，终端还在。");
      } else {
        feedbackStatus.textContent = readableError(error, "生成失败，可以重试。");
      }
    } finally {
      isGenerating = false;
      updateButtonState();
    }
  }

  async function rateOutput(rating) {
    if (isGenerating || !activeSampleId || isActiveOutputEdited() || rating === activeRating) return;

    const sampleId = activeSampleId;
    const previousRating = activeRating;
    if (pendingFeedbackSamples.has(sampleId)) return;

    pendingFeedbackSamples.add(sampleId);
    activeRating = rating;
    paintRating(activeRating);
    feedbackStatus.textContent = rating === 1 ? "已点赞" : "已点踩";
    screenStatus.textContent = "评价已登记";
    updateButtonState();

    try {
      const result = await requestJson("/api/feedback", "POST", {
        sample_id: sampleId,
        rating,
      });

      if (
        !result ||
        result.sample_id !== sampleId ||
        result.rating !== rating ||
        !result.counts ||
        !isInteger(result.counts.up) ||
        !isInteger(result.counts.down) ||
        !isInteger(result.counts.pending)
      ) {
        throw new Error("评价结果格式不完整，请重试。");
      }

      if (activeSampleId === sampleId) {
        feedbackStatus.textContent = rating === 1 ? "已点赞" : "已点踩";
        screenStatus.textContent = "已保存";
      }
    } catch (error) {
      if (activeSampleId === sampleId) {
        activeRating = previousRating;
        paintRating(activeRating);
        feedbackStatus.textContent = readableError(error, "保存失败，请重试。");
        screenStatus.textContent = "保存失败 · 可重试";
      }
    } finally {
      pendingFeedbackSamples.delete(sampleId);
      updateButtonState();
    }
  }

  function validateCorrectionResult(result, sampleId) {
    if (
      !result ||
      result.sample_id !== sampleId ||
      typeof result.chosen_sample_id !== "string" ||
      !result.chosen_sample_id.trim() ||
      typeof result.preference_id !== "number" ||
      !Number.isFinite(result.preference_id) ||
      typeof result.duplicate !== "boolean" ||
      typeof result.queued !== "boolean" ||
      typeof result.style !== "string" ||
      !result.style.trim() ||
      typeof result.message !== "string" ||
      !result.counts || typeof result.counts !== "object" || Array.isArray(result.counts)
    ) {
      throw new Error("改写结果格式不完整，内容仍保留，可重试。");
    }
    return result;
  }

  async function submitCorrection() {
    if (isGenerating || !activeSampleId || pendingCorrectionSamples.has(activeSampleId)) return;

    const sampleId = activeSampleId;
    const text = normalizeVisibleText(visibleMessageText());
    if (!text) {
      feedbackStatus.textContent = "改写内容不能为空，请先输入一句话。";
      return;
    }
    if (text === normalizeVisibleText(activeOriginalText)) {
      feedbackStatus.textContent = "请先修改原句，再提交改写。";
      return;
    }
    if (Array.from(text).length > maxCorrectionLength) {
      feedbackStatus.textContent = `改写内容不能超过 ${maxCorrectionLength} 字，请删减后重试。`;
      return;
    }

    pendingCorrectionSamples.add(sampleId);
    feedbackStatus.textContent = "正在保存改写…";
    updateButtonState();

    try {
      const result = validateCorrectionResult(
        await requestJson("/api/correct", "POST", { sample_id: sampleId, text }),
        sampleId,
      );
      if (activeSampleId === sampleId) {
        activeRating = -1;
        paintRating(activeRating);
        activeOriginalFeedbackStatus = "原输出已记为较差答案；改写句已作为优选答案保存。";
        feedbackStatus.textContent = result.training_status === "done"
          ? "这条改写已经完成过偏好训练，没有重复训练。"
          : result.queued ? "已保存，后台偏好学习" : result.message;
      }
      return result;
    } catch (error) {
      if (activeSampleId === sampleId) {
        feedbackStatus.textContent = readableError(error, "改写保存失败，内容仍保留，可重试。");
      }
    } finally {
      pendingCorrectionSamples.delete(sampleId);
      updateButtonState();
    }
  }

  function onMessageInput() {
    if (isActiveOutputEdited()) {
      feedbackStatus.textContent = "当前显示改写稿；点赞和点踩已停用，恢复原文后可评价原输出。";
    } else {
      feedbackStatus.textContent = activeOriginalFeedbackStatus;
    }
    updateButtonState();
  }

  function restoreOriginalOutput() {
    if (isGenerating || !activeSampleId || pendingCorrectionSamples.has(activeSampleId)) return;
    message.textContent = activeOriginalText;
    feedbackStatus.textContent = activeOriginalFeedbackStatus || "原文已恢复，可评价原输出。";
    updateButtonState();
  }

  async function feedSentence(event) {
    event.preventDefault();
    if (isFeeding) return;
    const requestedStyle = styleSelect.value;
    const requestedStyleRecord = styleRecords.get(requestedStyle);
    if (requestedStyleRecord?.ready !== true) {
      feedStatus.textContent = `「${requestedStyleRecord?.name || "当前风格"}」未就绪，暂时不能投喂。`;
      return;
    }
    const text = feedInput.value.trim();
    if (text.length < 2) {
      feedStatus.textContent = "请输入一句完整语料，不要只投喂单个字。";
      feedInput.focus();
      return;
    }

    isFeeding = true;
    feedStatus.textContent = "正在收下这句语料…";
    updateButtonState();
    try {
      const result = await requestJson("/api/feed", "POST", { text, style: requestedStyle });
      if (
        !result ||
        result.style !== requestedStyle ||
        typeof result.sample_id !== "string" ||
        typeof result.duplicate !== "boolean" ||
        !(result.job_id === null || isInteger(result.job_id)) ||
        !isInteger(result.fed_sentences)
      ) {
        throw new Error("投喂结果格式不完整，请重试。");
      }
      feedInput.value = "";
      feedStatus.textContent = result.duplicate
        ? `这句已经在语料库里了（投喂目标：${requestedStyleRecord.name}）。`
        : `整句已收下，正在学习${requestedStyle === "base" ? "原版底座" : `「${requestedStyleRecord.name}」LoRA`}。`;
      screenStatus.textContent = result.duplicate ? "语料已存在" : `${requestedStyleRecord.name} · 整句已投喂 · 后台学习中`;
    } catch (error) {
      feedStatus.textContent = readableError(error, "投喂失败，请重试。");
    } finally {
      isFeeding = false;
      updateButtonState();
    }
  }

  async function loadModelStatus() {
    try {
      const model = await requestJson("/api/model", "GET");
      applyModel(model);
    } catch {
      // The static model label remains visible; generation stays available.
    }
  }

  async function loadStyles() {
    try {
      const payload = await requestJson("/api/styles", "GET");
      applyStyleCatalog(payload);
    } catch (error) {
      styleLoadError = readableError(error, "暂时无法读取风格状态");
      styleCatalogLoaded = false;
      styleDefinitions.forEach((definition) => {
        styleRecords.set(definition.id, {
          ...definition,
          ready: definition.id === "base",
          step: null,
          parameters: null,
          stateKnown: false,
        });
      });
      renderStyleOptions();
      updateStylePresentation();
    }
  }

  drawButton.addEventListener("click", () => generate("rule"));
  aiButton.addEventListener("click", () => generate("garbage-1"));
  likeButton.addEventListener("click", () => rateOutput(1));
  dislikeButton.addEventListener("click", () => rateOutput(-1));
  correctionButton.addEventListener("click", () => { void submitCorrection(); });
  restoreButton.addEventListener("click", restoreOriginalOutput);
  message.addEventListener("input", onMessageInput);
  feedForm.addEventListener("submit", feedSentence);
  styleSelect.addEventListener("change", updateStylePresentation);
  syncButton.addEventListener("click", startLoraSync);

  void loadModelStatus();
  renderStyleOptions();
  updateStylePresentation();
  void loadStyles();
  restoreSyncStatus();
})();
