(() => {
  const drawButton = document.querySelector("#drawButton");
  const aiButton = document.querySelector("#aiButton");
  const likeButton = document.querySelector("#likeButton");
  const dislikeButton = document.querySelector("#dislikeButton");
  const message = document.querySelector("#message");
  const screenStatus = document.querySelector("#screenStatus");
  const feedbackPanel = document.querySelector("#feedbackPanel");
  const feedbackStatus = document.querySelector("#feedbackStatus");
  const modelStatus = document.querySelector("#modelStatus");
  const feedForm = document.querySelector("#feedForm");
  const feedInput = document.querySelector("#feedInput");
  const feedButton = document.querySelector("#feedButton");
  const feedStatus = document.querySelector("#feedStatus");
  const styleSelect = document.querySelector("#styleSelect");
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
    !feedbackStatus ||
    !modelStatus ||
    !feedForm ||
    !feedInput ||
    !feedButton ||
    !feedStatus ||
    !styleSelect ||
    !styleStatus ||
    !styleHelp
  ) {
    return;
  }

  const feedbackButtons = [likeButton, dislikeButton];
  const timeoutMs = 30_000;
  const pendingFeedbackSamples = new Set();
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
  let activeRating = 0;

  function currentStyleRecord() {
    return styleRecords.get(styleSelect.value) || styleRecords.get("base");
  }

  function updateButtonState() {
    const selectedStyleReady = currentStyleRecord()?.ready === true;
    drawButton.disabled = isGenerating;
    aiButton.disabled = isGenerating || !selectedStyleReady;

    const activeSamplePending =
      isGenerating || (activeSampleId !== null && pendingFeedbackSamples.has(activeSampleId));
    feedbackButtons.forEach((button) => {
      button.disabled = activeSamplePending;
    });
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
    message.textContent = result.text.trim();
    activeSampleId = result.sample_id;
    activeRating = result.rating;
    feedbackStatus.textContent = typeof result.style_name === "string" && result.style_name.trim()
      ? `本句归属风格：${result.style_name}`
      : "";
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
    if (isGenerating || !activeSampleId || rating === activeRating) return;

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
  feedForm.addEventListener("submit", feedSentence);
  styleSelect.addEventListener("change", updateStylePresentation);

  void loadModelStatus();
  renderStyleOptions();
  updateStylePresentation();
  void loadStyles();
})();
