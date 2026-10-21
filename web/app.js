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

  if (
    !drawButton ||
    !aiButton ||
    !likeButton ||
    !dislikeButton ||
    !message ||
    !screenStatus ||
    !feedbackPanel ||
    !feedbackStatus ||
    !modelStatus
  ) {
    return;
  }

  const generationButtons = [drawButton, aiButton];
  const feedbackButtons = [likeButton, dislikeButton];
  const timeoutMs = 30_000;
  const pendingFeedbackSamples = new Set();
  let isGenerating = false;
  let activeSampleId = null;
  let activeRating = 0;

  function updateButtonState() {
    generationButtons.forEach((button) => {
      button.disabled = isGenerating;
    });

    const activeSamplePending =
      isGenerating || (activeSampleId !== null && pendingFeedbackSamples.has(activeSampleId));
    feedbackButtons.forEach((button) => {
      button.disabled = activeSamplePending;
    });
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
        model.parameters === 100000 &&
        model.dtype === "float32" &&
        model.temperature === 0,
    );
  }

  function validateGeneration(result, source) {
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
    modelStatus.textContent = `${model.name} · 100K · FP32 · ${backend}`;
  }

  function installOutput(result) {
    message.textContent = result.text.trim();
    activeSampleId = result.sample_id;
    activeRating = result.rating;
    feedbackStatus.textContent = "";
    paintRating(activeRating);
    feedbackPanel.hidden = false;
    applyModel(result.model);
    updateButtonState();
  }

  async function generate(source) {
    if (isGenerating) return;

    isGenerating = true;
    updateButtonState();
    screenStatus.textContent = source === "rule" ? "规则抽卡中…" : "GARBAGE-1 · 正在生成…";

    try {
      const endpoint = source === "rule" ? "/api/draw" : "/api/ai";
      const result = await requestJson(endpoint, "POST", {});

      if (!validateGeneration(result, source)) {
        throw new Error("生成结果格式不完整，请重试。");
      }

      installOutput(result);
      screenStatus.textContent = source === "rule" ? "规则抽卡 · 完成" : "GARBAGE-1 · 贪心输出";
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

  async function loadModelStatus() {
    try {
      const model = await requestJson("/api/model", "GET");
      applyModel(model);
    } catch {
      // The static model label remains visible; generation stays available.
    }
  }

  drawButton.addEventListener("click", () => generate("rule"));
  aiButton.addEventListener("click", () => generate("garbage-1"));
  likeButton.addEventListener("click", () => rateOutput(1));
  dislikeButton.addEventListener("click", () => rateOutput(-1));

  void loadModelStatus();
})();
