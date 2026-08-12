"use strict";

const models = {
  cnn: {
    label: "ConvNeXt",
    variant: "cnn",
    localization: "assets/figures/localization_cnn.png",
    faithfulness: "assets/figures/faithfulness_cnn.png",
    spurious: "assets/figures/spurious_cnn_patched.png",
  },
  vit: {
    label: "ViT-B/16",
    variant: "vit",
    localization: "assets/figures/localization_vit.png",
    faithfulness: "assets/figures/faithfulness_vit.png",
    spurious: "assets/figures/spurious_vit_patched.png",
  },
};

const methodLabels = {
  gradcam: "Grad-CAM",
  integrated_gradients: "Integrated Gradients",
  occlusion: "Occlusion",
};
const baselineMethods = new Set(["center", "random", "uniform"]);
let fullSummary = null;

function mean(payload) {
  return Number(payload.mean);
}

function percent(value, digits = 1) {
  return (value * 100).toFixed(digits) + "%";
}

function bestAttributionPointing(summary, variant) {
  return Object.entries(summary.localization[variant])
    .filter(([method]) => !baselineMethods.has(method))
    .map(([method, payload]) => ({
      method,
      value: mean(payload.all.pointing_rate),
    }))
    .reduce((best, candidate) => (candidate.value > best.value ? candidate : best));
}

function spuriousPatchEnergyMax(summary) {
  const values = [];
  Object.values(summary.spurious).forEach((methods) => {
    Object.values(methods).forEach((tests) => {
      Object.values(tests).forEach((payload) => {
        const value = payload.patch_energy_patched_inputs.all.patch_energy.mean;
        if (value !== null) values.push(Number(value));
      });
    });
  });
  return Math.max(...values);
}

function setMetric(name, value) {
  document.querySelectorAll("[data-metric='" + name + "']").forEach((element) => {
    element.textContent = value;
  });
}

function selectModel(key) {
  const model = models[key];
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.model === key));
  });

  const localization = document.querySelector("#localization-figure");
  const faithfulness = document.querySelector("#faithfulness-figure");
  const spurious = document.querySelector("#spurious-figure");
  localization.src = model.localization;
  localization.alt = model.label + " localization 彙總圖，含 95% bootstrap CI";
  faithfulness.src = model.faithfulness;
  faithfulness.alt = model.label + " deletion 與 insertion faithfulness 曲線";
  spurious.src = model.spurious;
  spurious.alt = model.label + " spurious patch 歸因能量彙總圖";

  document.querySelector("#localization-title").textContent = model.label + "：localization";
  document.querySelector("#faithfulness-title").textContent = model.label + "：faithfulness";
  document.querySelector("#spurious-title").textContent = model.label + "：spurious patch";

  if (!fullSummary) return;
  const train = fullSummary.train[model.variant];
  setMetric("val-accuracy", percent(train.val_accuracy));
  setMetric("val-f1", percent(train.val_macro_f1));
  const best = bestAttributionPointing(fullSummary, model.variant);
  setMetric("best-pointing", methodLabels[best.method] + " · " + percent(best.value));
  const ig = mean(
    fullSummary.randomization[model.variant].integrated_gradients.all.abs_spearman,
  );
  setMetric("ig-randomization", ig.toFixed(3));
  setMetric("ig-randomization-compact", ig.toFixed(3));
}

async function loadEvidence() {
  const status = document.querySelector("#data-status");
  try {
    const [summaryResponse, canaryResponse] = await Promise.all([
      fetch("data/summary.json"),
      fetch("data/cuda-resume-canary.json"),
    ]);
    if (!summaryResponse.ok || !canaryResponse.ok) {
      throw new Error("Evidence response was not successful.");
    }
    const [summary, canary] = await Promise.all([
      summaryResponse.json(),
      canaryResponse.json(),
    ]);
    const sampleCount = summary.localization?.cnn?.center?.all?.pointing_rate?.n;
    if (summary.schema_version !== 1 || summary.experiment !== "full" || sampleCount !== 500) {
      throw new Error("Unexpected summary contract.");
    }
    if (
      canary.schema_version !== 1 ||
      canary.status !== "PASS" ||
      canary.scope?.not_full_scale !== true
    ) {
      throw new Error("Unexpected canary contract.");
    }

    fullSummary = summary;
    const center = mean(summary.localization.cnn.center.all.pointing_rate);
    setMetric("center-pointing", center.toFixed(3));
    setMetric("spurious-patch-energy", percent(spuriousPatchEnergyMax(summary), 2));

    const canaryMap = {
      head: "head_state_exact",
      optimizer: "optimizer_state_exact",
      scaler: "grad_scaler_state_exact",
      metrics: "stable_metrics_exact",
    };
    Object.entries(canaryMap).forEach(([elementName, evidenceName]) => {
      const element = document.querySelector("[data-canary='" + elementName + "']");
      element.textContent = canary.comparisons[evidenceName] ? "完全一致" : "存在差異";
    });

    const selected = document.querySelector("[data-model][aria-pressed='true']");
    selectModel(selected ? selected.dataset.model : "cnn");
    status.textContent = "完整規模 JSON 已載入，資料結構與範圍驗證通過。";
  } catch (error) {
    status.textContent = "實驗證據載入失敗；數值暫不顯示，請檢查 artifact。";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.addEventListener("click", () => selectModel(button.dataset.model));
  });
  loadEvidence();
});
