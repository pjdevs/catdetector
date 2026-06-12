<script lang="ts">
  type CatLabel = "vickie" | "oka" | "both" | "none";

  type Prediction = {
    label: CatLabel;
    probabilities: {
      vickie: number;
      oka: number;
    };
    detected: {
      vickie: boolean;
      oka: boolean;
    };
    thresholds: {
      vickie: number;
      oka: number;
    };
  };

  const labelText: Record<CatLabel, string> = {
    vickie: "Vickie",
    oka: "Oka",
    both: "Both",
    none: "None",
  };

  let selectedFile: File | null = null;
  let previewUrl: string | null = null;
  let prediction: Prediction | null = null;
  let errorMessage = "";
  let isLoading = false;

  function handleFileChange(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const nextFile = input.files?.[0] ?? null;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    selectedFile = nextFile;
    previewUrl = nextFile ? URL.createObjectURL(nextFile) : null;
    prediction = null;
    errorMessage = "";
  }

  async function analyze() {
    if (!selectedFile) {
      errorMessage = "Choose a photo first.";
      return;
    }

    const body = new FormData();
    body.append("image", selectedFile);
    isLoading = true;
    errorMessage = "";
    prediction = null;

    try {
      const response = await fetch("/api/predictions", {
        method: "POST",
        body,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "Prediction failed.");
      }

      prediction = payload as Prediction;
    } catch (error) {
      errorMessage =
        error instanceof Error ? error.message : "Prediction failed.";
    } finally {
      isLoading = false;
    }
  }

  function probabilityPercent(value: number) {
    return `${Math.round(value * 100)}%`;
  }
</script>

<main class="shell">
  <section class="camera-panel">
    <div class="title-row">
      <div>
        <p class="eyebrow">Cat Detector</p>
        <h1>Vickie or Oka?</h1>
      </div>
      {#if prediction}
        <span class:quiet={prediction.label === "none"} class="result-pill">
          {labelText[prediction.label]}
        </span>
      {/if}
    </div>

    <label class="photo-picker" for="photo-input">
      {#if previewUrl}
        <img src={previewUrl} alt="Selected cat preview" />
      {:else}
        <span class="camera-mark">⌁</span>
        <span>Take or choose a photo</span>
      {/if}
    </label>
    <input
      id="photo-input"
      type="file"
      accept="image/*"
      capture="environment"
      on:change={handleFileChange}
    />

    <button class="analyze-button" disabled={!selectedFile || isLoading} on:click={analyze}>
      {isLoading ? "Analyzing..." : "Analyze"}
    </button>

    {#if errorMessage}
      <p class="error">{errorMessage}</p>
    {/if}

    {#if prediction}
      <div class="results" aria-live="polite">
        <div class="result-line">
          <span>Vickie</span>
          <strong>{probabilityPercent(prediction.probabilities.vickie)}</strong>
        </div>
        <div class="meter">
          <span style={`width: ${probabilityPercent(prediction.probabilities.vickie)}`}></span>
        </div>

        <div class="result-line">
          <span>Oka</span>
          <strong>{probabilityPercent(prediction.probabilities.oka)}</strong>
        </div>
        <div class="meter oka">
          <span style={`width: ${probabilityPercent(prediction.probabilities.oka)}`}></span>
        </div>
      </div>
    {/if}
  </section>
</main>
