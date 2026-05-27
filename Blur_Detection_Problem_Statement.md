# Problem Statement: On-Capture Image Quality Detection

## 1. Background

Students submit handwritten answer sheets to our platform by photographing them through our application. A meaningful proportion of these captures are unsuitable for downstream processing — affected by motion blur, defocus, glare, partial framing, or insufficient lighting. At present, these issues are identified only after upload, during OCR and grading, at which point the student has typically moved on and re-capture is no longer feasible.

The objective of this project is to detect unusable captures at the moment of acquisition, prior to upload, and prompt the student to retake the photograph.

## 2. Objective

Develop a detection component that evaluates an image immediately after capture and returns one of three verdicts: accept the image, request a retake, or flag for review. The component must operate within the constraints below and must be reliable enough to be placed directly in the user-facing capture path.

## 3. Constraints

The following constraints are non-negotiable. Submissions that do not satisfy them will not be considered.

- **Execution environment:** CPU only. No GPU dependency.
- **No machine learning inference.** No convolutional networks, vision-language models, on-device neural models, or external model APIs. The solution must rely on classical image processing techniques (e.g. Laplacian, Sobel/Tenengrad, frequency-domain analysis, gradient and histogram statistics).
- **No network calls.** The verdict must be derivable from the image alone.
- **Latency:** under 100 ms per image on a mid-range laptop CPU, measured on a representative 8–12 MP capture. Sub-50 ms is preferred. Image downsampling is acceptable where justified.

## 4. Scope

### In scope

- A function or service endpoint that accepts a single image and returns a verdict.
- A test harness that runs the function over a folder of images and emits per-image verdicts for review.

### Out of scope

- The mobile or web capture interface. Integration into the production capture flow will be handled internally; the deliverable is a clean function or endpoint with a documented contract.
- OCR, grading, or any server-side reprocessing.
- Image enhancement or restoration. The component decides only whether to accept or request re-capture.

## 5. Output Contract

The function must return a structured response of the form:

```text
{
  verdict: "usable" | "retake" | "borderline",
  reason:  "<short human-readable description, e.g. 'motion blur', 'partial framing', 'insufficient lighting'>",
  signals: { <metric_name>: <numeric_value>, ... }
}
```

The `signals` field is required. The numeric measurements underlying each verdict must be exposed for inspection and threshold tuning.

## 6. Problem Considerations

A naive variance-of-Laplacian threshold is a reasonable starting point but is insufficient as a final solution. Real student captures exhibit several characteristics that defeat such a baseline:

- Faint pencil writing on clean paper produces low high-frequency content and may be misclassified as blurry despite being usable.
- A sharply-focused photograph dominated by glare contains substantial high-frequency content but is unusable.
- Motion blur and defocus blur have distinguishable spectral signatures; the candidate may choose to treat them separately or jointly, with justification.
- Captures that are sharp but improperly framed (page partially outside the image) are unusable for OCR despite passing any sharpness test.
- Baseline sharpness and noise characteristics vary substantially across phone models; thresholds tuned on a single device may not generalise.

The candidate is expected to define the scope of failure modes addressed by the solution and to justify exclusions explicitly.

## 7. Testing Requirements

A rigorous evaluation is a required component of the deliverable.

### 7.1 Test set

Assembling the test set is part of the project and is the candidate's responsibility. A labelled set of at least **100 images** is required, curated to cover the full range of failure modes addressed by the solution as well as a representative spread of usable captures.

Acceptable sources include photographs taken by the candidate (using one or more phones, across varied lighting, paper types, and handwriting styles), publicly available handwritten-document images, and any internal samples we make available on request. The set must reflect realistic conditions: a corpus consisting only of photographs taken in a single environment will not be sufficient.

The composition of the set — how many images per failure mode, what counts as `usable` versus `retake`, how edge cases are treated — is itself a design decision and should be justified briefly in the report. Each image must be labelled `usable` or `retake` (a `borderline` label is permitted but should be used sparingly). Labels must be checked into the repository in CSV or JSON form, alongside either the images themselves or a script that reconstructs the set.

### 7.2 Required metrics

The following metrics must be reported on the test set:

- Confusion matrix (verdict against ground truth).
- **False-reject rate** — the proportion of usable images incorrectly flagged for retake. This metric is the primary user-experience risk and will be weighted accordingly in evaluation.
- **False-accept rate** — the proportion of unusable images permitted through.
- Per-failure-mode performance, reported separately for motion blur, defocus, glare, framing, and lighting (as applicable to the scope chosen).
- Latency at p50, p95, and maximum, on a named CPU.

### 7.3 Threshold justification

The submission must include the distribution of key signals across the `usable` and `retake` classes (a histogram is sufficient), the chosen threshold(s), and a written justification for the threshold derived from the data rather than intuition.

### 7.4 Failure analysis

At least five misclassified images must be discussed individually, with an explanation of the cause of failure and the change required to address it.

### 7.5 Reproducibility

All reported numbers must be reproducible from the repository with a single documented command.

## 8. Deliverables

1. A runnable repository with complete setup instructions.
2. The labelled test set assembled for evaluation, with a brief note on its composition and sourcing.
3. A written report of one to two pages covering: approach, signals used, threshold derivation, metrics, failure analysis, and proposed next steps.
4. A short screen recording with voice-over, focused on design decisions rather than code walkthrough, demonstrating the harness operating over a mixed folder.

## 9. Evaluation

The submission will be assessed through a live review in which the candidate is asked to explain and modify design decisions in real time. Topics will include the choice of signals, threshold derivation, handling of specific failure cases, and the trade-off between false-reject and false-accept rates. The use of AI-assisted tooling during development is permitted; the candidate is nevertheless expected to fully understand and be able to defend the submitted work.

## 10. Acceptance Criteria

- **Minimum acceptable:** a CPU-only, inference-free detector that reliably identifies obvious motion blur and defocus on the provided sample set, within the latency budget, accompanied by the required metrics.
- **Strong:** the above, with at least one additional failure mode (glare, framing, or lighting) handled and reported separately, and thresholds derived from data.
- **Excellent:** the above, with empirical evidence that gating uploads with the detector would reduce downstream OCR failures on a representative sample of production captures.

A focused, well-justified scope is preferred to a broader scope that is not fully supported by the evaluation. Where requirements are ambiguous, candidates are expected to raise the question rather than assume.
