# Two-model MVP

1. Use calibrated E1 `svm_65_calibrated.joblib` as **ML**.
2. Use calibrated E4 `fusion_svm_calibrated.joblib` as **Ensemble**: 65 raw
   acoustic features followed by a 512-value Whisper base.en embedding.
3. Run both saved pipelines without fitting. Select Singapore when its calibrated
   probability is at least 0.50, matching both calibration notebooks.
4. Use Ensemble as the final result, show both predicted-class probabilities,
   and report whether their labels agree. Keep raw decision scores in the API.
5. Keep recording/upload and preview. Ask users to retry when fewer than three
   seconds remain after silence trimming, before feature extraction or inference.

Artifacts come from `E1_Feature_EngineeringV2/outputs/`:

- `e1_feature_study_accent_30000/svm_calibrated_probability_testing/`
- `svm65_fusion_comparison_30000/calibrated_probability_testing/`

Binaries remain outside Git and are attached to the test candidate release.
Deployment packages both classifiers and their metadata; Whisper weights are
cached separately. See the [MVP README](README.md) for setup and verification.
