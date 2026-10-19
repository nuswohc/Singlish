# Two-model MVP

1. Keep the existing calibrated E1 classifier and display it as **ML**.
2. Add **Ensemble** from E4_02's `fusion_svm.joblib`: 35 raw acoustic
   features followed by a 512-value Whisper base.en embedding. Use the saved
   scaling, PCA and SVM pipeline without training or averaging predictions.
3. Classify each recording with both models. Use Ensemble as the final result,
   show both predictions and report whether their class labels agree.
4. Keep the recording/upload flow. Transcription stays outside the MVP interface
   as future work from Appendix B; there is no separate deep accent classifier.
5. Verify saved-pipeline parity, input ordering, Whisper pooling, API/UI behavior
   and deployment packaging.

The E4 export is uncalibrated and has no `predict_proba`. Show its class label;
display signed raw SVM decision scores for both models in the comparison cards.
Positive means Singapore, negative means Non-Singapore, and near zero means the
decision boundary. Magnitudes are not directly comparable across models. ML uses
the fitted SVM inside its calibration wrapper and follows its raw class decision;
its calibrated probability remains available only in the API.

Local artifacts come from
`/home/mh/nusproj/E1_Feature_Engineering/outputs/svm35_fusion_comparison_upstream`.
Binaries remain outside Git. Deployment needs both SVM artifacts and cached
Whisper weights, with more memory than the previous ML-only demo.

## Verification completed

- Backend and UI/recorder checks cover both model outputs, signed scores,
  input limits and microphone cleanup. Duplicate cases and redundant format
  combinations are omitted to keep the MVP suite small.
- Two held-out speech clips (MNSC and Speech Accent Archive) reproduce the
  notebook's decisions and scores using cached embeddings. Fresh CPU embeddings
  preserve both class labels; scores differ slightly from GPU bfloat16.
- Both clips succeed through the running upload API with the actual Whisper encoder.
- The generated deployment package starts and reports both models ready.
  Docker is unavailable locally, so an image build was not tested.
- The hosted app has not been redeployed. Its earlier 1 GB VM needs a memory
  upgrade before running this version; deployment guidance now specifies 4 GB.
