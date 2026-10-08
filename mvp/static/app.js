const $ = (id) => document.getElementById(id);
let selectedFile = null;
let previewURL = null;
let busy = false;
let validFile = false;
const microphoneSupported = MicrophoneRecorder.supported();
const recorder = new MicrophoneRecorder({
  onState(state) {
    const labels = { idle: 'Record audio', requesting: 'Requesting access…', recording: 'Stop recording', processing: 'Preparing audio…' };
    $('record-label').textContent = labels[state];
    $('record-button').setAttribute('aria-pressed', String(state === 'recording'));
    $('recording-live').hidden = state !== 'recording';
    const messages = { requesting: 'Allow microphone access in your browser to start.',
      recording: 'Recording… speak clearly, then select Stop recording.', processing: 'Preparing your recording…' };
    if (messages[state]) $('recording-status').textContent = messages[state];
    updateControls();
  },
  onTime(seconds) { $('recording-timer').textContent = `0:${String(Math.floor(seconds)).padStart(2, '0')} / 0:10`; },
  onRecording(file) {
    selectFile(file);
    $('recording-status').textContent = 'Recording ready. Listen to the preview, then classify.';
  },
  onError(error) {
    const messages = {
      NotAllowedError: 'Microphone access was denied. Allow microphone access in your browser settings, or upload a WAV file.',
      SecurityError: 'Microphone access is blocked by the browser. Check the site permissions or upload a WAV file.',
      NotFoundError: 'No microphone was found. Connect one or upload a WAV file.',
      NotReadableError: 'The microphone could not be opened. Check whether another app is using it, or upload a WAV file.'
    };
    $('error').textContent = messages[error.name] || error.message || 'Recording failed. Please try again or upload a WAV file.';
    $('error').hidden = false;
    $('recording-status').textContent = 'Recording did not complete. You can retry or upload audio.';
  }
});
function updateControls() {
  const capturing = recorder.state !== 'idle';
  $('audio-file').disabled = busy || capturing;
  $('dropzone').setAttribute('aria-disabled', String(busy || capturing));
  $('submit').disabled = busy || capturing || !validFile;
  $('record-button').disabled = busy || !microphoneSupported || ['requesting', 'processing'].includes(recorder.state);
  $('cancel-recording').hidden = !capturing;
  $('file-info').hidden = capturing || !selectedFile;
  $('preview').hidden = capturing || !validFile;
}
function clearResult() {
  $('result').hidden = true;
  $('empty-result').hidden = false;
  $('error').hidden = true;
  $('duration').textContent = '—';
}
function selectFile(file) {
  if (busy || recorder.state !== 'idle') return;
  clearResult();
  selectedFile = file || null;
  validFile = false;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $('preview').pause();
  $('preview').removeAttribute('src');
  if (microphoneSupported) $('recording-status').textContent = 'Recording stops automatically at 10 seconds.';
  updateControls();
  if (!selectedFile) return;
  $('filename').textContent = selectedFile.name;
  $('filesize').textContent = `${(selectedFile.size / 1024 / 1024).toFixed(2)} MiB`;
  if (selectedFile.size > 10 * 1024 * 1024 || !selectedFile.name.toLowerCase().endsWith('.wav')) {
    $('error').textContent = 'Choose a .wav file up to 10 MiB.';
    $('error').hidden = false;
    $('submit').disabled = true;
    $('preview').hidden = true;
    return;
  }
  previewURL = URL.createObjectURL(selectedFile);
  $('preview').src = previewURL;
  validFile = true;
  updateControls();
}
$('record-button').addEventListener('click', () => {
  if (busy || !microphoneSupported) return;
  if (recorder.state === 'recording') { recorder.stop(); return; }
  if (recorder.state !== 'idle') return;
  clearResult();
  $('preview').pause();
  recorder.start();
});
$('cancel-recording').addEventListener('click', () => {
  recorder.cancel();
  $('recording-status').textContent = validFile ? 'Recording cancelled. Your previous audio is still available.' :
    'Recording cancelled. Record again or upload a WAV file.';
});
function cancelForPageLeave() {
  if (recorder.state !== 'idle') {
    recorder.cancel();
    $('recording-status').textContent = 'Recording cancelled because the page was left. Please try again.';
  }
}
globalThis.addEventListener('pagehide', cancelForPageLeave);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) cancelForPageLeave();
});
updateControls();
if (!microphoneSupported) $('recording-status').textContent = 'Microphone recording requires a supported browser on HTTPS or localhost. You can still upload a WAV file.';
$('audio-file').addEventListener('change', (e) => selectFile(e.target.files[0]));
for (const event of ['dragenter', 'dragover']) $('dropzone').addEventListener(event, (e) => {
  e.preventDefault(); if (!busy && recorder.state === 'idle') $('dropzone').classList.add('dragging');
});
for (const event of ['dragleave', 'drop']) $('dropzone').addEventListener(event, (e) => {
  e.preventDefault(); $('dropzone').classList.remove('dragging');
});
$('dropzone').addEventListener('drop', (e) => selectFile(e.dataTransfer.files[0]));
$('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault(); if (!validFile || busy || recorder.state !== 'idle') return;
  busy = true; clearResult();
  updateControls();
  $('empty-result').hidden = true;
  $('loading').hidden = false;
  $('result-card').setAttribute('aria-busy', 'true');
  const form = new FormData(); form.append('audio', selectedFile);
  try {
    const response = await fetch('/api/classify', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || 'Classification failed. Try again.');
    $('prediction').textContent = data.prediction.display_label;
    $('duration').textContent = `${data.audio.duration_seconds.toFixed(2)} s`;
    $('classified-duration').textContent = `${data.audio.classified_duration_seconds.toFixed(2)} s`;
    $('model-version').textContent = data.model_version;
    $('warnings').textContent = data.warnings.join(' ');
    $('warnings').hidden = data.warnings.length === 0;
    $('result').hidden = false;
  } catch (error) {
    $('error').textContent = error.message === 'Failed to fetch' ? 'Cannot reach the server. Please retry.' : error.message;
    $('error').hidden = false;
    $('empty-result').hidden = false;
  } finally {
    busy = false;
    $('loading').hidden = true;
    updateControls();
    $('result-card').setAttribute('aria-busy', 'false');
    if (!$('result').hidden) $('result-card').scrollIntoView({ block: 'nearest' });
  }
});
