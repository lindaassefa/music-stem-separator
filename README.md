# Music Stem Separator

I make music and study computer science, and I got tired of needing a paid tool just to separate vocals and instruments. Music Stem Separator is the app I built instead. You upload a WAV or MP3, and it comes back as four tracks you can play and download: vocals, drums, bass, and other.

The split uses **pretrained Hybrid Transformer Demucs** (`htdemucs`), published by Meta in the [Demucs](https://github.com/facebookresearch/demucs) project. This repository loads those weights and runs inference. It does not train Demucs, and it does not report separation quality numbers from this machine. The paper that introduced the model is Rouard, Massa, and Défossez, “Hybrid Transformers for Music Source Separation” (ICASSP 2023).

I ran it first on my Mac, on Apple MPS, so the project could stay free. A 3.5-minute song there took about 104 seconds inside the model once the weights were already loaded. That wait is what sent me to Google Colab’s free NVIDIA T4. The same application selects CUDA on that GPU. If CUDA or MPS fails, it separates the same audio again on CPU. A public demo has been a temporary Gradio share from the Colab session. Hugging Face Spaces is the free hosting option I am looking at next. No Space is set up in this repository yet.

Section 1 separates a mix with pretrained htdemucs. Section 2 measures the stems with ordinary signal processing and checks how closely they add back to the mixture. Section 3 times that same pretrained model on the processors this machine can use. There is no classifier.

## How the pipeline works

1. The Gradio app accepts a WAV or MP3 upload.
2. `stem_ai/validation.py` checks the extension and the file header, then decodes the audio. WAV uses `soundfile`. MP3 uses `miniaudio`, so ffmpeg is not required. Empty files, corrupt files, silence, very short clips, and files longer than the configured limit are rejected before the model is loaded.
3. The waveform is converted to the channel count and sample rate htdemucs expects (stereo, 44.1 kHz) with Demucs’s own `convert_audio`.
4. `stem_ai/separation.py` loads `htdemucs` with `demucs.pretrained.get_model` and separates the mixture with `demucs.apply.apply_model`. Those are the same calls the Demucs 4.0.1 command line uses. The PyPI release does not include `demucs.api`, so this project does not depend on that module. The file-based Demucs reader is not used, because that reader shells out to ffmpeg.
5. The four stems are checked for matching length, finite samples, and a duration close to the upload. They are clipped into the WAV range and saved as 16-bit PCM WAV files by `soundfile`.
6. The page shows four compact tracks. Each one plays and downloads as a WAV. It does not show the device, timings, reconstruction residual, spectrograms, or benchmark numbers.
7. `stem_ai/analysis.py` can measure each stem with librosa and NumPy, draw a waveform and a log-frequency spectrogram, and compare the sum of the four stems with the mixture Demucs received. The Gradio app calls separation with those plots turned off. `separate()` still measures by default, so the tests and a direct call keep that record. The numbers are not a separation-quality score.

The inference timer wraps the successful `apply_model` call. Loudness normalization is done just before that call and is not included. The timer also excludes downloading or loading the weights, moving the model onto the processor, and writing WAV files. If CUDA or MPS fails, the same audio is separated again on CPU. `format_report` records that fallback. The app does not display it. Only the successful pass is timed.

`STEM_SHIFTS` defaults to `1`, which is the Demucs default: one prediction with a random time shift of up to half a second. Set `STEM_SHIFTS=0` for a single unshifted pass. Because the shift is random, two runs with the default can differ slightly.

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Gradio page: upload, four stem players, and WAV downloads |
| `stem_ai/dashboard.css` | Dark player layout for that page |
| `stem_ai/validation.py` | Upload rules and decoding |
| `stem_ai/separation.py` | Pretrained htdemucs inference and timing |
| `stem_ai/deployment.py` | Upload caps, and file rules used when a public share is turned on |
| `stem_ai/analysis.py` | DSP features, plots, and the reconstruction residual |
| `stem_ai/benchmark.py` | CPU and MPS timing of one local clip |
| `stem_ai/benchmark_metrics.py` | Latency, real-time factor, stem agreement, JSON, CSV, and the chart |
| `stem_ai/export.py` | WAV writing and output checks |
| `stem_ai/device.py` | CPU / MPS / CUDA selection |
| `tests/` | Validation, output integrity, device selection, analysis, benchmark metrics, and UI construction |
| `scripts/setup.sh` | Creates `.venv` and installs dependencies |
| `outputs/` | Generated stems. Git ignores this directory |

A useful review order is `validation.py`, then `separation.py`, then `export.py`, then `app.py`.

## Environment

Development targeted a native arm64 Python on an Apple Silicon Mac running macOS 12. PyTorch 2.5 and later disable MPS on macOS 12, so this Mac pins **PyTorch 2.4.1** and **torchaudio 2.4.1**. The app still asks PyTorch at runtime which device exists. `auto` chooses CUDA, then MPS, then CPU. Google Colab’s free T4 already has a CUDA build, and that is the device the same code selects there.

Demucs 4.0.1 declares `torchaudio<2.1`. Installing it as a normal dependency would fight the 2.4.1 pin, so `scripts/setup.sh` installs Demucs with `--no-deps` and installs the libraries it imports (`dora-search`, `einops`, `julius`, `lameenc`, `openunmix`, PyYAML, tqdm) from `requirements.txt`. `lameenc` is imported by Demucs even though this app writes WAV rather than MP3.

Check the interpreter and the selected device after installation:

```bash
.venv/bin/python -m stem_ai.device
```

## Installation

From the repository root:

```bash
bash scripts/setup.sh
```

The script uses `python3.12` unless `PYTHON` points somewhere else. Python 3.10 or newer is expected. Activate the environment with:

```bash
source .venv/bin/activate
```

## Run the app

From the repository root, with the virtual environment active:

```bash
python app.py
```

Open http://127.0.0.1:7860. Upload a WAV or MP3, then choose **Separate Audio**. The page is a dark player with four tracks: vocals, drums, bass, and other. Each track plays and downloads as a WAV. It does not show device, timing, reconstruction error, spectrograms, or the benchmark chart. The WAV files are written under `outputs/<timestamp>-<id>/`.

Saved benchmark files in `outputs/benchmarks` stay on disk for the engineering record. The app does not display that chart or those CPU and MPS timings. They are local Mac measurements, not Hugging Face server benchmarks, and opening the app does not rerun them.

The dev server does not open a browser on its own. Set `STEM_PORT` to change the port.

## Tests

```bash
pytest -q
```

The tests cover file type checks, silence, corrupt and truncated audio, a real MP3 decode, stem shape and sample integrity, WAV round-trips, device selection, DSP features, reconstruction arithmetic, benchmark latency and stem-agreement math, JSON and CSV writing, and the report text. They also confirm that an invalid or silent upload does not call the model loader. They do not download htdemucs and they do not claim a separation-quality score.

## What is neural, and what is DSP

Only the stem split is neural. `htdemucs` is a pretrained Hybrid Transformer Demucs model. This repository loads the published weights and runs `apply_model`. It does not train that model.

After the stems exist, these numbers are conventional DSP, computed on the mono average of each stem:

- **RMS** is the root-mean-square amplitude from NumPy.
- **Spectral centroid** and **spectral bandwidth** are the means of librosa’s frame-wise STFT features. The window is 2048 samples with a hop of 512 when the stem is at least that long. Shorter audio uses the largest power-of-two window that fits.
- **Zero-crossing rate** is the mean fraction of successive samples that cross zero, also from librosa.

The spectrogram image is a log-frequency view of the STFT magnitude in decibels relative to that image’s own peak. A long stem may be drawn with a larger hop than the hop used for the numbers; the image title says when that happens. The waveform image may skip samples so it stays drawable. RMS still uses every sample.

The reconstruction check is also arithmetic, not a neural score. It subtracts the mixture that entered Demucs from the sum of the four stem tensors, before 16-bit clipping. RMSE, that RMSE divided by the mixture RMS, and the peak absolute residual are whatever that subtraction produces. A perfect sum is not assumed. Clipping when the WAV files are written can move the files a little further from the mixture than this tensor residual.

## Benchmark

The benchmark times the published `htdemucs` weights. It does not train them. From the repository root:

```bash
python -m stem_ai.benchmark /path/to/song.wav --duration 10
```

`--duration` is the number of seconds read from the start of a file that is already on disk. A file shorter than that request is rejected rather than padded. `--skip-cpu` omits the CPU passes when they would take too long. `--runs` defaults to 3 timed runs after one warm-up on each device. Results go to `outputs/benchmarks/` as JSON, CSV, and a latency chart. That directory is gitignored with the rest of `outputs/`.

Every device sees the same waveform, the same loaded weights, and the same inference settings: `shifts=0`, overlap 0.25, split enabled, and the model’s own segment length. `shifts=0` is one unshifted pass, so the comparison is not moved around by Demucs’s random shift. Separation in the Gradio app still defaults to `STEM_SHIFTS=1`. The app does not display this benchmark.

Model loading and `model.to(device)` are timed on their own. They are not included in the inference mean. On MPS, `torch.mps.synchronize` runs before the clock starts and again before it stops. CPU has no such queue, so its sync field is null. The benchmark clock wraps the forward call, including loudness normalization. The app timer still wraps only `apply_model`. That time is kept in this record and is not shown on the page.

Real-time factor is inference latency divided by the clip duration. Below 1, the timed pass finished faster than the clip plays. The mean and the sample standard deviation (divisor n − 1) use the timed runs only. The warm-up is stored separately. One timed run would leave the standard deviation null.

Memory is recorded only when this process can measure it:

- `ps -o rss=` is the current resident size of the whole process, in kilobytes on macOS, stored as bytes. It is not GPU memory and it is not isolated to one device.
- `ru_maxrss` is the process lifetime high-water mark. On macOS it is already in bytes. It is reported once for the process, not assigned to MPS or CPU.
- `torch.mps.current_allocated_memory` and `torch.mps.driver_allocated_memory` are PyTorch MPS allocator counters after synchronize. They are not total GPU memory and not a hardware peak. On CPU those fields are null.

The JSON also compares the last timed stems from each device with RMSE, RMSE divided by the reference RMS, and peak absolute error. The same comparison is made between the first and last timed run on one device. That is numerical agreement. It is not a source-separation quality score. The Section 2 reconstruction RMSE is also not a quality score.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `STEM_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `STEM_SHIFTS` | `1` | Shifted Demucs passes. `0` is one unshifted pass |
| `STEM_SEGMENT` | model default | Chunk length in seconds, at most 7.8 for htdemucs |
| `STEM_MAX_DURATION_SEC` | `600` | Uploads longer than this are rejected |
| `STEM_OUTPUT_DIR` | `outputs` | Directory for WAV stems |
| `STEM_PORT` | `7860` | Port for the Gradio app |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` if unset | Lets unimplemented MPS ops run on CPU |

Silence means the whole file stays under a peak of `0.001` (about -60 dBFS). Audio shorter than 0.5 seconds is rejected. Files larger than 250 MB are rejected before decoding.

## Limits

- The model is the published four-stem `htdemucs` checkpoint, not a model trained for this project.
- The first successful separation needs a network connection so PyTorch can download the weights.
- CPU inference on a full song can take longer than the song. The page does not estimate or display that time. Measured stage times go to the process log.
- MP3 decoding uses miniaudio. ffmpeg is not required for this app.
- Files with more than eight channels are rejected. Stereo and mono are the expected inputs; mono is duplicated to stereo for the model.
- A 15-second excerpt of a local WAV was separated on this machine. Those measurements are listed below. The audio file is not part of the repository. Listen to a song you can use before judging how useful the stems are.
- On macOS 12, PyTorch may warn that MPS has no native int64 support and will downcast. That warning appeared during a synthetic run on this Mac, and the forward pass still finished on MPS.

## What was run on this Mac

Checked on October 14, 2026, on the development machine (macOS 12.5.1, arm64):

- Python 3.12.2, PyTorch 2.4.1. MPS was built and available. CUDA was not. The selected device was `mps`.
- `pytest -q` reported 68 passed.
- The first 10.00 seconds of `outputs/_section2/excerpt_0_15s.wav` (the start of the same local WAV used above, 44100 Hz) were benchmarked with `shifts=0`. This process loaded htdemucs in 0.556 seconds; the checkpoint was already on disk, and the model was not already in this process. MPS transfer took 0.188 seconds and the MPS warm-up took 6.744 seconds. The three timed MPS runs were 4.818, 4.038, and 4.121 seconds. The mean was 4.326 seconds, the sample standard deviation was 0.428 seconds, and the mean real-time factor was 0.433. CPU transfer took 0.212 seconds and the CPU warm-up took 13.864 seconds. The three timed CPU runs were 13.252, 20.636, and 20.265 seconds. The mean was 18.051 seconds, the sample standard deviation was 4.160 seconds, and the mean real-time factor was 1.805. CPU synchronization was null. These times include loudness normalization and, on MPS, the device sync. They are not the same measurement as the 10.12 second `apply_model` time above, which used 15.00 seconds of audio and the app’s default of one random shift.
- Process lifetime peak resident size for that benchmark process was 1016414208 bytes (`ru_maxrss` on macOS, whole process, not GPU memory). Sampled process RSS after the MPS runs was 366231552, 320290816, and 381927424 bytes. After the CPU runs it was 842760192, 238043136, and 620658688 bytes. Those samples are the whole process and changed between runs, so they are not a per-device memory total. After each MPS sync, PyTorch reported 167961088 bytes currently allocated and 2862989312 bytes driver-allocated. Those are allocator counters, not total GPU memory. The CPU runs left both MPS counters null.
- The last MPS and CPU stems differed by an overall RMSE of 3.209616e-04, an RMSE / MPS RMS of 2.292898e-02, and a peak absolute error of 6.013770e-03. Per-stem RMSE was vocals 1.193792e-05, drums 1.689547e-04, bass 4.332436e-04, and other 4.423542e-04. The first and last timed run on each device matched exactly (RMSE 0). That is device agreement under `shifts=0`, not separation quality.
- The benchmark wrote `outputs/benchmarks/benchmark-20261014-153825.json`, the matching CSV, and a latency chart. Those files remain the record of the run. The app does not recompute them.
- On that date the Gradio app served an upload form, a separate button, a run-details box, and four stem players. That page has since been replaced by the four-track player described above. Pressing separate with no file showed “Upload a WAV or MP3 file.”
- A 2.00 second synthetic stereo tone, not a song, was separated with the downloaded `htdemucs` weights. `apply_model` on MPS took 17.87 seconds. The four outputs were 16-bit WAV files at 44100 Hz and 88200 frames. This checks that inference runs. It does not measure separation quality.
- The first 15.00 seconds of a local stereo WAV, `New_Project (1).wav`, were separated with the same pretrained model. That excerpt’s peak was 0.2164 and its RMS was 0.0314 at 44100 Hz. On MPS, `apply_model` took 10.12 seconds, processing took 18.58 seconds, and inference time / duration was 0.675. Reconstruction RMSE was 4.137653e-03, RMSE / mixture RMS was 1.316315e-01, and the peak absolute residual was 2.470180e-02. Stem RMS values were vocals 2.939901e-05, drums 4.897300e-03, bass 4.277377e-03, and other 1.834217e-02. Centroids were vocals 5177.281 Hz, drums 5024.305 Hz, bass 1028.016 Hz, and other 1303.332 Hz. Bandwidths were vocals 6416.553 Hz, drums 5622.526 Hz, bass 2938.742 Hz, and other 2602.870 Hz. Zero-crossing rates were vocals 6.695689e-02, drums 9.470011e-02, bass 5.983713e-03, and other 2.878252e-02. These are measurements of that excerpt, not a quality score.
- After the dashboard redesign, the running app separated that same 15.00 second excerpt again on MPS, with the app default of one random shift. `apply_model` took 11.92 seconds, processing took 21.78 seconds, and inference time / duration was 0.795. Reconstruction RMSE was 3.971091e-03, RMSE / mixture RMS was 1.263326e-01, and the peak absolute residual was 2.494777e-02. The four outputs were 16-bit stereo WAV files at 44100 Hz and 661500 frames. Because the shift is random, this run is not expected to match the 10.12 second measurement exactly.
