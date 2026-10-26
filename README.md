# Music Stem Separator 🎵

I make music and study computer science, so I wanted to build something that brought both of those interests together.

I've used tools that separate vocals and instruments from songs, but a lot of them either cost money or have limitations. I started wondering if I could make my own, and that's how this project started.

Music Stem Separator is a Python application that takes an MP3 or WAV file and separates it into four tracks: **vocals, drums, bass, and other instruments**. You can listen to each track separately and download it.

## How It Works

I'm using a pretrained model called **Hybrid Transformer Demucs (HTDemucs)**, developed by Meta. The model is designed to separate different sounds that are mixed together in a song.

When someone uploads an audio file, my application processes it, runs it through Demucs, and returns the four separated tracks.

The basic process looks like this:

```text
Upload MP3/WAV
      |
      v
Load and Process Audio
      |
      v
Choose Device (CUDA / MPS / CPU)
      |
      v
Run Demucs
      |
      v
Vocals | Drums | Bass | Other
      |
      v
Listen and Download
```

I didn't train Demucs from scratch. I'm using its pretrained weights and building the application around it.

## What I Built

- A web interface using Gradio where users can upload songs and listen to the results.
- Two processing options: Fast and High Quality.
- Device selection so the application can run on CUDA, Apple MPS, or CPU.
- A CPU fallback in case GPU processing fails.
- Docker support so the application can run in a container.
- A way to measure and compare inference performance.

## Testing and Performance

One of the first things I noticed was how much processing time changes depending on the hardware.

I initially ran the project on my Mac using Apple MPS. A song around 3.5 minutes long took approximately **104 seconds** for model inference after the weights had loaded.

I also experimented with Google Colab because it provides access to a free NVIDIA T4 GPU.

Later, I built a Docker image and tested the application using CPU inference in Google Cloud Shell. I uploaded a 2-minute, 54-second MP3, and the application successfully returned the separated tracks. It took several minutes, which made me realize that getting a model to work and making it fast enough for users are two different problems.

These are observations from my own testing, not controlled benchmarks.

## What I'm Learning

This project has been a way for me to learn more about audio machine learning beyond just reading about it.

I've been exploring how audio can be represented in the time and frequency domains, how source separation models work, and why running the same model on different hardware can produce very different processing times.

I'm also learning about **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**, which can be used to evaluate source separation when the original individual tracks are available.

I haven't completed a proper SI-SDR evaluation on a reference dataset yet, so I'm not reporting any separation quality scores.

## Running the Project

You'll need Python 3.12, FFmpeg, and the required Python packages.

Clone the repository:

```bash
git clone https://github.com/lindaassefa/music-stem-separator.git
cd music-stem-separator
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

You can also build and run it using Docker:

```bash
docker build -t music-stem-separator .
```

```bash
docker run --rm -p 8080:8080 \
  -e K_SERVICE=music-stem-separator \
  -e PORT=8080 \
  music-stem-separator
```

Then open `http://localhost:8080`.

The Docker setup was tested on CPU. The regular Python installation may need adjustments depending on your operating system and PyTorch setup.

## Things I Want to Improve

There are still quite a few things I want to experiment with:

- Test separation quality using SI-SDR and recordings with known reference stems.
- Compare processing times across CPU, MPS, and CUDA more systematically.
- Explore ONNX and ways to make inference more efficient.
- Experiment with separating Ethiopian instruments, especially the krar.
- Add audio examples so people can hear the results without running the application.

The Ethiopian instrument idea is something I'm especially interested in because I want to explore how well existing source separation models work with music and instruments that may be less represented in common training datasets.

## Tools I Used

Python, PyTorch, Torchaudio, Demucs, Gradio, Docker, FFmpeg.

## Credits

This project uses the pretrained Hybrid Transformer Demucs model from Meta's [Demucs repository](https://github.com/facebookresearch/demucs).

**Paper:** Rouard, Massa, and Défossez, *Hybrid Transformers for Music Source Separation* (ICASSP 2023).

## CPU Performance Benchmark

I benchmarked the pretrained HTDemucs model using a 10-second MP3 clip on CPU in a Docker container running in Google Cloud Shell.

| Metric | Result |
|---|---|
| Audio duration | 10 seconds |
| Sample rate | 44.1 kHz |
| Timed runs | 3 |
| Mean inference latency | 26.57 seconds |
| Standard deviation | 0.14 seconds |
| Mean real-time factor (RTF) | 2.66 |
| Model loading time | 1.73 seconds |
| Warm-up time | 50.03 seconds |
| Process lifetime peak RSS | 1.65 GiB |

For this benchmark, I used `shifts=0`, `overlap=0.25`, and `split=True`. Model loading, device transfer, and warm-up were measured separately from the timed inference runs.

RTF is calculated as inference latency divided by audio duration. An RTF of 2.66 means processing took approximately 2.66 times the duration of the audio, so CPU inference was not real-time in this environment.

The three runs were consistent, and the first and last runs produced identical outputs according to the benchmark's RMSE comparison. This measures repeatability, not separation quality.

These measurements are specific to this CPU environment and test clip. I have not yet measured GPU performance or evaluated separation accuracy using ground-truth stems.
