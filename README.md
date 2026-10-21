# Music Stem Separator

I make music and study computer science, and I got tired of needing a paid tool just to separate vocals and instruments. Music Stem Separator is the app I built instead. You upload a WAV or MP3, and it comes back as four tracks you can play and download: vocals, drums, bass, and other.

The split uses **pretrained Hybrid Transformer Demucs** (`htdemucs`), published by Meta in the [Demucs](https://github.com/facebookresearch/demucs) project. This repository loads those weights and runs inference. It does not train Demucs, and it does not report separation quality numbers from this machine. The paper that introduced the model is Rouard, Massa, and Défossez, “Hybrid Transformers for Music Source Separation” (ICASSP 2023).

I ran it first on my Mac, on Apple MPS, so the project could stay free. A 3.5-minute song there took about 104 seconds inside the model once the weights were already loaded. That wait is what sent me to Google Colab’s free NVIDIA T4. The same application selects CUDA on that GPU. If CUDA or MPS fails, it separates the same audio again on CPU. A public demo has been a temporary Gradio share from the Colab session. Hugging Face Spaces is the free hosting option I am looking at next. No Space is set up in this repository yet.

