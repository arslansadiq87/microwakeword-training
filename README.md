# microWakeWord Training and TechPanda Voice Assistant

This repository contains two related projects:

- [`microWakeWord/`](microWakeWord/README.md) — a Python framework and notebook for preparing data, training wake-word models, and exporting quantized streaming TFLite models.
- [`TechPanda Voice Assistant/`](TechPanda%20Voice%20Assistant/README.md) — PlatformIO firmware for a Seeed XIAO ESP32-S3 and ReSpeaker XVF3800. It embeds the “Hi Tech Panda” model and uses Groq services for transcription, chat, and speech output.

## Start here

Read the README inside the project you want to use. Training and data preparation are described in `microWakeWord/README.md`; firmware hardware, local Wi-Fi/API configuration, build, and upload steps are in `TechPanda Voice Assistant/README.md`.

The firmware uses a model with a specific input shape and quantization. Follow its README when replacing the embedded model with a newly trained one. Large training datasets and generated training artifacts are not stored in this repository. Firmware credentials are also excluded; create `TechPanda Voice Assistant/include/secrets.h` from the supplied `secrets.example.h` and enter your own values locally.