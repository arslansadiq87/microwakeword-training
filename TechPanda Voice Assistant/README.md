# TechPanda Voice Assistant

Firmware for a wake-word activated voice assistant built with a Seeed XIAO ESP32-S3, a ReSpeaker XVF3800 in I2S mode, and PlatformIO. It detects **“Hi Tech Panda”** locally, records a spoken question, sends it to Groq for transcription and a short answer, and plays Groq text-to-speech audio through the I2S speaker output.

## What is in this repository

- `src/` — firmware for I2S audio capture/playback, wake-word inference, transcription, chat, and speech output.
- `include/audio_config.h` — I2S wiring and audio format settings.
- `include/wake_word_model.h` and `src/wake_word_model.cpp` — the embedded, quantized wake-word model.
- `models/hi_tech_panda.tflite` — the corresponding TFLite model artifact.
- `platformio.ini` — board, framework, libraries, and serial settings.
- `include/secrets.example.h` — safe template for local Wi-Fi and Groq credentials.

## Hardware

The firmware is configured for:

- Seeed XIAO ESP32-S3 (with PSRAM recommended)
- ReSpeaker XVF3800 connected in I2S mode
- An I2S speaker/amplifier connected to the XVF3800 playback path

The current I2S pin assignment is in `include/audio_config.h`: BCLK GPIO 8, WS/LRCLK GPIO 7, data in GPIO 43, and data out GPIO 44. Check your XVF3800 carrier board pinout before wiring; board revisions and breakout wiring can differ. Audio is mono 16 kHz PCM for the assistant and 32-bit stereo I2S slots for the codec interface.

## Requirements

- VS Code with the PlatformIO IDE extension, or PlatformIO Core CLI
- A USB data cable and the XIAO ESP32-S3
- A 2.4 GHz Wi-Fi network with internet access
- A Groq account and API key with access to the configured speech, chat, and TTS models

The configured Groq model IDs are `whisper-large-v3-turbo` for transcription, `openai/gpt-oss-20b` for chat, and `canopylabs/orpheus-v1-english` for speech output. Model availability and account access are controlled by Groq and may change.

## Configure credentials

### Get a Groq API key

1. Create or sign in to your account at [Groq Console](https://console.groq.com/).
2. Open the **API Keys** section in the console and create a new key. Copy it when shown; treat it like a password.
3. Check Groq's current model availability and account requirements for the configured transcription, chat, and text-to-speech models. Usage may be subject to account limits or charges.

Keep the key private: do not put it in a public issue, screenshot, source file, or Git commit. If it is exposed, revoke it in the console and create another one.

From the project directory, copy the example header and edit the copy:

```powershell
Copy-Item include/secrets.example.h include/secrets.h
```

On macOS/Linux, use `cp include/secrets.example.h include/secrets.h`. Set `WIFI_SSID`, `WIFI_PASSWORD`, and `GROQ_API_KEY` in `include/secrets.h`. That file is intentionally ignored by Git. Do not paste credentials into source files or commit the local header. If a real key was exposed, revoke it and create a replacement before using the firmware.

## Build, upload, and monitor

Open this folder in VS Code with PlatformIO installed, select the `seeed_xiao_esp32s3` environment, then run **Build**, **Upload**, and **Monitor** from the PlatformIO toolbar.

With PlatformIO Core, run these commands in this folder:

```sh
pio run -e seeed_xiao_esp32s3
pio run -e seeed_xiao_esp32s3 -t upload
pio device monitor -b 115200
```

If PlatformIO does not find the board's serial port, select the correct port in PlatformIO's device list or add `upload_port` / `monitor_port` locally to `platformio.ini`. Connect to the board over USB, then open the serial monitor at 115200 baud. At boot, the log reports PSRAM detection, I2S setup, model initialization, and Wi-Fi status. Say “Hi Tech Panda” after the detector is ready; speak a question after the listening prompt. The assistant returns to wake-word listening after speaking.

## Use a newly trained wake word model

The training project and notebook are in the separate `microWakeWord` repository: [microWakeWord](https://github.com/kahrendt/microWakeWord). Its `notebooks/basic_training_notebook.ipynb` is a starting point for training a streaming quantized model. Training needs prepared positive samples and negative/ambient feature datasets; follow the notebook and the microWakeWord documentation for data preparation and evaluation.

This firmware currently expects a quantized streaming TFLite model with input shape `[1, 3, 40]` and the tensor quantization values checked in `src/wake_word.cpp`. The detector also uses the microWakeWord 40-feature frontend settings in that file. A model trained with different dimensions or quantization will not initialize correctly. To replace the current model, update both `models/hi_tech_panda.tflite` and the matching C array in `src/wake_word_model.cpp`, then check the model tensor shape, input/output types, quantization parameters, and supported operators against the checks and resolver in `src/wake_word.cpp` before building.

## Privacy and network behavior

Wake-word inference and audio capture run on the device. After a wake-word activation, the recorded question is sent to Groq's transcription API; the transcript is sent to Groq chat; and the answer text is sent to Groq TTS. Internet access and a valid API key are required for those online steps. Avoid speaking sensitive information if you do not want it sent to the configured provider.

## Troubleshooting

- **Wi-Fi does not connect:** verify the SSID/password and use a 2.4 GHz network supported by the ESP32-S3.
- **Groq returns 401/403:** check the locally configured key, its account access, and model availability. Never print or share the key when troubleshooting.
- **Model initialization fails:** confirm that the embedded array matches `models/hi_tech_panda.tflite`, the tensor metadata matches the firmware checks, and the board has sufficient memory.
- **No wake detections or poor audio:** verify I2S wiring and codec configuration, microphone/speaker connection, and ambient noise. The detection threshold and timing are configured in `src/wake_word.cpp` and `src/assistant.cpp`.
- **Upload or serial monitor fails:** use PlatformIO's detected port or configure the port locally; do not commit machine-specific port settings.
