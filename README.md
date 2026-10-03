# Hi Tech Panda: Custom Wake Word and ESP32-S3 Voice Assistant

This repository brings together the **microWakeWord training project** and the **TechPanda Voice Assistant firmware**. Use the included notebook to prepare and train a custom wake-word model, or skip training and build the assistant with its supplied “Hi Tech Panda” model.

## Before you start

- For the training notebook: Windows, PowerShell, Git, Python 3.10, JupyterLab or VS Code with the Python and Jupyter extensions, and a GPU if available. The notebook downloads and converts large audio datasets, so allow time and disk space.
- For the firmware: Seeed Studio XIAO ESP32-S3 (PSRAM recommended), ReSpeaker XVF3800 in I2S mode, a connected speaker/amplifier, a USB data cable, PlatformIO, and 2.4 GHz Wi-Fi.
- Internet is needed to install dependencies, download training audio, and use the online assistant services. Wake-word detection runs locally; speech transcription, chat, and speech generation use Groq.

The firmware and model are already in this repository. You do not need to create another PlatformIO project or generate the firmware by pasting coding prompts into an assistant.

## Projects in this repository

| Folder | What it contains |
| --- | --- |
| [`microWakeWord/`](microWakeWord/README.md) | Python library, custom training notebook, and dataset notes. |
| [`TechPanda Voice Assistant/`](TechPanda%20Voice%20Assistant/README.md) | Complete PlatformIO firmware, embedded model, model file, audio configuration, and a safe secrets template. |

Training data, generated audio, feature maps, checkpoints, build caches, and real credentials are not committed.

## 1. Set up the training environment

Clone this combined repository, then create an environment in the microWakeWord project. Run these commands from its root in PowerShell:

```powershell
git clone https://github.com/arslansadiq87/microwakeword-training.git
cd microwakeword-training\microWakeWord
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel
cd notebooks
jupyter lab basic_training_notebook.ipynb
```

In VS Code, open the notebook and select the Python environment at `microWakeWord/.venv`. If PowerShell blocks virtual environment activation, you can allow it for the current PowerShell process and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Keep the notebook's working directory at `microWakeWord/notebooks`: the cells write their data and models there, and the install cell runs `%pip install -e ..`. Run the notebook cells from top to bottom. Its first code cell installs an `audio-metadata` source revision and the local microWakeWord package; on macOS it also installs a `pymicro-features` compatibility fork. Restart the kernel after the installation cell completes.

### Notebook platform note

The current notebook is set up primarily for Windows. Its Piper model download invokes PowerShell, and its last cell uses `google.colab.files` to download the exported model. In local Windows Jupyter, run through the training cell and use the resulting file path listed below; the Colab download cell is unnecessary. The notebook will not run unchanged in Colab/Linux: adapt the PowerShell download, the working-directory-relative install, and the final download cell for that environment.

## 2. Run the training notebook

The existing notebook is at `microWakeWord/notebooks/basic_training_notebook.ipynb`. Use the cells in that file; the guide below describes what they do and what to check.

1. **Choose a phrase and check a sample.** In cell 2, set `target_word` (the default is `hi tech panda`). The cell installs Piper audio packages, clones Piper Sample Generator if needed, downloads the `en_US-libritts_r-medium.pt` voice model, generates one WAV sample, and plays it. Confirm that Piper pronounces the phrase clearly before generating the full set.
2. **Generate positive clips.** Cell 3 generates 1,000 samples under `microWakeWord/notebooks/generated_samples`. Review the clips for mispronunciations, cut-off words, and overly repetitive speech. If you change the phrase, rerun the test sample and full generation cells.
3. **Prepare augmentation audio.** Cell 4 installs `datasets==2.19.0` and `soundfile`, then downloads MIT room impulse responses, an AudioSet balanced audio archive, and the Free Music Archive XSmall dataset. It converts the AudioSet and FMA audio to 16 kHz WAV. This step can take a long time and use substantial disk space. If a download is interrupted, remove its incomplete archive before retrying; the helper skips existing nonempty files.
4. **Configure and preview augmentation.** Cell 5 splits positive clips into about 80% training, 10% validation, and 10% testing. It applies pitch, EQ, distortion, colored/background noise, gain, and room-response augmentation. Cell 6 creates and plays `augmented_clip.wav`; listen to ensure augmentation has not obscured the phrase.
5. **Generate positive feature sets.** Cell 7 writes Ragged Mmap spectrogram features beneath `generated_augmented_features/training`, `validation`, and `testing`. Training examples are repeated/shifted; validation and testing use their own splits.
6. **Download negative feature sets.** Cell 8 downloads and extracts `speech`, `dinner_party`, `no_speech`, and `dinner_party_eval` into `negative_datasets/`. They provide speech and ambient examples that should not trigger the wake word.
7. **Write the training config.** Cell 9 creates `training_parameters.yaml`. The example uses 10,000 training steps, batch size 128, a 1,500 ms clip duration, evaluation every 500 steps, weighted negative samples, and `average_viable_recall` for model selection. Change weights, steps, and augmentation settings as you experiment.
8. **Train and export.** Cell 10 installs TensorBoard and launches MixedNet training with a quantized streaming TFLite export. The exported model should appear here relative to the repository root:

   ```text
   microWakeWord/notebooks/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
   ```

   Cell 11 calls a Colab download helper. On Windows, use the file at the path above.

Training creates datasets, generated WAV files, feature maps, configuration, logs, and checkpoints below `microWakeWord/notebooks`. They are local working files, not files to upload to Git. Keep the configuration and checkpoints if you need to reproduce or resume a run.

### Evaluate before deployment

The example notebook is a starting point; one successful training run does not make a reliable detector. Test with multiple speakers, microphones, distances, speaking styles, and background conditions. Use long recordings without the wake phrase to check false activations as well as positive samples to check missed detections. Review the source and terms for downloaded datasets in [`microWakeWord/documentation/data_sources.md`](microWakeWord/documentation/data_sources.md).

## 3. Set up the included voice assistant

### Hardware and wiring

The firmware targets the Seeed Studio XIAO ESP32-S3 and a ReSpeaker XVF3800 in I2S mode. The configured pins in `TechPanda Voice Assistant/include/audio_config.h` are BCLK GPIO 8, WS/LRCLK GPIO 7, data in GPIO 43, and data out GPIO 44. Check the pinout for your specific carrier board before wiring; the firmware uses mono 16 kHz PCM in 32-bit stereo I2S slots. A connected I2S speaker/amplifier is required for spoken responses.

### Create a Groq API key and configure Wi-Fi

1. Sign in or create an account at [Groq Console](https://console.groq.com/).
2. Open [API Keys](https://console.groq.com/keys), create a key, and copy it. Treat the key like a password.
3. Check the [current Groq model list](https://console.groq.com/docs/models), account access, usage limits, and pricing. This firmware requests `whisper-large-v3-turbo` for speech-to-text, `openai/gpt-oss-20b` for chat, and `canopylabs/orpheus-v1-english` for text-to-speech. The Orpheus model is currently marked as a preview and can change or be withdrawn.
4. In PowerShell, move into the firmware folder and copy the safe template:

   ```powershell
   cd "TechPanda Voice Assistant"
   Copy-Item include/secrets.example.h include/secrets.h
   ```

5. Edit `include/secrets.h` and enter your `WIFI_SSID`, `WIFI_PASSWORD`, and `GROQ_API_KEY`. For macOS/Linux, use `cp include/secrets.example.h include/secrets.h` instead.

`include/secrets.h` is ignored by Git. Do not commit or share it. If you expose an API key, revoke it in Groq Console and create a new one. The same Groq key is used by this firmware for transcription, chat, and speech generation.

## 4. Build, upload, and test the firmware

Keep the terminal in `TechPanda Voice Assistant/`. PlatformIO installs the libraries listed in `platformio.ini` when building. You can use the PlatformIO Build, Upload, and Monitor buttons in VS Code, or run:

```powershell
pio run -e seeed_xiao_esp32s3
pio run -e seeed_xiao_esp32s3 -t upload
pio device monitor -b 115200
```

Connect the board with a USB data cable. If the serial port is not detected, select the board's port in PlatformIO or add `upload_port` and `monitor_port` locally to `platformio.ini`; do not commit machine-specific ports. The monitor is set to 115200 baud.

At startup, look for PSRAM, I2S, model, tensor allocation, Wi-Fi, and detector-ready messages. Say **“Hi Tech Panda”**. After the spoken acknowledgement and `WAITING FOR QUESTION SPEECH`, ask a question. The firmware detects speech and silence, transcribes the recording, asks Groq for an answer, and plays the answer through the speaker. It records up to 12 seconds and ends after about 1.3 seconds of silence following speech. It then returns to wake-word listening.

Typical runtime messages include:

```text
LISTENING FOR WAKE WORD
WAKE WORD DETECTED
SPEAKING: Wake acknowledgement
WAITING FOR QUESTION SPEECH
PROCESSING SPEECH
TRANSCRIPTION: ...
ASKING AI
AI RESPONSE: ...
SPEAKING
RETURNING TO WAKE WORD MODE
LISTENING FOR WAKE WORD
```

Wake-word detection runs locally. After activation, the recorded question is sent to Groq for transcription, the transcript is sent for chat, and the answer text is sent for speech generation. Do not speak information you do not want sent to the provider. The current HTTPS clients call `setInsecure()`, so they do not validate the server certificate; configure CA certificate validation before using the firmware in a security-sensitive deployment.

## 5. Put a newly trained model into the firmware

The included firmware already embeds `models/hi_tech_panda.tflite` in `src/wake_word_model.cpp`. To replace it, first confirm the exported model matches the checks in `src/wake_word.cpp`: quantized streaming input shape `[1, 3, 40]`, int8 input scale `0.1019607857` and zero point `-128`, uint8 output scale `1/256` and zero point `0`. The model must also use operators registered by the firmware. A different shape, quantization, or unsupported operator needs corresponding firmware changes.

After checking compatibility, run this from the repository root with the training output in place. It copies the TFLite file and regenerates the C++ array from the same bytes so the two files stay in sync:

```python
from pathlib import Path

source = Path(
    "microWakeWord/notebooks/trained_models/wakeword/"
    "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
)
model = source.read_bytes()

firmware = Path("TechPanda Voice Assistant")
(firmware / "models/hi_tech_panda.tflite").write_bytes(model)

rows = []
for start in range(0, len(model), 16):
    row = model[start : start + 16]
    rows.append("  " + ", ".join(f"0x{byte:02x}" for byte in row) + ",")

cpp = '''#include "wake_word_model.h"

#if defined(__GNUC__)
__attribute__((aligned(16)))
#endif
const uint8_t hi_tech_panda_model[] = {
''' + "\n".join(rows) + '''
};
const size_t hi_tech_panda_model_len = sizeof(hi_tech_panda_model);
'''
(firmware / "src/wake_word_model.cpp").write_text(cpp, encoding="utf-8")
print(f"Embedded {len(model)} model bytes")
```

Rebuild, upload, and evaluate detections on the real microphone. Adjust the probability cutoff in `src/wake_word.cpp` using measured false-trigger and missed-detection behavior. A successful compile alone does not prove the new model detects correctly.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `py -3.10` is unavailable | Install Python 3.10, reopen PowerShell, and confirm with `py -3.10 --version`. |
| Jupyter cannot import microWakeWord | Select the project's `.venv` kernel and keep the notebook working directory at `microWakeWord/notebooks`. Run the first notebook cell again if needed. |
| Piper model is missing | Complete the Piper setup in notebook cell 2. If the generator folder exists from a partial setup, finish or remove it before rerunning setup. |
| Dataset archive or extraction fails | Remove the incomplete archive, check free disk space and internet access, and rerun the download cell. |
| Feature folders already exist | Preserve results you need, then use a fresh output folder or remove stale generated files before regenerating. Keep the config paths consistent. |
| Training runs out of memory | Reduce the training batch size in `training_parameters.yaml`; consider fewer samples or a GPU with more memory. |
| Wi-Fi does not connect | Check the local SSID/password and use a supported 2.4 GHz network. |
| Groq returns 401/403 | Check the key in the ignored local `include/secrets.h`, account access, and current model availability. Never print the key to logs. |
| Model initialization fails | Confirm the `.tflite` file and C++ byte array match, then check tensor shape, quantization, and supported operators. |
| Upload or serial monitor fails | Close the serial monitor before upload, check the USB data cable and selected port, and retry. |
| No or frequent wake detections | Check mic wiring and signal level; evaluate varied positive and negative examples and tune the detector cutoff. |
| No spoken response | Check Wi-Fi/API status, model access, I2S playback wiring, and speaker/amplifier configuration. |

## References

- [microWakeWord project guide](microWakeWord/README.md)
- [TechPanda Voice Assistant guide](TechPanda%20Voice%20Assistant/README.md)
- [Dataset sources and notes](microWakeWord/documentation/data_sources.md)
- [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator)
- [Groq API quickstart](https://console.groq.com/docs/quickstart)
- [Groq current model list](https://console.groq.com/docs/models)
