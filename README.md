# microWakeWord Training and TechPanda Voice Assistant

This repository contains a complete example workflow: prepare and train a custom wake-word model with microWakeWord, then run an embedded model on an ESP32-S3 voice assistant.

| Project | Folder | Purpose |
| --- | --- | --- |
| microWakeWord training | [`microWakeWord/`](microWakeWord/README.md) | Generate training audio, prepare background data, train a model, and export a quantized streaming TFLite file. |
| TechPanda Voice Assistant | [`TechPanda Voice Assistant/`](TechPanda%20Voice%20Assistant/README.md) | Detect “Hi Tech Panda” on device, capture a question, use Groq for transcription and chat, and play a spoken response. |

The training notebook and firmware are separate parts of the workflow. The firmware is already supplied with a model; you do not need to train a new one to build and try the existing assistant.

## Repository layout

```text
microWakeWord/
  notebooks/basic_training_notebook.ipynb   # custom wake-word training walkthrough
TechPanda Voice Assistant/
  include/secrets.example.h                 # safe credentials template
  models/hi_tech_panda.tflite               # model artifact used by the firmware
  src/wake_word_model.cpp                   # same model embedded as a C++ byte array
README.md                                    # end-to-end guide
```

Training datasets, generated audio, feature maps, checkpoints, and local credentials are intentionally not included. The notebook downloads or creates the data it uses.

## Part 1: Train a wake-word model

### Requirements and notebook platform

- Windows with PowerShell, Git, and Python 3.10. The notebook's Piper model download cell invokes `powershell` directly.
- JupyterLab or another Jupyter notebook interface.
- A GPU is strongly recommended for model training. Data downloads and audio conversion need substantial time and free disk space.
- An internet connection for Python packages, Piper, and the audio datasets.

The edited notebook is Windows-oriented. Its final export cell imports `google.colab.files`, so on a local Windows Jupyter session run cells through training/export and get the resulting model directly from the output folder below. The notebook will not run unchanged on Colab/Linux: the sample-generation cell calls PowerShell, the editable install uses a path relative to the notebook working directory, and the last cell is Colab-only. Those cells need platform-specific edits for a Colab/Linux workflow.

### Create the environment and start Jupyter in the expected folder

Run these commands from the repository root in PowerShell:

```powershell
cd microWakeWord
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install jupyterlab
cd notebooks
jupyter lab basic_training_notebook.ipynb
```

Keep Jupyter's working directory at `microWakeWord/notebooks`. Notebook paths such as `generated_samples`, `negative_datasets`, and `trained_models` are relative to this directory, and the install cell runs `%pip install -e ..`. In the notebook, run cells from top to bottom. Its first code cell installs an `audio-metadata` source revision and the local microWakeWord package; on macOS it also installs a `pymicro-features` compatibility fork. Restart the kernel after installation if prompted.

### What each notebook stage does

1. **Choose the phrase and generate a test clip.** In cell 2, change `target_word` to the phrase you want, written as Piper should speak it. The cell clones Piper Sample Generator if needed, installs `torch`, `torchaudio`, and `piper-tts==1.3.0`, downloads the `en_US-libritts_r-medium.pt` voice model, generates one WAV file, and plays it. Check pronunciation before continuing.
2. **Generate positive examples.** Cell 3 creates 1,000 WAV samples for `target_word` under `notebooks/generated_samples`. If you change the phrase, rerun the sample and generation cells. Review samples for mispronunciations and unnatural clips before training.
3. **Download and prepare augmentation audio.** Cell 4 installs `datasets==2.19.0` and `soundfile`, then fetches MIT room impulse responses, an AudioSet balanced training archive, and the Free Music Archive XSmall dataset. It converts AudioSet and FMA audio to 16 kHz WAV. This is a large download and conversion step. Check the sources and license terms in [`microWakeWord/documentation/data_sources.md`](microWakeWord/documentation/data_sources.md) before using or redistributing the data.
4. **Configure augmentation and inspect an example.** Cell 5 splits positive clips into approximately 80% training, 10% validation, and 10% testing, and configures pitch, EQ, distortion, background noise, gain, and room impulse response augmentation. Cell 6 plays one augmented example so you can confirm it still sounds like the target phrase.
5. **Create positive feature maps.** Cell 7 writes spectrogram features in Ragged Mmap format under `generated_augmented_features/training`, `validation`, and `testing`. It uses repeated/shifted examples for training and separate validation and test splits.
6. **Fetch precomputed negative feature maps.** Cell 8 downloads and extracts the `speech`, `dinner_party`, `no_speech`, and `dinner_party_eval` sets into `negative_datasets/`. These provide speech and ambient examples that should not trigger the wake word.
7. **Write the training configuration.** Cell 9 creates `training_parameters.yaml`. The example uses 10,000 training steps, batch size 128, 1.5-second clips, evaluation every 500 steps, weighted negative examples, and `average_viable_recall` as the maximization metric. Adjust paths, weights, augmentation, and training duration to suit your datasets and experiments.
8. **Train and export.** Cell 10 installs TensorBoard and runs the MixedNet training/conversion command. The notebook requests the best weights and a quantized streaming TFLite export. Cell 11 downloads the result in Colab; for local Windows, find the file directly at `microWakeWord/notebooks/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`.

Training outputs and downloaded data live under `microWakeWord/notebooks`; do not commit them. Keep copies of the training configuration and checkpoints you need to reproduce or resume an experiment.

### Evaluate before deploying

The notebook is a starting point, not evidence that a model is ready for daily use. Listen to generated and augmented samples. Test with multiple speakers, microphones, distances, speaking styles, and noisy rooms. Pay particular attention to false activations in long recordings that do not contain the phrase. A model can have good notebook metrics and still perform poorly on your device or in your room.

If you change `target_word`, also update firmware log messages and user-facing wake-word text if you use the TechPanda assistant. Training a different phrase does not automatically change the embedded firmware model.

## Part 2: Build and use the voice assistant

### Hardware and software

- Seeed XIAO ESP32-S3; PSRAM is recommended.
- ReSpeaker XVF3800 configured for I2S, plus an I2S speaker/amplifier on the codec playback path.
- VS Code with PlatformIO IDE, or PlatformIO Core CLI; USB data cable.
- A 2.4 GHz Wi-Fi network with internet access.
- A Groq account and API key with access to the models configured in the firmware.

The pin settings are in `TechPanda Voice Assistant/include/audio_config.h`: BCLK GPIO 8, WS/LRCLK GPIO 7, data in GPIO 43, and data out GPIO 44. Check your specific carrier board's wiring before connecting hardware. The firmware uses mono 16 kHz PCM audio and 32-bit stereo I2S slots.

### Create a Groq API key and configure the firmware

1. Sign in or create an account at the [Groq Console](https://console.groq.com/).
2. Open [API Keys](https://console.groq.com/keys), create a key, and copy it somewhere private. Groq shows a new key for copying; treat it like a password.
3. Check the [current model list](https://console.groq.com/docs/models) and your account's limits, availability, and pricing. The firmware currently requests `whisper-large-v3-turbo` for transcription, `openai/gpt-oss-20b` for chat, and `canopylabs/orpheus-v1-english` for speech. The Orpheus model is listed as a preview and may change or be withdrawn.
4. From the firmware project folder, make a local secrets header:

   ```powershell
   Copy-Item include/secrets.example.h include/secrets.h
   ```

   On macOS/Linux, use `cp include/secrets.example.h include/secrets.h`.
5. Edit `include/secrets.h` and set `WIFI_SSID`, `WIFI_PASSWORD`, and `GROQ_API_KEY` to your values. `include/secrets.h` is ignored by Git; never add it to a commit. If you expose a key, revoke it in the Groq Console and create a replacement.

### Build, upload, and run

Open `TechPanda Voice Assistant/` as the PlatformIO project. In VS Code, choose the `seeed_xiao_esp32s3` environment and run **Build**, **Upload**, and **Monitor**. Or open a terminal in that folder and run:

```powershell
pio run -e seeed_xiao_esp32s3
pio run -e seeed_xiao_esp32s3 -t upload
pio device monitor -b 115200
```

If PlatformIO cannot find the serial port, select the connected board's port in PlatformIO or add local `upload_port` and `monitor_port` settings to `platformio.ini`. Do not commit machine-specific port settings. At startup, the serial log reports PSRAM, I2S, wake model, and Wi-Fi status. When the detector is ready, say **“Hi Tech Panda”**, then ask a question after the listening prompt.

The wake-word detector runs locally. After activation, the recorded question is sent to Groq speech-to-text, the transcript is sent to Groq chat, and the response text is sent to Groq text-to-speech. Those steps require Wi-Fi, an active API key, and provider model availability. Do not speak information you do not want sent to the configured provider.

## Use a newly trained model in the TechPanda firmware

The firmware currently validates a quantized streaming TFLite model with input shape `[1, 3, 40]`, int8 input scale `0.1019607857` and zero point `-128`, and uint8 output scale `1/256` and zero point `0`. These checks and the supported operator list are in `TechPanda Voice Assistant/src/wake_word.cpp`. A model with different tensor dimensions, quantization, or operators will fail initialization or require firmware changes, including possible operator registrations in the TFLite Micro resolver.

After confirming the new model matches those requirements, run this Python snippet from the repository root to copy the notebook export into the firmware and regenerate its embedded C++ byte array:

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

Review the threshold and model metadata checks in `src/wake_word.cpp`, then build and try the firmware on the target hardware. The current wake probability cutoff is also in that file. Update it based on real-world false-trigger and missed-trigger measurements. Keep the `.tflite` file and generated C++ array in sync.

## Privacy, data, and troubleshooting

- Do not commit `TechPanda Voice Assistant/include/secrets.h`, API keys, Wi-Fi credentials, local environments, or downloaded/generated datasets. The project ignore files exclude local secrets and common build/data artifacts.
- Review the source dataset license and conditions before using or redistributing audio; see [`microWakeWord/documentation/data_sources.md`](microWakeWord/documentation/data_sources.md).
- **Wi-Fi does not connect:** check SSID/password, use a supported 2.4 GHz network, and inspect the serial log.
- **Groq returns 401/403:** check the local API key, account permissions, and current model access. Do not print the key in logs.
- **Wake model initialization fails:** check the serial tensor metadata message; confirm the TFLite artifact and embedded C++ array are from the same file and match the firmware's expected metadata.
- **No or frequent detections:** verify the mic wiring and ambient audio, evaluate more positive and negative examples, and tune the probability threshold using long recordings.
- **Upload or monitor fails:** check the selected serial port and USB data cable.

## More detail

- [microWakeWord project README](microWakeWord/README.md)
- [TechPanda Voice Assistant README](TechPanda%20Voice%20Assistant/README.md)
- [Groq API quickstart](https://console.groq.com/docs/quickstart)
- [Groq model availability](https://console.groq.com/docs/models)
