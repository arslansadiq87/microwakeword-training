# Hi Tech Panda — Custom Wake Word and ESP32-S3 AI Voice Assistant

A step-by-step companion to the Tech Panda video: generate “Hi Tech Panda” samples on a Windows PC, train a microWakeWord model, deploy local wake-word detection, and extend the same firmware into an online conversational assistant.

Channel: https://www.youtube.com/@TechPanda-4k

## Before you start

- Windows PC with PowerShell, Git, Python 3.10, VS Code, Microsoft Python and Jupyter extensions, and PlatformIO.
- Seeed Studio XIAO ESP32-S3 with ReSpeaker XVF3800, USB data cable, and a compatible connected speaker for the final assistant.
- Internet access for dependencies, datasets, and online assistant services. Wake-word inference runs locally; speech recognition, AI responses, and API-based speech generation require internet access.
- Use your actual hardware configuration and firmware files for I2S pins, channel selection, and ReSpeaker firmware. The video script does not supply a wiring table or ReSpeaker flashing procedure; do not guess these values.

This README includes every command and all three coding prompts from the script. Missing feature-generation and configuration cells were filled from the linked original notebook. The training command was adapted to use the active kernel on Windows. The inconsistent full-sample Piper cell was aligned with the test-sample command.

## Contents

1. Install Python and clone microWakeWord
2. Create the environment and open the notebook
3. Run training cells in order
4. Create the PlatformIO project — Codex prompt 1
5. Deploy local detection — Codex prompt 2
6. Build, upload, and test
7. Add the conversational assistant — Codex prompt 3
8. Configure credentials and test the final assistant
9. Troubleshooting and references

## 1. Install Python and clone microWakeWord

Create a new empty folder, such as **MicroWakeWord Training**, and open PowerShell in it. Check Python first:


```powershell
py -3.10 --version
```

If Python 3.10 is missing, install it:


```powershell
winget install --id Python.Python.3.10 -e
```

Close and reopen PowerShell after installation, then check again:


```powershell
py -3.10 --version
```

Clone the repository:


```powershell
git clone https://github.com/kahrendt/microWakeWord
```

Open the newly created `microWakeWord` folder in VS Code. All environment commands below run from this repository root.

## 2. Create the environment and open the notebook

Open the VS Code PowerShell terminal and run each command:


```powershell
py -3.10 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ipykernel
```

Open `notebooks/basic_training_notebook.ipynb`. Install Microsoft's Jupyter extension if prompted. Select **Select Kernel → Python Environments → env**; select the interpreter in `microWakeWord/env`.

## 3. Run training cells in order

Copy the following code into the corresponding notebook cells. `%pip` is Jupyter syntax: run these cells in the notebook, not PowerShell. Keep the notebook working directory set to `microWakeWord/notebooks` because paths such as `..`, `generated_samples`, and `training_parameters.yaml` depend on it.

### 3.1 Install microWakeWord


```python
# Installs microWakeWord. Restart the kernel after this is finished.

import platform

if platform.system() == "Darwin":
    %pip install 'git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version'

%pip install 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'

%pip install -e ..
```

Restart the notebook kernel after installation. Run subsequent cells in order.

### 3.2 Generate one test sample

Set your wake phrase in `target_word` and listen before generating the full dataset. The PyTorch compatibility change is for this downloaded Piper checkpoint; use the linked official source.


```python
# Generates 1 sample of the target word for manual verification.

target_word = 'hi tech panda'

import os
import sys
import platform

from IPython.display import Audio

if not os.path.exists("./piper-sample-generator"):

    !git clone https://github.com/rhasspy/piper-sample-generator

    %pip install torch torchaudio piper-phonemize-fix

    # Compatibility fix for newer PyTorch versions
    !powershell -Command "(Get-Content 'piper-sample-generator/generate_samples.py') -replace 'torch.load\(model_path\)', 'torch.load(model_path, weights_only=False)' | Set-Content 'piper-sample-generator/generate_samples.py'"

    if not os.path.exists("piper-sample-generator/models"):
        os.makedirs("piper-sample-generator/models")

    !powershell -Command "Invoke-WebRequest -Uri 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt' -OutFile 'piper-sample-generator/models/en_US-libritts_r-medium.pt'"

    if "piper-sample-generator/" not in sys.path:
        sys.path.append("piper-sample-generator/")

!{sys.executable} piper-sample-generator/generate_samples.py "{target_word}" --max-samples 1 --batch-size 1 --output-dir generated_samples

Audio("generated_samples/0.wav", autoplay=True)
```

On reruns, the setup in the original cell only runs if `piper-sample-generator` does not exist. If a download or install failed, finish that setup before continuing. Ensure the model exists at `piper-sample-generator/models/en_US-libritts_r-medium.pt`.

### 3.3 Generate 1,000 samples

The script's full-dataset cell referenced undefined `model_path` and `piper_dir` and a different module entry point. This replacement uses the same script as the one-sample cell and the current notebook interpreter.


```python
import sys
import subprocess

subprocess.run([
    sys.executable,
    "piper-sample-generator/generate_samples.py",
    target_word,
    "--max-samples", "1000",
    "--batch-size", "100",
    "--output-dir", "generated_samples",
], check=True)
print("Wake-word sample generation finished.")
```

Keep batch size 100 for the video settings. If you run out of memory, reduce it. Existing sample files may be overwritten; use a separate folder for a different wake phrase.

### 3.4 Download augmentation audio

This is the Windows-friendly cell from the script. Downloads can take time and require disk space. If interrupted, remove the incomplete archive before rerunning: the original helper skips any existing nonempty file.

The upstream notebook notes mixed dataset licenses and describes models trained using these augmentation datasets as appropriate for noncommercial personal use. Check dataset terms before commercial reuse.


```python
# ============================================================
# ONE-GO CELL - Download and Prepare All Augmentation Datasets
# ============================================================

%pip install datasets==2.19.0 soundfile

import os
import zipfile
import tarfile
import requests
import datasets
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from tqdm import tqdm


# ============================================================
# Helper: Download File
# ============================================================

def download_file(url, destination):

    # Remove empty/failed previous download
    if os.path.exists(destination) and os.path.getsize(destination) == 0:
        os.remove(destination)

    if os.path.exists(destination):
        print(f"Already downloaded: {destination}")
        return

    print(f"Downloading: {destination}")

    with requests.get(
        url,
        stream=True,
        allow_redirects=True
    ) as r:

        r.raise_for_status()

        total_size = int(
            r.headers.get("content-length", 0)
        )

        with open(destination, "wb") as f:

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as progress:

                for chunk in r.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))


# ============================================================
# 1. MIT Room Impulse Responses
# ============================================================

print("\n========================================")
print("STEP 1/3 - MIT Room Impulse Responses")
print("========================================")

mit_dir = "mit_rirs"

if not os.path.exists(mit_dir):

    os.makedirs(mit_dir, exist_ok=True)

    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True
    )

    for row in tqdm(
        rir_dataset,
        desc="Processing MIT RIR"
    ):

        name = Path(
            row["audio"]["path"]
        ).name

        output_file = os.path.join(
            mit_dir,
            name
        )

        scipy.io.wavfile.write(
            output_file,
            16000,
            (
                row["audio"]["array"] * 32767
            ).astype(np.int16)
        )

    print("MIT RIR dataset ready.")

else:

    print("MIT RIR dataset already exists. Skipping.")


# ============================================================
# 2. AudioSet
# ============================================================

print("\n========================================")
print("STEP 2/3 - AudioSet Background Audio")
print("========================================")

audioset_dir = "audioset"
audioset_16k_dir = "audioset_16k"

os.makedirs(
    audioset_dir,
    exist_ok=True
)

os.makedirs(
    audioset_16k_dir,
    exist_ok=True
)

audioset_archive = os.path.join(
    audioset_dir,
    "bal_train09.tar"
)

audioset_url = (
    "https://huggingface.co/datasets/"
    "agkphysics/AudioSet/resolve/"
    "196c0900867eff791b8f4d4be57db277e9a5b131/"
    "bal_train09.tar"
)

download_file(
    audioset_url,
    audioset_archive
)


# Extract AudioSet

audioset_audio_dir = os.path.join(
    audioset_dir,
    "audio"
)

if not os.path.exists(audioset_audio_dir):

    print("Extracting AudioSet...")

    with tarfile.open(
        audioset_archive,
        "r"
    ) as tar:

        tar.extractall(
            audioset_dir
        )

    print("AudioSet extracted.")

else:

    print("AudioSet already extracted.")


# Find FLAC files anywhere under extracted AudioSet

audioset_files = list(
    Path(audioset_dir).glob("**/*.flac")
)

print(
    f"Found {len(audioset_files)} AudioSet files."
)


# Convert AudioSet to 16 kHz WAV

if audioset_files:

    audioset_dataset = datasets.Dataset.from_dict({
        "audio": [
            str(file)
            for file in audioset_files
        ]
    })

    audioset_dataset = (
        audioset_dataset.cast_column(
            "audio",
            datasets.Audio(
                sampling_rate=16000
            )
        )
    )

    for row in tqdm(
        audioset_dataset,
        desc="Converting AudioSet"
    ):

        source_name = Path(
            row["audio"]["path"]
        ).stem

        output_file = os.path.join(
            audioset_16k_dir,
            source_name + ".wav"
        )

        if os.path.exists(output_file):
            continue

        audio_data = (
            row["audio"]["array"] * 32767
        ).astype(np.int16)

        scipy.io.wavfile.write(
            output_file,
            16000,
            audio_data
        )

    print("AudioSet conversion complete.")

else:

    raise RuntimeError(
        "AudioSet downloaded, but no FLAC files were found."
    )


# ============================================================
# 3. Free Music Archive
# ============================================================

print("\n========================================")
print("STEP 3/3 - Free Music Archive")
print("========================================")

fma_dir = "fma"
fma_16k_dir = "fma_16k"

os.makedirs(
    fma_dir,
    exist_ok=True
)

os.makedirs(
    fma_16k_dir,
    exist_ok=True
)

fma_archive = os.path.join(
    fma_dir,
    "fma_xs.zip"
)

fma_url = (
    "https://huggingface.co/datasets/"
    "mchl914/fma_xsmall/resolve/main/"
    "fma_xs.zip"
)

download_file(
    fma_url,
    fma_archive
)


# Extract FMA

fma_audio_dir = os.path.join(
    fma_dir,
    "fma_small"
)

if not os.path.exists(fma_audio_dir):

    print("Extracting Free Music Archive...")

    with zipfile.ZipFile(
        fma_archive,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            fma_dir
        )

    print("FMA extracted.")

else:

    print("FMA already extracted.")


# Find MP3 files

fma_files = list(
    Path(fma_dir).glob("**/*.mp3")
)

print(
    f"Found {len(fma_files)} FMA files."
)


# Convert FMA to 16 kHz WAV

if fma_files:

    fma_dataset = datasets.Dataset.from_dict({
        "audio": [
            str(file)
            for file in fma_files
        ]
    })

    fma_dataset = (
        fma_dataset.cast_column(
            "audio",
            datasets.Audio(
                sampling_rate=16000
            )
        )
    )

    for row in tqdm(
        fma_dataset,
        desc="Converting FMA"
    ):

        source_name = Path(
            row["audio"]["path"]
        ).stem

        output_file = os.path.join(
            fma_16k_dir,
            source_name + ".wav"
        )

        if os.path.exists(output_file):
            continue

        audio_data = (
            row["audio"]["array"] * 32767
        ).astype(np.int16)

        scipy.io.wavfile.write(
            output_file,
            16000,
            audio_data
        )

    print("FMA conversion complete.")

else:

    raise RuntimeError(
        "FMA downloaded, but no MP3 files were found."
    )


# ============================================================
# Final Verification
# ============================================================

mit_count = len(
    list(Path(mit_dir).glob("**/*.wav"))
)

audioset_count = len(
    list(Path(audioset_16k_dir).glob("*.wav"))
)

fma_count = len(
    list(Path(fma_16k_dir).glob("*.wav"))
)


print("\n========================================")
print("ALL AUGMENTATION DATASETS ARE READY")
print("========================================")

print(f"MIT RIR files     : {mit_count}")
print(f"AudioSet WAV files: {audioset_count}")
print(f"FMA WAV files     : {fma_count}")
```

### 3.5 Set up augmentation


```python
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

clips = Clips(
    input_directory='generated_samples',
    file_pattern='*.wav',
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=['mit_rirs'],
    background_paths=['fma_16k', 'audioset_16k'],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)
```

### 3.6 Preview an augmented sample


```python
from IPython.display import Audio
from microwakeword.audio.audio_utils import save_clip

random_clip = clips.get_random_clip()
augmented_clip = augmenter.augment_clip(random_clip)

save_clip(
    augmented_clip,
    'augmented_clip.wav'
)

Audio(
    "augmented_clip.wav",
    autoplay=True
)
```

Listen to the result before continuing.

### 3.7 Generate training, validation, and testing features

The video says to leave this cell unchanged. Its code is reproduced from the original notebook below. Run once and wait for completion.


```python
# Augment samples and save the training, validation, and testing sets.
# Validating and testing samples generated the same way can make the model
# benchmark better than it performs in real-word use. Use real samples or TTS
# samples generated with a different TTS engine to potentially get more accurate
# benchmarks.

import os
from mmap_ninja.ragged import RaggedMmap

output_dir = 'generated_augmented_features'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

splits = ["training", "validation", "testing"]
for split in splits:
  out_dir = os.path.join(output_dir, split)
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)


  split_name = "train"
  repetition = 2

  spectrograms = SpectrogramGeneration(clips=clips,
                                     augmenter=augmenter,
                                     slide_frames=10,    # Uses the same spectrogram repeatedly, just shifted over by one frame. This simulates the streaming inferences while training/validating in nonstreaming mode.
                                     step_ms=10,
                                     )
  if split == "validation":
    split_name = "validation"
    repetition = 1
  elif split == "testing":
    split_name = "test"
    repetition = 1
    spectrograms = SpectrogramGeneration(clips=clips,
                                     augmenter=augmenter,
                                     slide_frames=1,    # The testing set uses the streaming version of the model, so no artificial repetition is necessary
                                     step_ms=10,
                                     )

  RaggedMmap.from_generator(
      out_dir=os.path.join(out_dir, 'wakeword_mmap'),
      sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
      batch_size=100,
      verbose=True,
  )
```

### 3.8 Download negative features

These are pre-generated negative spectrogram features, used to teach the model what should not trigger the wake word.


```python
import os
import zipfile
import requests
from tqdm import tqdm

output_dir = './negative_datasets'
os.makedirs(output_dir, exist_ok=True)

link_root = (
    "https://huggingface.co/datasets/"
    "kahrendt/microwakeword/resolve/main/"
)

filenames = [
    'dinner_party.zip',
    'dinner_party_eval.zip',
    'no_speech.zip',
    'speech.zip'
]

for fname in filenames:

    link = link_root + fname
    zip_path = os.path.join(output_dir, fname)

    if not os.path.exists(zip_path):

        print(f"Downloading {fname}...")

        with requests.get(link, stream=True) as r:
            r.raise_for_status()

            total_size = int(
                r.headers.get('content-length', 0)
            )

            with open(zip_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=fname
            ) as bar:

                for chunk in r.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    print(f"Extracting {fname}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

print("Negative datasets ready.")
```

### 3.9 Write the training configuration

Original notebook configuration: 10,000 training steps, batch size 128, maximum wake-word duration 1,500 ms. This creates `training_parameters.yaml`.


```python
# Save a yaml config that controls the training process
# These hyperparamters can make a huge different in model quality.
# Experiment with sampling and penalty weights and increasing the number of
# training steps.

import yaml
import os

config = {}

config["window_step_ms"] = 10

config["train_dir"] = (
    "trained_models/wakeword"
)


# Each feature_dir should have at least one of the following folders with this structure:
#  training/
#    ragged_mmap_folders_ending_in_mmap
#  testing/
#    ragged_mmap_folders_ending_in_mmap
#  testing_ambient/
#    ragged_mmap_folders_ending_in_mmap
#  validation/
#    ragged_mmap_folders_ending_in_mmap
#  validation_ambient/
#    ragged_mmap_folders_ending_in_mmap
#
#  sampling_weight: Weight for choosing a spectrogram from this set in the batch
#  penalty_weight: Penalizing weight for incorrect predictions from this set
#  truth: Boolean whether this set has positive samples or negative samples
#  truncation_strategy = If spectrograms in the set are longer than necessary for training, how are they truncated
#       - random: choose a random portion of the entire spectrogram - useful for long negative samples
#       - truncate_start: remove the start of the spectrogram
#       - truncate_end: remove the end of the spectrogram
#       - split: Split the longer spectrogram into separate spectrograms offset by 100 ms. Only for ambient sets

config["features"] = [
    {
        "features_dir": "generated_augmented_features",
        "sampling_weight": 2.0,
        "penalty_weight": 1.0,
        "truth": True,
        "truncation_strategy": "truncate_start",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/speech",
        "sampling_weight": 10.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/dinner_party",
        "sampling_weight": 10.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
    {
        "features_dir": "negative_datasets/no_speech",
        "sampling_weight": 5.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "random",
        "type": "mmap",
    },
    { # Only used for validation and testing
        "features_dir": "negative_datasets/dinner_party_eval",
        "sampling_weight": 0.0,
        "penalty_weight": 1.0,
        "truth": False,
        "truncation_strategy": "split",
        "type": "mmap",
    },
]

# Number of training steps in each iteration - various other settings are configured as lists that corresponds to different steps
config["training_steps"] = [10000]

# Penalizing weight for incorrect class predictions - lists that correspond to training steps
config["positive_class_weight"] = [1]
config["negative_class_weight"] = [20]

config["learning_rates"] = [
    0.001,
]  # Learning rates for Adam optimizer - list that corresponds to training steps
config["batch_size"] = 128

config["time_mask_max_size"] = [
    0
]  # SpecAugment - list that corresponds to training steps
config["time_mask_count"] = [0]  # SpecAugment - list that corresponds to training steps
config["freq_mask_max_size"] = [
    0
]  # SpecAugment - list that corresponds to training steps
config["freq_mask_count"] = [0]  # SpecAugment - list that corresponds to training steps

config["eval_step_interval"] = (
    500  # Test the validation sets after every this many steps
)
config["clip_duration_ms"] = (
    1500  # Maximum length of wake word that the streaming model will accept
)

# The best model weights are chosen first by minimizing the specified minimization metric below the specified target_minimization
# Once the target has been met, it chooses the maximum of the maximization metric. Set 'minimization_metric' to None to only maximize
# Available metrics:
#   - "loss" - cross entropy error on validation set
#   - "accuracy" - accuracy of validation set
#   - "recall" - recall of validation set
#   - "precision" - precision of validation set
#   - "false_positive_rate" - false positive rate of validation set
#   - "false_negative_rate" - false negative rate of validation set
#   - "ambient_false_positives" - count of false positives from the split validation_ambient set
#   - "ambient_false_positives_per_hour" - estimated number of false positives per hour on the split validation_ambient set
config["target_minimization"] = 0.9
config["minimization_metric"] = None  # Set to None to disable

config["maximization_metric"] = "average_viable_recall"

with open(os.path.join("training_parameters.yaml"), "w") as file:
    documents = yaml.dump(config, file)
```

### 3.10 Train and export the model

The script asks to replace the training cell but does not include its replacement. This adapts the original notebook's argument list to a Windows-friendly `subprocess` call using the selected kernel.


```python
import sys
import subprocess

subprocess.run([
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config=training_parameters.yaml",
    "--train", "1",
    "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0",
    "--test_tflite_nonstreaming", "0",
    "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0",
    "--test_tflite_streaming_quantized", "1",
    "--use_weights", "best_weights",
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
], check=True)
```

Allow training and quantized model export to finish. Look for this file relative to the notebook working directory:

`trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`

Keep a copy of the model and training configuration. This workflow has not been executed end to end as part of preparing this README; check the notebook output and test detection on your hardware.

## 4. Create the PlatformIO project — Codex prompt 1

Create a new empty folder named `TechPandaVoiceAssistant` outside the training repository. Open it in VS Code, open Codex, and paste this entire prompt. Its project tree describes the current folder; do not create a second nested folder.


```text
Create a new PlatformIO project in the current folder for the
Seeed Studio XIAO ESP32-S3.

Use:
- PlatformIO
- Arduino framework
- Board: seeed_xiao_esp32s3

Do everything required to initialize and configure the project.

Create this project structure:

TechPandaVoiceAssistant/
│
├── include/
│   ├── wake_word_model.h
│   ├── audio_config.h
│   └── secrets.h
│
├── src/
│   ├── main.cpp
│   ├── audio_input.cpp
│   ├── wake_word.cpp
│   ├── speech_to_text.cpp
│   ├── assistant.cpp
│   └── text_to_speech.cpp
│
├── models/
│   └── hi_tech_panda.tflite
│
└── platformio.ini

Requirements:

1. Initialize the PlatformIO project directly inside the current folder.
   Do not create another nested project folder.

2. Configure platformio.ini for:
   - Seeed Studio XIAO ESP32-S3
   - Arduino framework
   - Serial Monitor at 115200 baud.

3. Create all folders and source/header files shown above.

4. For now, create clean placeholder implementations in the source files
   so the project can compile.

5. main.cpp should initialize Serial at 115200 and print:

   Tech Panda Voice Assistant
   System initialized.

6. audio_config.h should be reserved for ReSpeaker XVF3800 audio/I2S
   configuration.

7. wake_word_model.h should be reserved for the embedded microWakeWord
   TensorFlow Lite model.

8. secrets.h should contain placeholders only for:
   - Wi-Fi SSID
   - Wi-Fi password
   - API key

   Do not put any real credentials in the file.

9. The models folder should be created for:

   hi_tech_panda.tflite

   Do not generate a fake TensorFlow Lite model.
   If the model file is not present, leave the models folder ready for me
   to copy the trained file into it.

10. Create the initial interfaces/stubs needed for:

    audio_input.cpp
    wake_word.cpp
    speech_to_text.cpp
    assistant.cpp
    text_to_speech.cpp

    Do not implement the AI assistant yet.

11. Add any corresponding header files that are technically required for
    clean modular C++ code.

12. Build the PlatformIO project.

13. If the build fails, diagnose and fix the errors until the initial
    project builds successfully.

14. Do not add unnecessary libraries or functionality yet.

15. At the end, briefly tell me:
    - which files you created
    - whether the PlatformIO build succeeded
    - where I should copy hi_tech_panda.tflite

Do not only give me instructions or code snippets.

Actually create and configure the files in the current VS Code workspace.
```

After Codex completes the initial build, copy `stream_state_internal_quant.tflite` into the new project's `models` folder and rename it `hi_tech_panda.tflite`.

Alternative shown in the script: **VS Code → PlatformIO → New Project**, name `TechPandaVoiceAssistant`, board **Seeed Studio XIAO ESP32S3**, framework **Arduino**. Use either this manual route or the prompt above. The board identifier is `seeed_xiao_esp32s3`; Serial Monitor baud rate is 115200.

## 5. Deploy local detection — Codex prompt 2

Paste this into Codex in the same project:


```text
We already have a working PlatformIO project for the Seeed Studio XIAO ESP32-S3.

The project contains our trained microWakeWord model here:

models/hi_tech_panda.tflite

The wake phrase is:

"Hi Tech Panda"

Now implement ONLY the local wake-word detection stage.

Hardware:
- Seeed Studio XIAO ESP32-S3
- ReSpeaker XVF3800
- Audio input from the ReSpeaker over I2S

Requirements:

1. Use the existing PlatformIO project.
2. Do not create a new project.
3. Do not add Wi-Fi, Groq, speech-to-text, or text-to-speech yet.
4. Configure the ReSpeaker XVF3800 I2S audio input correctly.
5. Capture microphone audio continuously.
6. Integrate the trained microWakeWord TensorFlow Lite model from:
   models/hi_tech_panda.tflite
7. Inspect the model and existing microWakeWord requirements.
8. Do not guess the model input shape, sample rate, or feature format.
9. Convert the .tflite model into the required C/C++ representation for the firmware if needed.
10. Run inference continuously on the ESP32-S3.
11. When the wake phrase is detected, print:

   WAKE WORD DETECTED: Hi Tech Panda

   to the Serial Monitor.

12. Add useful startup messages for:
   - PSRAM
   - I2S initialization
   - model initialization
   - tensor allocation
   - detector ready

13. Keep the code modular using the existing project structure.
14. Build the project.
15. Fix all compile errors until the project builds successfully.
16. Do not replace or retrain the model.
17. At the end, tell me exactly what files were changed and how to upload and test the firmware.
```

The prompt requests model inspection and the correct feature pipeline. A successful compile alone does not demonstrate detection; verify using microphone input on the real board.

## 6. Build, upload, and test local detection

In the PlatformIO project terminal, compile:


```powershell
pio run
```

Connect the board by USB and upload:


```powershell
pio run -t upload
```

Open the Serial Monitor:


```powershell
pio device monitor
```

Confirm initialization, then say **“Hi Tech Panda”**. Expected detection message:


```text
WAKE WORD DETECTED: Hi Tech Panda
```

Repeat several times and test ordinary speech and background noise. Proceed after local detection works reliably. Stop the monitor with **Ctrl+C** before uploading again.

## 7. Add the conversational assistant — Codex prompt 3

Paste this into Codex in the same working project:


```text
The custom wake-word detection is now working correctly on the existing
PlatformIO project.

Current hardware:
- Seeed Studio XIAO ESP32-S3
- ReSpeaker XVF3800
- Custom microWakeWord model
- Wake phrase: "Hi Tech Panda"

Current behavior:
- The ESP32 continuously listens for the wake word.
- "Hi Tech Panda" is detected locally.
- Detection is confirmed in the Serial Monitor.

Do not replace or rewrite the working wake-word implementation.

Now extend the existing project into a conversational AI voice assistant.

Required flow:

Hi Tech Panda
    ↓
Wake word detected locally
    ↓
Start recording the user's question
    ↓
Detect end of speech / silence
    ↓
Speech-to-text
    ↓
Send the text to Groq
    ↓
Receive AI response
    ↓
Convert response to speech
    ↓
Play the response through the ReSpeaker
    ↓
Return to wake-word listening mode

Requirements:

1. Keep the existing wake-word detection unchanged.

2. Wake-word detection must remain fully local and offline.

3. After the wake word is detected, begin recording microphone audio.

4. Detect when the user stops speaking so recording ends automatically.

5. Use PSRAM for audio buffering where appropriate.

6. Add speech-to-text using a Whisper-compatible API.

7. Use Groq for the AI response.

8. Store Wi-Fi credentials and API keys in:
   include/secrets.h

9. Do not hard-code credentials in source files.

10. Add these Serial Monitor states:

    LISTENING FOR WAKE WORD
    WAKE WORD DETECTED
    LISTENING FOR QUESTION
    PROCESSING SPEECH
    TRANSCRIPTION: ...
    ASKING AI
    AI RESPONSE: ...
    SPEAKING
    RETURNING TO WAKE WORD MODE

11. Add text-to-speech so the AI response can be played through the
    ReSpeaker.

12. Prevent the speaker output from retriggering the wake-word detector
    while the assistant is speaking.

13. After the response finishes playing, automatically return to
    listening for "Hi Tech Panda".

14. Keep the existing modular project structure.

15. Add any required libraries to platformio.ini.

16. Build the project and fix all compilation errors.

17. Do not remove or break the working wake-word code.

18. At the end, tell me:
    - which files were changed
    - which libraries were added
    - what information I need to add to secrets.h
    - how to test the complete voice assistant

Example final interaction:

User:
"Hi Tech Panda"

Device:
Wake word detected

User:
"How far away is the Moon?"

Device:
"The Moon is about 384,000 kilometers away from Earth."

Then automatically return to listening for:
"Hi Tech Panda"
```

### Confirm the service configuration

The script specifies Groq for responses and a Whisper-compatible speech-to-text API, but does not specify exact API model IDs, endpoints, a TTS voice, or the generated audio format. Ask Codex to identify these from the final implementation and document them before running it. Use service-supported models and a playback format your firmware can decode.

The narration says the Groq key is used for all three stages. Confirm that this matches the actual providers chosen in the generated code; a different speech provider may require its own key. Do not assume the same key works for every service.

## 8. Configure credentials and test the final assistant

Open `include/secrets.h` and replace its placeholders with your Wi-Fi SSID, Wi-Fi password, Groq API key, and any other keys required by the actual implementation. Preserve the variable names Codex created.

Keep real credentials out of uploaded source files. Share a placeholder example file with viewers.

Compile, upload, and open the monitor, running commands separately:


```powershell
pio run
pio run -t upload
pio device monitor
```

Check Wi-Fi connection, audio initialization, model initialization, and the listening state. Say **“Hi Tech Panda”**, wait for the question-listening state, then ask **“How far away is the Moon?”**.

Expected flow:


```text
LISTENING FOR WAKE WORD
WAKE WORD DETECTED
LISTENING FOR QUESTION
PROCESSING SPEECH
TRANSCRIPTION: How far away is the Moon?
ASKING AI
AI RESPONSE: ...
SPEAKING
RETURNING TO WAKE WORD MODE
```

The speaker should play the answer, then the device should resume wake-word detection. Confirm that playback does not retrigger detection. Exact startup text and answer wording depend on the generated firmware and AI response.

## 9. Troubleshooting

| Problem | Action |
| --- | --- |
| `py -3.10` fails | Install Python 3.10, reopen the terminal, and repeat the version check. |
| `git` is unavailable | Install Git, reopen the terminal, and retry cloning. |
| Clone says the destination already exists | Open the existing repository if it is the intended copy, or choose a new empty parent folder. |
| PowerShell blocks environment activation | For this terminal session, run the command below, then activate again. |
| Notebook cannot import a package | Confirm the kernel is the project's `env`; install using `%pip` in that notebook. |
| Editable install `-e ..` fails | Confirm the notebook working directory is `microWakeWord/notebooks`. |
| Piper model or packages are missing | Complete the setup in cell 3.2; existing folders can cause its setup block to be skipped. |
| Dataset extraction fails | Remove the incomplete downloaded archive and rerun its download cell. |
| Feature-generation output already exists | Preserve the previous output and use a fresh feature folder when regenerating; update configuration paths consistently. |
| Training runs out of memory | Reduce batch size; training performance depends on your machine. |
| `pio` is unavailable | Use PlatformIO's integrated terminal or its Build, Upload, and Monitor buttons in VS Code. |
| Upload fails | Close Serial Monitor, check the USB data cable and selected port, and retry. |
| Model builds but does not detect | Inspect I2S format/channel, PCM levels, feature extraction, model tensor requirements, and detection thresholds. |
| Online requests fail | Check Wi-Fi, the configured provider/model, API credentials, quota, and Serial Monitor errors. |
| No spoken response | Verify TTS output format, decoding, playback I2S configuration, and physical speaker connection. |

Optional PowerShell activation fix, limited to the current process:


```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

## References

- Training repository: https://github.com/kahrendt/microWakeWord
- Original notebook: https://github.com/kahrendt/microWakeWord/blob/main/notebooks/basic_training_notebook.ipynb
- Piper sample generator: https://github.com/rhasspy/piper-sample-generator
- Piper checkpoint used in the script: https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt

The original notebook was retrieved while preparing this guide to fill the omitted cells. Repositories and dependencies can change; preserve the notebook and working firmware used for your own build.
