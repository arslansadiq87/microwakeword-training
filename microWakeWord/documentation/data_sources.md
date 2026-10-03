# Data Sources for Training Wake Words

The training notebook downloads audio and precomputed spectrogram features at runtime. These resources are not part of this Git repository. Review the upstream source and its current terms before downloading, using, or redistributing them; a dataset mirror's terms may not cover each audio recording.

## Positive Wake-Word Samples

[Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator) and the voice model selected by the notebook generate positive examples of the target phrase. Check the generator and voice-model license terms before sharing the generated audio or a model trained from it. Other microWakeWord recipes may also use adversarial phrases from [openWakeWord](https://github.com/dscripka/openWakeWord).

## Audio Downloaded by This Repository's Notebook

- [MIT Environmental Impulse Responses](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) supplies room responses for reverb augmentation. The current Hugging Face dataset card reports that its license is unknown; check the linked [original MIT source](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) for applicable terms.
- [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) supplies the `bal_train09.tar` background-audio archive that the notebook converts to 16 kHz WAV. The Hugging Face mirror currently labels its dataset `CC-BY-4.0`; review the card and upstream dataset conditions before reuse.
- [Free Music Archive (FMA)](https://github.com/mdeff/fma) supplies music. The notebook downloads an [FMA XSmall mirror](https://huggingface.co/datasets/mchl914/fma_xsmall) and converts its MP3 tracks to 16 kHz WAV. FMA tracks can have different Creative Commons terms; check the track metadata and terms for the specific audio you use.
- [Precomputed microWakeWord negative features](https://huggingface.co/datasets/kahrendt/microwakeword) provide the `speech`, `dinner_party`, `no_speech`, and `dinner_party_eval` sets used by the notebook. The dataset card currently states `CC-BY-NC-4.0`; check the card for the terms and feature-set details.

The augmentation code applies pitch, EQ, distortion, gain, background noise, and room impulse responses. It resamples background audio to 16 kHz WAV for processing.

## Other Sources Used by microWakeWord Recipes

Other experiments and pretrained models described by microWakeWord may use:

- [FSD50K](https://arxiv.org/abs/2010.00475) for labeled environmental sounds; source files have differing Creative Commons licenses.
- [WHAM!](https://arxiv.org/abs/1907.01160) for noisy speech/background audio; review its noncommercial license terms.
- [VOiCES](https://arxiv.org/abs/1804.05053), [Common Voice](https://commonvoice.mozilla.org/), and [DiPCo](https://www.amazon.science/publications/dipco-dinner-party-corpus) for speech and ambient evaluation, subject to their respective licenses and data agreements.
- [BIRD room impulse responses](https://arxiv.org/abs/2010.09930) for reverberation in other augmentation pipelines.

The sources and exact splits vary by experiment. Keep training, validation, and test recordings separate to avoid measuring performance on audio that the model has already seen.
