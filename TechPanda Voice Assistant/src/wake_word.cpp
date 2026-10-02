#include "wake_word.h"

#include "audio_input.h"
#include "audio_config.h"
#include "assistant.h"
#include "wake_word_model.h"

#include <Arduino.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <new>
#include <esp_heap_caps.h>
#include <frontend_util.h>
#include <frontend.h>
#include <tensorflow/lite/micro/micro_allocator.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_resource_variable.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace {
constexpr size_t kFeatureCount = 40;
constexpr size_t kTensorArenaBytes = 64 * 1024;
constexpr size_t kVariableArenaBytes = 1024;
constexpr size_t kSlidingWindowSize = 5;
constexpr uint8_t kProbabilityCutoff = 247;  // 0.97 * 255, microWakeWord default
constexpr size_t kWarmupFeatureSlices = 100;
constexpr size_t kExpectedInputSlices = 3;
constexpr size_t kOpResolverCapacity = 20;

alignas(16) uint8_t *tensor_arena = nullptr;
uint8_t *variable_arena = nullptr;
tflite::MicroAllocator *variable_allocator = nullptr;
tflite::MicroResourceVariables *resource_variables = nullptr;
tflite::MicroMutableOpResolver<kOpResolverCapacity> op_resolver;
tflite::MicroInterpreter *interpreter = nullptr;
FrontendConfig frontend_config{};
FrontendState frontend_state{};
int8_t feature_window[kExpectedInputSlices * kFeatureCount]{};
uint8_t recent_probabilities[kSlidingWindowSize]{};
size_t model_input_slices = 0;
size_t current_slice = 0;
size_t probability_index = 0;
size_t feature_slices_seen = 0;
bool frontend_ready = false;
bool detector_ready = false;
bool wake_latched = false;
bool detection_enabled = true;

bool registerModelOps() {
  return op_resolver.AddCallOnce() == kTfLiteOk &&
         op_resolver.AddVarHandle() == kTfLiteOk &&
         op_resolver.AddReshape() == kTfLiteOk &&
         op_resolver.AddReadVariable() == kTfLiteOk &&
         op_resolver.AddStridedSlice() == kTfLiteOk &&
         op_resolver.AddConcatenation() == kTfLiteOk &&
         op_resolver.AddAssignVariable() == kTfLiteOk &&
         op_resolver.AddConv2D() == kTfLiteOk &&
         op_resolver.AddMul() == kTfLiteOk &&
         op_resolver.AddAdd() == kTfLiteOk &&
         op_resolver.AddMean() == kTfLiteOk &&
         op_resolver.AddFullyConnected() == kTfLiteOk &&
         op_resolver.AddLogistic() == kTfLiteOk &&
         op_resolver.AddQuantize() == kTfLiteOk &&
         op_resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         op_resolver.AddAveragePool2D() == kTfLiteOk &&
         op_resolver.AddMaxPool2D() == kTfLiteOk &&
         op_resolver.AddPad() == kTfLiteOk &&
         op_resolver.AddPack() == kTfLiteOk &&
         op_resolver.AddSplitV() == kTfLiteOk;
}

bool initializeFrontend() {
  FrontendFillConfigWithDefaults(&frontend_config);
  // Upstream microWakeWord frontend settings; these match its training pipeline.
  frontend_config.window.size_ms = 30;
  frontend_config.window.step_size_ms = 10;
  frontend_config.filterbank.num_channels = kFeatureCount;
  frontend_config.filterbank.lower_band_limit = 125.0f;
  frontend_config.filterbank.upper_band_limit = 7500.0f;
  frontend_config.noise_reduction.smoothing_bits = 10;
  frontend_config.noise_reduction.even_smoothing = 0.025f;
  frontend_config.noise_reduction.odd_smoothing = 0.06f;
  frontend_config.noise_reduction.min_signal_remaining = 0.05f;
  frontend_config.pcan_gain_control.enable_pcan = true;
  frontend_config.pcan_gain_control.strength = 0.95f;
  frontend_config.pcan_gain_control.offset = 80.0f;
  frontend_config.pcan_gain_control.gain_bits = 21;
  frontend_config.log_scale.enable_log = true;
  frontend_config.log_scale.scale_shift = 6;
  frontend_ready = FrontendPopulateState(&frontend_config, &frontend_state, audio_config::SAMPLE_RATE_HZ) != 0;
  return frontend_ready;
}

bool initializeInterpreter() {
  if (!registerModelOps()) {
    Serial.println("Model initialization failed: operator registration");
    return false;
  }
  const tflite::Model *model = tflite::GetModel(hi_tech_panda_model);
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model initialization failed: invalid TFLite model/schema");
    return false;
  }
  Serial.printf("Model initialization: OK (%u bytes, TFLite schema %d)\n",
                static_cast<unsigned int>(hi_tech_panda_model_len), model->version());

  tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(
      kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(
        kTensorArenaBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  }
  variable_arena = static_cast<uint8_t *>(heap_caps_malloc(
      kVariableArenaBytes, MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr || variable_arena == nullptr) {
    Serial.println("Tensor allocation failed: insufficient memory");
    return false;
  }
  variable_allocator = tflite::MicroAllocator::Create(variable_arena, kVariableArenaBytes);
  if (variable_allocator == nullptr) {
    Serial.println("Tensor allocation failed: variable allocator creation");
    return false;
  }
  resource_variables = tflite::MicroResourceVariables::Create(variable_allocator, 20);
  if (resource_variables == nullptr) {
    Serial.println("Tensor allocation failed: model state creation");
    return false;
  }
  Serial.printf("Tensor allocation: OK (tensor arena %u bytes, variable arena %u bytes)\n",
                static_cast<unsigned int>(kTensorArenaBytes),
                static_cast<unsigned int>(kVariableArenaBytes));

  interpreter = new (std::nothrow) tflite::MicroInterpreter(
      model, op_resolver, tensor_arena, kTensorArenaBytes, resource_variables);
  if (interpreter == nullptr || interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed: TFLite AllocateTensors");
    return false;
  }

  TfLiteTensor *input = interpreter->input(0);
  TfLiteTensor *output = interpreter->output(0);
  if (input == nullptr || output == nullptr || input->type != kTfLiteInt8 ||
      input->dims == nullptr || input->dims->size != 3 || input->dims->data[0] != 1 ||
      input->dims->data[2] != static_cast<int>(kFeatureCount) ||
      input->dims->data[1] != static_cast<int>(kExpectedInputSlices) ||
      input->params.zero_point != -128 || fabsf(input->params.scale - 0.1019607857f) > 0.00001f ||
      output->type != kTfLiteUInt8 || output->dims == nullptr || output->dims->size != 2 ||
      output->dims->data[0] != 1 || output->dims->data[1] != 1 ||
      output->params.zero_point != 0 || fabsf(output->params.scale - (1.0f / 256.0f)) > 0.000001f) {
    Serial.println("Model initialization failed: tensor metadata does not match the inspected model");
    return false;
  }
  model_input_slices = static_cast<size_t>(input->dims->data[1]);
  if (!initializeFrontend()) {
    Serial.println("microWakeWord frontend initialization failed");
    return false;
  }
  detector_ready = true;
  Serial.printf("Detector ready: Hi Tech Panda (input [1,%u,40], mono PCM %u Hz)\n",
                static_cast<unsigned int>(model_input_slices),
                static_cast<unsigned int>(audio_config::SAMPLE_RATE_HZ));
  return true;
}

void processFeature(const uint16_t *feature_values, size_t feature_count) {
  if (feature_count != kFeatureCount || interpreter == nullptr) return;
  int8_t *destination = feature_window + current_slice * kFeatureCount;
  for (size_t i = 0; i < kFeatureCount; ++i) {
    // microWakeWord training: frontend uint16 -> float feature / 25.6 / 26,
    // then int8 quantization (input scale 0.1019607857, zero point -128).
    int32_t quantized = ((static_cast<int32_t>(feature_values[i]) * 256) + 333) / 666 - 128;
    destination[i] = static_cast<int8_t>(std::max<int32_t>(-128, std::min<int32_t>(127, quantized)));
  }
  ++feature_slices_seen;
  ++current_slice;
  if (current_slice < model_input_slices) return;

  TfLiteTensor *input = interpreter->input(0);
  memcpy(input->data.int8, feature_window, model_input_slices * kFeatureCount);
  current_slice = 0;
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Wake-word inference failed");
    return;
  }

  const uint8_t probability = interpreter->output(0)->data.uint8[0];
  recent_probabilities[probability_index] = probability;
  probability_index = (probability_index + 1) % kSlidingWindowSize;
  if (feature_slices_seen < kWarmupFeatureSlices) return;

  uint16_t total = 0;
  uint8_t minimum = 255;
  for (uint8_t value : recent_probabilities) {
    total += value;
    minimum = std::min(minimum, value);
  }
  const bool detected = total > kProbabilityCutoff * kSlidingWindowSize;
  if (!detected && minimum < kProbabilityCutoff) wake_latched = false;
  if (detected && !wake_latched) {
    Serial.println("WAKE WORD DETECTED: Hi Tech Panda");
    wake_latched = true;
    assistant_on_wake_word();
  }
}
}

namespace wake_word {
void begin() {
  Serial.println("Model initialization: loading embedded microWakeWord model");
  if (!initializeInterpreter()) return;
}

void update() {
  if (!detector_ready) return;
  // The assistant owns these samples while recording a question.
  if (!detection_enabled) return;
  size_t sample_count = 0;
  const int16_t *samples = audio_input::availableSamples(sample_count);
  size_t offset = 0;
  while (offset < sample_count) {
    size_t consumed = 0;
    const FrontendOutput output = FrontendProcessSamples(
        &frontend_state, samples + offset, sample_count - offset, &consumed);
    if (consumed == 0) break;
    offset += consumed;
    if (output.size != 0) processFeature(output.values, output.size);
    // A wake callback can play audio and overwrite the shared sample buffer.
    if (!detection_enabled) break;
  }
  audio_input::consumeSamples(sample_count);
}
void setEnabled(bool enabled) {
  if (detection_enabled == enabled) return;
  detection_enabled = enabled;
  if (!enabled) {
    wake_latched = true;
  } else {
    // Drop scores/features from the previous turn so the detector rearms only
    // after it has observed fresh ambient audio following assistant playback.
    memset(recent_probabilities, 0, sizeof(recent_probabilities));
    memset(feature_window, 0, sizeof(feature_window));
    probability_index = 0;
    current_slice = 0;
    feature_slices_seen = 0;
    if (frontend_ready) FrontendReset(&frontend_state);
    if (resource_variables != nullptr) resource_variables->ResetAll();
    wake_latched = false;
  }
}
}
