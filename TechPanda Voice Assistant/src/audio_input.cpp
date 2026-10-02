#include "audio_input.h"
#include "audio_config.h"

#include <Arduino.h>
#include <driver/i2s_std.h>
#include <algorithm>

namespace {
i2s_chan_handle_t rx_channel = nullptr;
i2s_chan_handle_t tx_channel = nullptr;
int32_t raw_stereo[audio_config::BLOCK_FRAMES * audio_config::CHANNEL_COUNT];
int16_t mono_samples[audio_config::BLOCK_FRAMES];
size_t samples_ready = 0;
bool rx_enabled = false;
bool tx_enabled = false;
bool playback_active = false;
int32_t silent_stereo[audio_config::BLOCK_FRAMES * audio_config::CHANNEL_COUNT] = {};
}

namespace audio_input {
void begin() {
  i2s_chan_config_t channel_config = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  channel_config.auto_clear = true;
  channel_config.dma_desc_num = 6;
  channel_config.dma_frame_num = audio_config::BLOCK_FRAMES;
  esp_err_t error = i2s_new_channel(&channel_config, &tx_channel, &rx_channel);
  if (error != ESP_OK) {
    Serial.printf("I2S channel creation failed: %s\n", esp_err_to_name(error));
    return;
  }

  i2s_std_config_t standard_config = {};
  standard_config.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(audio_config::SAMPLE_RATE_HZ);
  standard_config.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
      I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
  standard_config.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT;
  standard_config.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;
  standard_config.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  standard_config.gpio_cfg.bclk = audio_config::I2S_BCLK_PIN;
  standard_config.gpio_cfg.ws = audio_config::I2S_WS_PIN;
  standard_config.gpio_cfg.din = audio_config::I2S_DATA_IN_PIN;
  standard_config.gpio_cfg.dout = audio_config::I2S_DATA_OUT_PIN;
  standard_config.gpio_cfg.invert_flags = {};

  error = i2s_channel_init_std_mode(rx_channel, &standard_config);
  if (error == ESP_OK) error = i2s_channel_init_std_mode(tx_channel, &standard_config);
  if (error == ESP_OK) { error = i2s_channel_enable(rx_channel); rx_enabled = error == ESP_OK; }
  if (error == ESP_OK) { error = i2s_channel_enable(tx_channel); tx_enabled = error == ESP_OK; }
  if (error != ESP_OK) {
    Serial.printf("I2S initialization failed: %s\n", esp_err_to_name(error));
    i2s_del_channel(rx_channel);
    i2s_del_channel(tx_channel);
    rx_channel = nullptr;
    return;
  }
  Serial.println("I2S initialization: OK (16 kHz, 32-bit stereo)");
}

void update() {
  samples_ready = 0;
  if (rx_channel == nullptr) return;
  // The XVF3800 reference path writes silent TX frames while listening to
  // keep the shared master I2S clocks running for the codec and microphone.
  if (tx_enabled && !playback_active) {
    size_t silent_written = 0;
    const esp_err_t tx_error = i2s_channel_write(tx_channel, silent_stereo, sizeof(silent_stereo),
                                                  &silent_written, pdMS_TO_TICKS(20));
    if (tx_error != ESP_OK) Serial.printf("I2S silent-clock write failed: %s\n", esp_err_to_name(tx_error));
  }
  size_t bytes_read = 0;
  const esp_err_t error = i2s_channel_read(rx_channel, raw_stereo, sizeof(raw_stereo),
                                           &bytes_read, pdMS_TO_TICKS(20));
  if (error != ESP_OK || bytes_read == 0) return;
  const size_t frames_read = bytes_read / (sizeof(int32_t) * audio_config::CHANNEL_COUNT);
  for (size_t i = 0; i < frames_read; ++i) {
    // XVF3800 PCM occupies the high 16 bits of its 32-bit I2S slots.
    mono_samples[i] = static_cast<int16_t>(raw_stereo[i * audio_config::CHANNEL_COUNT] >> 16);
  }
  samples_ready = frames_read;
}

const int16_t *availableSamples(size_t &sample_count) {
  sample_count = samples_ready;
  return mono_samples;
}

void consumeSamples(size_t sample_count) {
  (void) sample_count;
  samples_ready = 0;
}

bool readBlock(int16_t *destination, size_t capacity, size_t &sample_count) {
  size_t bytes_read = 0;
  sample_count = 0;
  if (destination == nullptr || capacity == 0 || rx_channel == nullptr) return false;
  const size_t frames_to_read = std::min(capacity, static_cast<size_t>(audio_config::BLOCK_FRAMES));
  const esp_err_t error = i2s_channel_read(rx_channel, raw_stereo,
      frames_to_read * audio_config::CHANNEL_COUNT * sizeof(int32_t), &bytes_read, pdMS_TO_TICKS(50));
  if (error != ESP_OK || bytes_read == 0) return false;
  const size_t frames = bytes_read / (sizeof(int32_t) * audio_config::CHANNEL_COUNT);
  for (size_t i = 0; i < frames; ++i) destination[i] = static_cast<int16_t>(raw_stereo[i * audio_config::CHANNEL_COUNT] >> 16);
  sample_count = frames;
  return true;
}

bool beginPlayback() {
  if (tx_channel == nullptr || rx_channel == nullptr) return false;
  if (rx_enabled) {
    const esp_err_t error = i2s_channel_disable(rx_channel);
    if (error != ESP_OK) return false;
    rx_enabled = false;
  }
  if (!tx_enabled) {
    const esp_err_t error = i2s_channel_enable(tx_channel);
    if (error != ESP_OK) return false;
    tx_enabled = true;
  }
  playback_active = true;
  return true;
}

void endPlayback() {
  playback_active = false;
  if (rx_channel != nullptr && !rx_enabled) {
    if (i2s_channel_enable(rx_channel) == ESP_OK) rx_enabled = true;
  }
}

bool writePcm(const int16_t *samples, size_t sample_count) {
  if (tx_channel == nullptr || samples == nullptr) return false;
  int32_t stereo[audio_config::BLOCK_FRAMES * audio_config::CHANNEL_COUNT];
  size_t offset = 0;
  while (offset < sample_count) {
    const size_t frames = std::min(static_cast<size_t>(audio_config::BLOCK_FRAMES), sample_count - offset);
    for (size_t i = 0; i < frames; ++i) {
      // Multiplication preserves the signed 16-bit sample in the high half
      // of each XVF3800 32-bit slot without shifting a negative signed value.
      const int32_t sample = static_cast<int32_t>(samples[offset + i]) * 65536;
      stereo[2 * i] = sample;
      stereo[2 * i + 1] = sample;
    }
    size_t bytes_written = 0;
    const esp_err_t error = i2s_channel_write(tx_channel, stereo,
        frames * audio_config::CHANNEL_COUNT * sizeof(int32_t), &bytes_written, portMAX_DELAY);
    if (error != ESP_OK || bytes_written == 0) return false;
    offset += bytes_written / (audio_config::CHANNEL_COUNT * sizeof(int32_t));
  }
  return true;
}
}
