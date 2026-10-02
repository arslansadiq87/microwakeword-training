#include "text_to_speech.h"
#include "audio_input.h"
#include "audio_config.h"
#include "secrets.h"
#include "wake_word.h"

#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_heap_caps.h>
#include <algorithm>
#include <cstring>

namespace {
constexpr char kGroqTtsUrl[] = "https://api.groq.com/openai/v1/audio/speech";
constexpr size_t kMaxWavBytes = 1024 * 1024;

class PsramBufferStream final : public Stream {
 public:
  explicit PsramBufferStream(size_t capacity)
      : data_(static_cast<uint8_t *>(heap_caps_malloc(capacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT))),
        capacity_(capacity) {}
  ~PsramBufferStream() override { if (data_) heap_caps_free(data_); }
  bool valid() const { return data_ != nullptr; }
  size_t size() const { return size_; }
  const uint8_t *data() const { return data_; }
  int available() override { return static_cast<int>(size_ - position_); }
  int read() override { return position_ < size_ ? data_[position_++] : -1; }
  int peek() override { return position_ < size_ ? data_[position_] : -1; }
  void flush() override {}
  size_t write(uint8_t byte) override { return write(&byte, 1); }
  size_t write(const uint8_t *buffer, size_t length) override {
    if (!data_ || length > capacity_ - size_) return 0;
    memcpy(data_ + size_, buffer, length);
    size_ += length;
    return length;
  }
 private:
  uint8_t *data_ = nullptr;
  size_t capacity_ = 0;
  size_t size_ = 0;
  size_t position_ = 0;
};

uint16_t read16(const uint8_t *p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t read32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

bool playWav(const uint8_t *wav, size_t size) {
  if (!wav || size < 12 || memcmp(wav, "RIFF", 4) != 0 || memcmp(wav + 8, "WAVE", 4) != 0) {
    Serial.println("Groq TTS response is not a RIFF/WAVE file.");
    return false;
  }
  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t sample_rate = 0;
  const uint8_t *audio_data = nullptr;
  size_t audio_bytes = 0;
  for (size_t offset = 12; offset + 8 <= size;) {
    const uint32_t chunk_size = read32(wav + offset + 4);
    const size_t chunk_start = offset + 8;
    const size_t available = size - chunk_start;
    const bool data_chunk = memcmp(wav + offset, "data", 4) == 0;
    // Groq's streamed RIFF responses use 0xFFFFFFFF to mean the data size is unknown.
    if (data_chunk && (chunk_size == 0xFFFFFFFFu || chunk_size > available)) {
      audio_data = wav + chunk_start;
      audio_bytes = available & ~size_t(1);
      break;
    }
    if (chunk_size > available) break;
    if (memcmp(wav + offset, "fmt ", 4) == 0 && chunk_size >= 16) {
      format = read16(wav + chunk_start);
      channels = read16(wav + chunk_start + 2);
      sample_rate = read32(wav + chunk_start + 4);
      bits = read16(wav + chunk_start + 14);
    } else if (data_chunk) {
      audio_data = wav + chunk_start;
      audio_bytes = chunk_size;
    }
    const size_t next = chunk_start + chunk_size + (chunk_size & 1u);
    if (next <= offset || next > size) break;
    offset = next;
  }
  if (format != 1 || channels != 1 || bits != 16 || sample_rate == 0 || !audio_data || audio_bytes < 2) {
    Serial.printf("Unsupported Groq WAV: PCM format %u, %u Hz, %u channel(s), %u bit(s)\n",
                  format, sample_rate, channels, bits);
    return false;
  }

  const size_t source_samples = audio_bytes / 2;
  const size_t output_samples = (source_samples * audio_config::SAMPLE_RATE_HZ) / sample_rate;
  int16_t output[256];
  size_t produced = 0;
  for (size_t first = 0; first < output_samples;) {
    const size_t block = std::min(static_cast<size_t>(256), output_samples - first);
    for (size_t i = 0; i < block; ++i) {
      const uint64_t position = uint64_t(first + i) * sample_rate;
      const size_t source_index = static_cast<size_t>(position / audio_config::SAMPLE_RATE_HZ);
      const uint32_t fraction = static_cast<uint32_t>(position % audio_config::SAMPLE_RATE_HZ);
      const size_t next_index = std::min(source_index + 1, source_samples - 1);
      const int16_t a = static_cast<int16_t>(read16(audio_data + source_index * 2));
      const int16_t b = static_cast<int16_t>(read16(audio_data + next_index * 2));
      output[i] = static_cast<int16_t>(a + (int32_t(b - a) * fraction) / audio_config::SAMPLE_RATE_HZ);
    }
    if (!audio_input::writePcm(output, block)) {
      Serial.println("I2S write failed during Groq TTS playback.");
      return false;
    }
    produced += block;
    first += block;
  }
  Serial.printf("Groq TTS WAV played: %u Hz mono, %u source samples, %u output samples\n",
                static_cast<unsigned>(sample_rate), static_cast<unsigned>(source_samples),
                static_cast<unsigned>(produced));
  return produced > 0;
}
}

namespace text_to_speech {
bool speak(const String &text, bool resume_wake_detection) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Groq TTS unavailable: Wi-Fi is disconnected.");
    return false;
  }
  DynamicJsonDocument request(2048);
  request["model"] = "canopylabs/orpheus-v1-english";
  request["voice"] = "hannah";
  request["input"] = text;
  request["response_format"] = "wav";
  String body;
  serializeJson(request, body);

  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.setConnectTimeout(10000);
  http.setTimeout(60000);
  if (!http.begin(client, kGroqTtsUrl)) return false;
  http.addHeader("Authorization", String("Bearer ") + GROQ_API_KEY);
  http.addHeader("Content-Type", "application/json");
  const int status = http.POST(body);
  bool played = false;
  if (status >= 200 && status < 300) {
    const int response_size = http.getSize();
    const size_t capacity = response_size > 0 ? static_cast<size_t>(response_size) : kMaxWavBytes;
    if (capacity > 44 && capacity <= kMaxWavBytes) {
      PsramBufferStream wav(capacity);
      if (wav.valid() && http.writeToStream(&wav) >= 0 && wav.size() > 44) {
        // Stop mic capture and wake inference while audio is being sent to the speaker.
        wake_word::setEnabled(false);
        if (audio_input::beginPlayback()) {
          played = playWav(wav.data(), wav.size());
          audio_input::endPlayback();
          // Let the final DMA frames and acoustic tail clear while wake
          // inference remains disabled; consume the queued mic samples too.
          const uint32_t settle_started = millis();
          while (millis() - settle_started < 350) {
            audio_input::update();
            delay(10);
          }
        } else {
          Serial.println("Could not switch I2S to speaker playback.");
        }
        if (resume_wake_detection) wake_word::setEnabled(true);
      } else Serial.println("Groq TTS returned an empty or incomplete WAV response.");
    } else Serial.printf("Groq TTS WAV size is unsupported: %d bytes\n", response_size);
  } else {
    Serial.printf("Groq TTS HTTP error: %d\n", status);
    Serial.println(http.getString());
  }
  http.end();
  return played;
}
}
