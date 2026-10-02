#include "assistant.h"
#include "audio_input.h"
#include "audio_config.h"
#include "wake_word.h"
#include "speech_to_text.h"
#include "text_to_speech.h"
#include "secrets.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_heap_caps.h>
#include <algorithm>
#include <cstring>

namespace {
enum class State { Wake, Record, Process, Think, Speak };
State state = State::Wake;
int16_t *recording = nullptr;
size_t samples_used = 0;
size_t samples_capacity = 0;
uint32_t last_voice_ms = 0;
uint32_t record_start_ms = 0;
bool has_voice_activity = false;
uint8_t voice_blocks = 0;
uint32_t next_record_log_ms = 0;
constexpr uint32_t kSilenceMs = 1300;
constexpr uint32_t kQuestionStartDelayMs = 700;
constexpr uint32_t kNoSpeechTimeoutMs = 8000;
constexpr uint32_t kMaxRecordMs = 12000;
constexpr size_t kMinSamples = audio_config::SAMPLE_RATE_HZ / 3;
constexpr size_t kVoiceLevelThreshold = 650;
constexpr uint8_t kVoiceBlocksToStart = 5;

void returnToWake() {
  if (recording) heap_caps_free(recording);
  recording = nullptr;
  samples_used = samples_capacity = 0;
  state = State::Wake;
  wake_word::setEnabled(true);
  Serial.println("RETURNING TO WAKE WORD MODE");
  Serial.println("LISTENING FOR WAKE WORD");
}

void freeRecording() { if (recording) heap_caps_free(recording); recording = nullptr; samples_used = samples_capacity = 0; }
bool allocateRecording() {
  samples_capacity = audio_config::SAMPLE_RATE_HZ * (kMaxRecordMs / 1000);
  recording = static_cast<int16_t *>(heap_caps_malloc(samples_capacity * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!recording) recording = static_cast<int16_t *>(malloc(samples_capacity * sizeof(int16_t)));
  return recording != nullptr;
}
bool isRetryableHttpStatus(int status) {
  // Transport failures are negative; retry rate limits and server-side errors.
  return status < 0 || status == 408 || status == 429 || status >= 500;
}
bool askGroq(const String &question, String &answer) {
  DynamicJsonDocument request(6144);
  request["model"] = "openai/gpt-oss-20b";
  request["temperature"] = 0.4;
  request["max_tokens"] = 180;
  JsonArray messages = request.createNestedArray("messages");
  JsonObject system = messages.createNestedObject();
  system["role"] = "system";
  system["content"] = "You are Tech Panda, a concise voice assistant. Answer naturally in one or two short sentences.";
  JsonObject user = messages.createNestedObject(); user["role"] = "user"; user["content"] = question;
  String body; serializeJson(request, body);

  constexpr uint8_t kMaxAttempts = 3;
  for (uint8_t attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.printf("Groq chat attempt %u/%u skipped: Wi-Fi disconnected.\n", attempt, kMaxAttempts);
      if (attempt < kMaxAttempts) { WiFi.reconnect(); delay(1500 * attempt); continue; }
      return false;
    }

    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;
    http.setConnectTimeout(10000);
    http.setTimeout(45000);
    if (!http.begin(client, "https://api.groq.com/openai/v1/chat/completions")) {
      Serial.printf("Groq chat attempt %u/%u: could not initialize HTTPS request.\n", attempt, kMaxAttempts);
      if (attempt < kMaxAttempts) { delay(1000 * attempt); continue; }
      return false;
    }
    http.addHeader("Authorization", String("Bearer ") + GROQ_API_KEY);
    http.addHeader("Content-Type", "application/json");
    const int status = http.POST(body);
    if (status >= 200 && status < 300) {
      String response_body = http.getString();
      DynamicJsonDocument response(8192);
      const DeserializationError error = deserializeJson(response, response_body);
      if (!error && response["choices"][0]["message"]["content"].is<const char *>()) {
        answer = response["choices"][0]["message"]["content"].as<String>();
        answer.trim();
        http.end();
        if (answer.length()) return true;
        Serial.println("Groq chat returned an empty answer.");
      } else {
        Serial.printf("Groq chat response could not be parsed: %s\n", error ? error.c_str() : "missing content field");
      }
      http.end();
      // Invalid or unexpected success payloads are unlikely to improve on retry.
      return false;
    }

    Serial.printf("Groq chat HTTP status: %d (attempt %u/%u)\n", status, attempt, kMaxAttempts);
    http.end();
    if (!isRetryableHttpStatus(status)) {
      if (status == 401 || status == 403) Serial.println("Check the Groq API key and account access.");
      else if (status == 400 || status == 404) Serial.println("Check the chat model name and request parameters.");
      return false;
    }
    if (attempt < kMaxAttempts) delay(1200 * attempt);
  }
  return false;
}
}

namespace assistant {
void begin() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to Wi-Fi SSID: %s\n", WIFI_SSID);
  const uint32_t started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 30000) { delay(250); Serial.print('.'); }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Wi-Fi connected; IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.printf("Wi-Fi failed (status %d); online assistant unavailable. Check SSID/password and hotspot compatibility.\n",
                  static_cast<int>(WiFi.status()));
  }
  state = State::Wake;
  Serial.println("LISTENING FOR WAKE WORD");
}

void update() {
  static uint32_t next_wifi_retry = 0;
  if (WiFi.status() != WL_CONNECTED && static_cast<int32_t>(millis() - next_wifi_retry) >= 0) {
    next_wifi_retry = millis() + 15000;
    Serial.printf("Wi-Fi disconnected (status %d); retrying SSID %s\n",
                  static_cast<int>(WiFi.status()), WIFI_SSID);
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
  if (state == State::Wake) return;
  if (state == State::Record) {
    size_t count = 0;
    const int16_t *block = audio_input::availableSamples(count);
    if (block != nullptr && count) {
      size_t energy = 0;
      for (size_t i = 0; i < count; ++i) energy += abs(static_cast<int>(block[i]));
      const size_t level = energy / count;
      const bool eligible_for_voice = millis() - record_start_ms >= kQuestionStartDelayMs;
      if (eligible_for_voice && level >= kVoiceLevelThreshold) {
        if (voice_blocks < kVoiceBlocksToStart) ++voice_blocks;
      } else {
        voice_blocks = 0;
      }
      const bool voice = has_voice_activity
          ? (eligible_for_voice && level >= kVoiceLevelThreshold)
          : (voice_blocks >= kVoiceBlocksToStart);
      if (voice) {
        if (!has_voice_activity) Serial.printf("QUESTION VOICE DETECTED (level=%u)\n", static_cast<unsigned>(level));
        last_voice_ms = millis();
        has_voice_activity = true;
      }
      if (eligible_for_voice) {
        const size_t copy_count = std::min(count, samples_capacity - samples_used);
        memcpy(recording + samples_used, block, copy_count * sizeof(int16_t)); samples_used += copy_count;
      }
      audio_input::consumeSamples(count);
      if (static_cast<int32_t>(millis() - next_record_log_ms) >= 0) {
        next_record_log_ms = millis() + 1000;
        Serial.printf("Question mic: level=%u, recorded=%u ms, voice=%s\n",
                      static_cast<unsigned>(level),
                      static_cast<unsigned>(samples_used * 1000 / audio_config::SAMPLE_RATE_HZ),
                      has_voice_activity ? "yes" : "waiting");
      }
    }
    const uint32_t elapsed = millis() - record_start_ms;
    if (!has_voice_activity && elapsed >= kNoSpeechTimeoutMs) {
      Serial.printf("No question speech detected after 8 seconds (%u samples captured).\n",
                    static_cast<unsigned>(samples_used));
      returnToWake();
      return;
    }
    if ((has_voice_activity && millis() - last_voice_ms >= kSilenceMs && samples_used >= kMinSamples) ||
        elapsed >= kMaxRecordMs || samples_used >= samples_capacity) {
      Serial.printf("Question recording complete: %u samples, %u ms elapsed.\n",
                    static_cast<unsigned>(samples_used), static_cast<unsigned>(elapsed));
      state = State::Process;
    }
    return;
  }
  if (state == State::Process) {
    Serial.println("PROCESSING SPEECH");
    String transcript;
    if (samples_used < kMinSamples || !speech_to_text::transcribe(recording, samples_used, transcript)) {
      Serial.println("Speech transcription failed or was empty.");
      freeRecording(); state = State::Wake; wake_word::setEnabled(true); Serial.println("RETURNING TO WAKE WORD MODE"); Serial.println("LISTENING FOR WAKE WORD"); return;
    }
    Serial.print("TRANSCRIPTION: "); Serial.println(transcript);
    Serial.println("ASKING AI"); state = State::Think;
    String answer;
    if (!askGroq(transcript, answer)) {
      answer = "I'm having trouble reaching the AI service right now. Please try again in a moment.";
    }
    Serial.print("AI RESPONSE: "); Serial.println(answer);
    freeRecording();
    Serial.println("SPEAKING"); state = State::Speak;
    if (!text_to_speech::speak(answer)) Serial.println("TTS playback failed.");
    Serial.println("RETURNING TO WAKE WORD MODE");
    state = State::Wake;
    wake_word::setEnabled(true);
    Serial.println("LISTENING FOR WAKE WORD");
  }
}

void onWakeWord() {
  if (state != State::Wake) return;
  Serial.println("WAKE WORD DETECTED");
  if (WiFi.status() != WL_CONNECTED) { Serial.println("Wi-Fi disconnected; cannot process question."); return; }
  if (!allocateRecording()) { Serial.println("Question audio allocation failed."); return; }
  samples_used = 0;
  has_voice_activity = false;
  voice_blocks = 0;
  wake_word::setEnabled(false);
  Serial.println("SPEAKING: Wake acknowledgement");
  if (!text_to_speech::speak("Yes, I’m listening.", false)) {
    Serial.println("Wake acknowledgement speech failed; continuing silently.");
  }
  size_t stale_samples = 0;
  audio_input::availableSamples(stale_samples);
  audio_input::consumeSamples(stale_samples);
  record_start_ms = millis();
  next_record_log_ms = record_start_ms;
  last_voice_ms = record_start_ms;
  state = State::Record;
  Serial.println("WAITING FOR QUESTION SPEECH");
}
}

// Wake-word module calls this only when its local detector fires.
extern "C" void assistant_on_wake_word() { assistant::onWakeWord(); }
