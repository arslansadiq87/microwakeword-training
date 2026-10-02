#include "speech_to_text.h"
#include "audio_config.h"
#include "secrets.h"

#include <ArduinoJson.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_heap_caps.h>
#include <algorithm>

namespace {
void writeLe16(uint8_t *out, uint16_t value) { out[0] = value & 0xff; out[1] = (value >> 8) & 0xff; }
void writeLe32(uint8_t *out, uint32_t value) { for (uint8_t i = 0; i < 4; ++i) out[i] = (value >> (8 * i)) & 0xff; }
String jsonEscape(const String &value) {
  String out;
  for (size_t i = 0; i < value.length(); ++i) {
    const char c = value[i];
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if (static_cast<uint8_t>(c) >= 0x20) out += c;
  }
  return out;
}
}

namespace speech_to_text {
bool transcribe(const int16_t *pcm, size_t sample_count, String &transcript) {
  if (pcm == nullptr || sample_count == 0 || WiFi.status() != WL_CONNECTED) return false;
  WiFiClientSecure client;
  client.setInsecure(); // HTTPS transport; configure CA validation for production deployments.
  const String boundary = "----TechPandaVoiceBoundary";
  const String prefix = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"question.wav\"\r\nContent-Type: audio/wav\r\n\r\n";
  const String suffix = "\r\n--" + boundary + "--\r\n";
  const size_t data_size = sample_count * sizeof(int16_t);
  const size_t wav_size = 44 + data_size;
  const size_t total_size = prefix.length() + wav_size + suffix.length();
  if (!client.connect("api.groq.com", 443)) return false;
  String request = "POST /openai/v1/audio/transcriptions HTTP/1.1\r\nHost: api.groq.com\r\nAuthorization: Bearer ";
  request += GROQ_API_KEY;
  request += "\r\nContent-Type: multipart/form-data; boundary=" + boundary + "\r\nContent-Length: " + String(total_size) + "\r\nConnection: close\r\n\r\n";
  client.print(request);
  client.print(prefix);
  uint8_t wav_header[44] = {};
  memcpy(wav_header, "RIFF", 4); writeLe32(wav_header + 4, static_cast<uint32_t>(36 + data_size));
  memcpy(wav_header + 8, "WAVEfmt ", 8); writeLe32(wav_header + 16, 16); writeLe16(wav_header + 20, 1);
  writeLe16(wav_header + 22, 1); writeLe32(wav_header + 24, audio_config::SAMPLE_RATE_HZ);
  writeLe32(wav_header + 28, audio_config::SAMPLE_RATE_HZ * 2); writeLe16(wav_header + 32, 2);
  writeLe16(wav_header + 34, 16); memcpy(wav_header + 36, "data", 4); writeLe32(wav_header + 40, data_size);
  client.write(wav_header, sizeof(wav_header));
  const uint8_t *pcm_bytes = reinterpret_cast<const uint8_t *>(pcm);
  size_t sent = 0;
  while (sent < data_size) {
    const size_t chunk = std::min(static_cast<size_t>(1024), data_size - sent);
    const size_t written = client.write(pcm_bytes + sent, chunk);
    if (written == 0) { client.stop(); return false; }
    sent += written;
  }
  client.print(suffix);
  const uint32_t response_start = millis();
  while (!client.available() && client.connected() && millis() - response_start < 45000) delay(2);
  String response_body;
  int status_code = 0;
  bool first_line = true;
  bool body_started = false;
  while ((client.connected() || client.available()) && millis() - response_start < 45000) {
    String line = client.readStringUntil('\n');
    if (first_line) {
      first_line = false;
      const int first_space = line.indexOf(' ');
      if (first_space >= 0) status_code = line.substring(first_space + 1).toInt();
      continue;
    }
    if (line == "\r" || line.length() == 0) { body_started = true; continue; }
    if (body_started) response_body += line;
  }
  client.stop();
  if (status_code >= 200 && status_code < 300 && response_body.length() > 0) {
    DynamicJsonDocument doc(4096);
    const DeserializationError error = deserializeJson(doc, response_body);
    if (!error && doc["text"].is<const char *>()) transcript = jsonEscape(String(doc["text"].as<const char *>()));
  } else Serial.printf("Whisper HTTP error: %d\n", status_code);
  return transcript.length() != 0;
}
}
