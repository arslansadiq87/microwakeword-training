#pragma once
#include <Arduino.h>
#include <cstddef>
#include <cstdint>
namespace speech_to_text {
bool transcribe(const int16_t *pcm, size_t sample_count, String &transcript);
}
