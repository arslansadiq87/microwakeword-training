#pragma once
#include <cstddef>
#include <cstdint>

namespace audio_input {
void begin();
void update();
const int16_t *availableSamples(size_t &sample_count);
void consumeSamples(size_t sample_count);
bool readBlock(int16_t *destination, size_t capacity, size_t &sample_count);
bool writePcm(const int16_t *samples, size_t sample_count);
bool beginPlayback();
void endPlayback();
}
