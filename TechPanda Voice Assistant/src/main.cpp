#include <Arduino.h>
#include "audio_input.h"
#include "wake_word.h"
#include "assistant.h"

void setup() {
  Serial.begin(115200);
  // The XIAO uses native USB CDC. Allow the monitor time to open after reset,
  // but continue booting if it is running standalone.
  const uint32_t serial_wait_start = millis();
  while (!Serial && millis() - serial_wait_start < 10000) {
    delay(10);
  }
  delay(200);
  Serial.println("Tech Panda Voice Assistant");
  Serial.printf("PSRAM: %s (%u bytes)\n", psramFound() ? "available" : "not detected",
                static_cast<unsigned int>(ESP.getPsramSize()));
  audio_input::begin();
  wake_word::begin();
  assistant::begin();
}

void loop() {
  audio_input::update();
  wake_word::update();
  assistant::update();
}
