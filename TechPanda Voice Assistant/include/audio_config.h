#pragma once

#include <driver/gpio.h>
#include <cstddef>
#include <cstdint>

// ReSpeaker XVF3800 I2S mode wiring for the XIAO ESP32-S3.
namespace audio_config {
constexpr gpio_num_t I2S_BCLK_PIN = GPIO_NUM_8;
constexpr gpio_num_t I2S_WS_PIN = GPIO_NUM_7;
constexpr gpio_num_t I2S_DATA_IN_PIN = GPIO_NUM_43;
constexpr gpio_num_t I2S_DATA_OUT_PIN = GPIO_NUM_44;
constexpr uint32_t SAMPLE_RATE_HZ = 16000;
constexpr uint8_t CHANNEL_COUNT = 2;
constexpr size_t BLOCK_FRAMES = 160;  // 10 ms at the configured sample rate
}
