#pragma once

/**
 * wake_word.h — microWakeWord integration for onju-v2 firmware
 *
 * Detects "OK Nabu" wake word using TensorFlow Lite Micro + the
 * esp-micro-speech-features frontend (same preprocessing as ESPHome).
 *
 * Architecture:
 *   - Runs as a FreeRTOS task on Core 1 (low priority)
 *   - Reads I2S microphone ONLY when micTask is inactive
 *     (mic_timeout expired AND no playback)
 *   - On detection: sets mic_timeout + LED feedback
 *   - Respects mute switch (GPIO38)
 *
 * Dependencies:
 *   - TensorFlowLite_ESP32 (Arduino library)
 *   - esp-micro-speech-features (local lib/)
 *   - okay_nabu_model.h (generated from okay_nabu.tflite)
 */

#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// TFLite Micro
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_resource_variable.h>
#include <tensorflow/lite/micro/micro_allocator.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Audio frontend (same preprocessing as ESPHome microWakeWord)
#include "microfrontend/frontend.h"
#include "microfrontend/frontend_util.h"

// Generated model data
#include "okay_nabu_model.h"

// ============================================================
// Preprocessor settings — MUST match training parameters
// ============================================================
#define PREPROCESSOR_FEATURE_SIZE  40
#define FEATURE_DURATION_MS        30
#define FEATURE_STEP_SIZE_MS       10

// ============================================================
// Wake word detection parameters
// ============================================================
#define WW_PROBABILITY_CUTOFF      0.5f     // 0.0–1.0, quantized to 0–255
#define WW_SLIDING_WINDOW_SIZE     10       // number of recent probabilities to average
#define WW_MIN_SLICES_BEFORE_DETECTION 100  // cooldown after detection
#define WW_TENSOR_ARENA_SIZE       60000    // bytes for TFLite tensor arena

// ============================================================
// Audio capture parameters
// ============================================================
#define WW_SAMPLE_RATE             16000
#define WW_CHUNK_SIZE_MS           32       // process 32ms chunks
#define WW_CHUNK_SAMPLES           (WW_SAMPLE_RATE * WW_CHUNK_SIZE_MS / 1000)  // 512

// ============================================================
// External globals (defined in onjuino.ino)
// ============================================================
extern i2s_config_t i2s_config;
extern i2s_pin_config_t pin_config;
extern volatile uint32_t mic_timeout;
extern volatile bool isPlaying;
extern volatile bool deviceEnabled;
extern volatile uint8_t ledColor[3];
extern volatile uint16_t ledLevel;
extern volatile uint8_t ledFade;
extern const unsigned long MIC_LISTEN_MS;

// Mute pin
#define MUTE_PIN 38

// ============================================================
// Wake word state
// ============================================================
volatile bool wakeWordEnabled = true;
volatile bool wakeWordDetected = false;

// ============================================================
// TFLite model + interpreter (allocated in PSRAM)
// ============================================================
static tflite::MicroMutableOpResolver<20> *ww_op_resolver = nullptr;
static tflite::MicroInterpreter *ww_interpreter = nullptr;
static uint8_t *ww_tensor_arena = nullptr;
static uint8_t *ww_var_arena = nullptr;
static tflite::MicroAllocator *ww_ma = nullptr;
static tflite::MicroResourceVariables *ww_mrv = nullptr;

// ============================================================
// Audio frontend state
// ============================================================
static FrontendConfig ww_frontend_config;
static FrontendState ww_frontend_state;
static bool ww_frontend_initialized = false;

// ============================================================
// Sliding window for probability averaging
// ============================================================
static uint8_t ww_recent_probs[WW_SLIDING_WINDOW_SIZE] = {0};
static size_t ww_prob_index = 0;
static int16_t ww_ignore_windows = -WW_MIN_SLICES_BEFORE_DETECTION;
static uint8_t ww_current_stride = 0;
static uint8_t ww_model_stride = 0;  // set from model input dims

// ============================================================
// I2S buffer for wake word audio capture
// ============================================================
static int32_t ww_mic_buffer[WW_CHUNK_SAMPLES];
static int16_t ww_audio_buffer[WW_CHUNK_SAMPLES];

// ============================================================
// Helper: set LED (delegates to onjuino.ino's setLed)
// ============================================================
extern void setLed(uint8_t r, uint8_t g, uint8_t b, uint8_t level, uint8_t fade);

// ============================================================
// Initialize the TFLite model and frontend
// Returns true on success
// ============================================================
static bool wakeWordInit() {
    // --- Allocate variable arena for resource variables ---
    // TFLite Micro needs a separate arena for resource variables
    #define WW_VAR_ARENA_SIZE 1024
    ww_var_arena = (uint8_t *)heap_caps_malloc(WW_VAR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ww_var_arena) {
        Serial.println("[WW] ERROR: Failed to allocate variable arena");
        return false;
    }

    ww_ma = tflite::MicroAllocator::Create(ww_var_arena, WW_VAR_ARENA_SIZE);
    ww_mrv = tflite::MicroResourceVariables::Create(ww_ma, 20);

    // --- Allocate tensor arena ---
    ww_tensor_arena = (uint8_t *)heap_caps_malloc(WW_TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ww_tensor_arena) {
        Serial.println("[WW] ERROR: Failed to allocate tensor arena");
        return false;
    }

    // --- Register ops ---
    ww_op_resolver = new tflite::MicroMutableOpResolver<20>();
    ww_op_resolver->AddCallOnce();
    ww_op_resolver->AddVarHandle();
    ww_op_resolver->AddReshape();
    ww_op_resolver->AddReadVariable();
    ww_op_resolver->AddStridedSlice();
    ww_op_resolver->AddConcatenation();
    ww_op_resolver->AddAssignVariable();
    ww_op_resolver->AddConv2D();
    ww_op_resolver->AddMul();
    ww_op_resolver->AddAdd();
    ww_op_resolver->AddMean();
    ww_op_resolver->AddFullyConnected();
    ww_op_resolver->AddLogistic();
    ww_op_resolver->AddQuantize();
    ww_op_resolver->AddDepthwiseConv2D();
    ww_op_resolver->AddAveragePool2D();
    ww_op_resolver->AddMaxPool2D();
    ww_op_resolver->AddPad();
    ww_op_resolver->AddPack();
    ww_op_resolver->AddSplitV();

    // --- Load model ---
    const tflite::Model *model = tflite::GetModel(okay_nabu_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[WW] ERROR: Model schema version mismatch (got %d, expected %d)\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // --- Create interpreter ---
    ww_interpreter = new tflite::MicroInterpreter(
        model, *ww_op_resolver, ww_tensor_arena, WW_TENSOR_ARENA_SIZE, ww_mrv);

    if (ww_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[WW] ERROR: Failed to allocate TFLite tensors");
        return false;
    }

    // --- Validate model input/output ---
    TfLiteTensor *input = ww_interpreter->input(0);
    if (input->dims->size != 3 || input->dims->data[0] != 1 ||
        input->dims->data[2] != PREPROCESSOR_FEATURE_SIZE) {
        Serial.println("[WW] ERROR: Unexpected input dimensions");
        return false;
    }
    ww_model_stride = input->dims->data[1];
    Serial.printf("[WW] Model input: [1, %d, %d]\n", ww_model_stride, PREPROCESSOR_FEATURE_SIZE);

    TfLiteTensor *output = ww_interpreter->output(0);
    Serial.printf("[WW] Model output: [%d, %d], type=%d\n",
                  output->dims->data[0], output->dims->data[1], output->type);

    // --- Initialize audio frontend ---
    FrontendFillConfigWithDefaults(&ww_frontend_config);
    ww_frontend_config.window.size_ms = FEATURE_DURATION_MS;
    ww_frontend_config.window.step_size_ms = FEATURE_STEP_SIZE_MS;
    ww_frontend_config.filterbank.num_channels = PREPROCESSOR_FEATURE_SIZE;
    ww_frontend_config.filterbank.lower_band_limit = 125.0f;
    ww_frontend_config.filterbank.upper_band_limit = 7500.0f;
    ww_frontend_config.noise_reduction.smoothing_bits = 10;
    ww_frontend_config.noise_reduction.even_smoothing = 0.025f;
    ww_frontend_config.noise_reduction.odd_smoothing = 0.06f;
    ww_frontend_config.noise_reduction.min_signal_remaining = 0.05f;
    ww_frontend_config.pcan_gain_control.enable_pcan = true;
    ww_frontend_config.pcan_gain_control.strength = 0.95f;
    ww_frontend_config.pcan_gain_control.offset = 80.0f;
    ww_frontend_config.pcan_gain_control.gain_bits = 21;
    ww_frontend_config.log_scale.enable_log = true;
    ww_frontend_config.log_scale.scale_shift = 6;

    if (FrontendPopulateState(&ww_frontend_config, &ww_frontend_state, WW_SAMPLE_RATE) != 0) {
        Serial.println("[WW] ERROR: Failed to initialize audio frontend");
        return false;
    }
    ww_frontend_initialized = true;

    // --- Reset probabilities ---
    memset(ww_recent_probs, 0, sizeof(ww_recent_probs));
    ww_prob_index = 0;
    ww_ignore_windows = -WW_MIN_SLICES_BEFORE_DETECTION;
    ww_current_stride = 0;

    Serial.println("[WW] Wake word engine initialized successfully");
    Serial.printf("[WW] Tensor arena used: %zu / %d bytes\n",
                  ww_interpreter->arena_used_bytes(), WW_TENSOR_ARENA_SIZE);

    return true;
}

// ============================================================
// Process one chunk of audio through the frontend + model
// Returns true if wake word was detected in this chunk
// ============================================================
static bool wakeWordProcessChunk(const int16_t *samples, size_t sample_count) {
    if (!ww_frontend_initialized || !ww_interpreter) return false;

    size_t samples_processed = 0;
    size_t total_processed = 0;

    while (total_processed < sample_count) {
        // Run frontend on remaining samples
        struct FrontendOutput frontend_out = FrontendProcessSamples(
            &ww_frontend_state,
            samples + total_processed,
            sample_count - total_processed,
            &samples_processed);

        total_processed += samples_processed;

        if (frontend_out.size == 0) {
            // Not enough samples for a full feature yet
            break;
        }

        // Convert frontend output (uint16) to int8 features
        int8_t features[PREPROCESSOR_FEATURE_SIZE];
        for (size_t i = 0; i < PREPROCESSOR_FEATURE_SIZE; i++) {
            // The frontend outputs 16-bit values; quantize to int8
            // (same as ESPHome: right-shift by log_scale.scale_shift = 6)
            int32_t val = (int32_t)frontend_out.values[i] - 128;
            val = val >> 6;  // log_scale.scale_shift
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            features[i] = (int8_t)val;
        }

        // Feed features into the streaming model
        TfLiteTensor *input = ww_interpreter->input(0);
        ww_current_stride = ww_current_stride % ww_model_stride;

        memmove(
            (int8_t *)(tflite::GetTensorData<int8_t>(input)) +
                PREPROCESSOR_FEATURE_SIZE * ww_current_stride,
            features,
            PREPROCESSOR_FEATURE_SIZE);
        ww_current_stride++;

        // Run inference when we have enough stride slices
        if (ww_current_stride >= ww_model_stride) {
            if (ww_interpreter->Invoke() != kTfLiteOk) {
                Serial.println("[WW] WARNING: Inference failed");
                return false;
            }

            TfLiteTensor *output = ww_interpreter->output(0);
            uint8_t probability = output->data.uint8[0];

            // Store in sliding window
            ww_prob_index = (ww_prob_index + 1) % WW_SLIDING_WINDOW_SIZE;
            ww_recent_probs[ww_prob_index] = probability;

            // Check for detection
            if (ww_ignore_windows < 0) {
                ww_ignore_windows++;
                continue;
            }

            // Compute average probability over sliding window
            uint32_t sum = 0;
            for (size_t i = 0; i < WW_SLIDING_WINDOW_SIZE; i++) {
                sum += ww_recent_probs[i];
            }

            uint8_t quantized_cutoff = (uint8_t)(WW_PROBABILITY_CUTOFF * 255.0f);
            uint32_t threshold = (uint32_t)quantized_cutoff * WW_SLIDING_WINDOW_SIZE;

            if (sum > threshold) {
                // Wake word detected!
                Serial.printf("[WW] WAKE WORD DETECTED! avg_prob=%d/%d\n",
                              (int)(sum / WW_SLIDING_WINDOW_SIZE), quantized_cutoff);

                // Reset to avoid duplicate detections
                memset(ww_recent_probs, 0, sizeof(ww_recent_probs));
                ww_ignore_windows = -WW_MIN_SLICES_BEFORE_DETECTION;
                return true;
            }
        }

        // Cooldown tracking
        if (ww_recent_probs[ww_prob_index] < quantized_cutoff) {
            ww_ignore_windows = (ww_ignore_windows < 0) ? ww_ignore_windows + 1 : 1;
            if (ww_ignore_windows > 0) ww_ignore_windows = 0;
        }
    }

    return false;
}

// ============================================================
// Read a chunk of audio from I2S
// Returns number of samples read (0 on error)
// ============================================================
static size_t wakeWordReadI2S(int16_t *buffer, size_t max_samples) {
    size_t bytes_to_read = max_samples * sizeof(int32_t);
    size_t bytes_read = 0;

    esp_err_t err = i2s_read(I2S_NUM_0, ww_mic_buffer, bytes_to_read, &bytes_read, pdMS_TO_TICKS(100));
    if (err != ESP_OK) {
        return 0;
    }

    size_t samples_read = bytes_read / sizeof(int32_t);

    // Convert 32-bit I2S samples to 16-bit (take upper 16 bits)
    for (size_t i = 0; i < samples_read; i++) {
        buffer[i] = (int16_t)(ww_mic_buffer[i] >> 14);
    }

    return samples_read;
}

// ============================================================
// wakeWordTask — FreeRTOS task, runs on Core 1
// ============================================================
void wakeWordTask(void *pvParameters) {
    Serial.println("[WW] Wake word task started on Core " + String(xPortGetCoreID()));

    // Initialize the wake word engine
    if (!wakeWordInit()) {
        Serial.println("[WW] FATAL: Failed to initialize wake word engine. Task exiting.");
        vTaskDelete(NULL);
        return;
    }

    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(WW_CHUNK_SIZE_MS);

    // LED heartbeat counter for subtle "listening" indication
    uint32_t led_heartbeat = 0;

    while (1) {
        vTaskDelayUntil(&xLastWakeTime, xPeriod);

        // --- Guard conditions: don't run if... ---
        if (!wakeWordEnabled) {
            continue;  // disabled by mute or user
        }

        if (!deviceEnabled) {
            continue;  // device disabled by double-tap
        }

        if (isPlaying) {
            continue;  // audio playback active
        }

        if (mic_timeout > millis()) {
            continue;  // micTask is active (user is speaking)
        }

        // Check mute switch (active low with pullup)
        if (digitalRead(MUTE_PIN) == HIGH) {
            continue;  // muted
        }

        // --- Subtle LED heartbeat while listening for wake word ---
        led_heartbeat++;
        if (led_heartbeat % 60 == 0) {  // every ~2 seconds
            setLed(0, 255, 50, 255, 10);  // green pulse = listening for wake word
        }

        // --- Read audio from I2S ---
        size_t samples_read = wakeWordReadI2S(ww_audio_buffer, WW_CHUNK_SAMPLES);
        if (samples_read == 0) {
            continue;  // no data available
        }

        // --- Process through wake word pipeline ---
        if (wakeWordProcessChunk(ww_audio_buffer, samples_read)) {
            // Wake word detected!
            wakeWordDetected = true;

            // Trigger mic listening (same as center tap)
            mic_timeout = millis() + MIC_LISTEN_MS;

            // LED feedback: white flash
            setLed(255, 255, 255, 120, 8);

            Serial.println("[WW] Mic activated for " + String(MIC_LISTEN_MS / 1000) + "s");
        }
    }
}
