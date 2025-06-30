// Clean real-time speech transcription
// Based on whisper.cpp stream example but outputs only new text segments
// Waits for silence before outputting transcribed text

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include <SDL.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <set>
#include <fstream>

// Global variable to store the minimum log level
static int g_whisper_log_level = GGML_LOG_LEVEL_ERROR;

// Callback to filter whisper logging by level
static void whisper_log_callback_filtered(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    // Only show messages at or above the configured log level
    if (level >= g_whisper_log_level) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t capture_id = -1;
    int32_t max_tokens = 128;
    int32_t audio_ctx  = 0;

    int32_t audio_buffer_ms = 2000;  // Audio buffer duration - must be longer than transcription time
    int32_t silence_ms = 500;    // Silence duration before outputting text
    int32_t min_step_ms = 500;   // Minimum time between audio collection steps
    int32_t beam_size = 5;       // Beam search size (0 or 1 = greedy, 2+ = beam search)
    float vad_thold    = 0.5f;   // VAD speech probability threshold

    bool no_fallback   = true;
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool verbose       = false;
    bool list_devices  = false;
    int32_t whisper_log_level = 4;  // 0=NONE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string vad_model = "models/ggml-silero-v5.1.2.bin";
};

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads\n", params.n_threads);
            fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n", params.language.c_str());
            fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n", params.model.c_str());
            fprintf(stderr, "  --vad-model FNAME         [%-7s] VAD model path\n", params.vad_model.c_str());
            fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n", params.capture_id);
            fprintf(stderr, "  --audio-buffer N          [%-7d] audio buffer duration (ms)\n", params.audio_buffer_ms);
            fprintf(stderr, "  --silence N               [%-7d] silence duration before output (ms)\n", params.silence_ms);
            fprintf(stderr, "  --min-step N              [%-7d] minimum time between audio steps (ms)\n", params.min_step_ms);
            fprintf(stderr, "  --beam-size N             [%-7d] beam search size (0 or 1 = greedy, 2+ = beam search)\n", params.beam_size);
            fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] VAD speech probability threshold\n", params.vad_thold);
            fprintf(stderr, "  --no-gpu                  [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
            fprintf(stderr, "  -fa,      --flash-attn    [%-7s] enable flash attention\n", params.flash_attn ? "true" : "false");
            fprintf(stderr, "  -v,       --verbose       [%-7s] enable verbose/debug output\n", params.verbose ? "true" : "false");
            fprintf(stderr, "  --list-devices            [%-7s] list available audio capture devices and exit\n", "false");
            fprintf(stderr, "  --whisper-log-level N     [%-7d] whisper log level (0=NONE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR)\n", params.whisper_log_level);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")   { params.n_threads  = std::stoi(argv[++i]); }
        else if (arg == "-l"    || arg == "--language")  { params.language   = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")     { params.model      = argv[++i]; }
        else if (                  arg == "--vad-model") { params.vad_model  = argv[++i]; }
        else if (arg == "-c"    || arg == "--capture")   { params.capture_id = std::stoi(argv[++i]); }
        else if (                  arg == "--audio-buffer") { params.audio_buffer_ms = std::stoi(argv[++i]); }
        else if (                  arg == "--silence")   { params.silence_ms = std::stoi(argv[++i]); }
        else if (                  arg == "--min-step")  { params.min_step_ms = std::stoi(argv[++i]); }
        else if (                  arg == "--beam-size") { params.beam_size = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold") { params.vad_thold  = std::stof(argv[++i]); }
        else if (                  arg == "--no-gpu")    { params.use_gpu    = false; }
        else if (arg == "-fa"   || arg == "--flash-attn") { params.flash_attn = true; }
        else if (arg == "-v"    || arg == "--verbose")   { params.verbose    = true; }
        else if (                  arg == "--list-devices") { params.list_devices = true; }
        else if (                  arg == "--whisper-log-level") { params.whisper_log_level = std::stoi(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }

    // Parameter validation
    params.audio_buffer_ms = std::max(params.audio_buffer_ms, 1000);
    params.silence_ms = std::max(params.silence_ms, 500);
    params.min_step_ms = std::max(params.min_step_ms, 100);
    params.whisper_log_level = std::max(0, std::min(params.whisper_log_level, 5));

    // Language validation
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        return false;
    }

    return true;
}

// Detect voice activity using Silero VAD
static bool detect_voice_activity(
    whisper_vad_context* vad_ctx,
    const std::vector<float>& audio_samples,
    float vad_threshold) {

    if (!vad_ctx || audio_samples.empty()) {
        return false;
    }

    // Run VAD detection
    bool vad_success = whisper_vad_detect_speech(vad_ctx, audio_samples.data(), audio_samples.size());
    if (!vad_success) {
        return false;
    }

    // Get speech probabilities from VAD context
    int n_probs = whisper_vad_n_probs(vad_ctx);
    float* probs = whisper_vad_probs(vad_ctx);

    if (n_probs <= 0 || probs == nullptr) {
        return false;
    }

    // Use maximum probability across all chunks (most sensitive to any speech activity)
    float max_prob = *std::max_element(probs, probs + n_probs);

    bool voice_detected = max_prob > vad_threshold;

    return voice_detected;
}

static std::string transcribe_audio_segment(
    whisper_context* ctx,
    const std::vector<float>& pcmf32_segment,
    const whisper_params& params) {
    
    if (pcmf32_segment.empty()) {
        return "";
    }
    
    if (params.verbose) {
        fprintf(stderr, "[DEBUG] Running whisper inference on %.1f seconds of audio\n",
                pcmf32_segment.size() / (float)WHISPER_SAMPLE_RATE);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Run whisper inference
    // Choose strategy based on beam_size parameter
    whisper_sampling_strategy strategy = (params.beam_size <= 1) ? WHISPER_SAMPLING_GREEDY : WHISPER_SAMPLING_BEAM_SEARCH;
    whisper_full_params wparams = whisper_full_default_params(strategy);
    wparams.print_progress   = false;
    wparams.print_special    = false;  // Always hide special tokens
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.suppress_nst     = true;   // Suppress non-speech tokens
    wparams.translate        = false;  // Always transcribe in original language
    wparams.single_segment   = false;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.audio_ctx        = params.audio_ctx;
    wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

    // Set beam size for beam search strategy
    if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
        wparams.beam_search.beam_size = params.beam_size;
    }

    if (whisper_full(ctx, wparams, pcmf32_segment.data(), pcmf32_segment.size()) == 0) {
        auto t_end = std::chrono::high_resolution_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        if (params.verbose) {
            float audio_duration = pcmf32_segment.size() / (float)WHISPER_SAMPLE_RATE * 1000.0f; // ms
            float real_time_factor = audio_duration / inference_time;
            fprintf(stderr, "[DEBUG] Inference completed in %ld ms (%.1fx real-time, flash_attn=%s)\n",
                    inference_time, real_time_factor, params.flash_attn ? "on" : "off");
        }

        // Extract and output text segments
        const int n_segments = whisper_full_n_segments(ctx);
        std::string full_text;

        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            if (text && strlen(text) > 0) {
                full_text += text;
            }
        }

        // Clean up the text (remove leading/trailing whitespace)
        return ::trim(full_text);
    }
    
    return "";
}

// List available audio capture devices
static void list_audio_devices() {
    // Initialize SDL audio subsystem
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        fprintf(stderr, "error: failed to initialize SDL audio: %s\n", SDL_GetError());
        return;
    }

    // Get number of recording devices
    int num_devices = SDL_GetNumAudioDevices(1); // 1 = capture devices
    if (num_devices < 0) {
        fprintf(stderr, "error: failed to get audio devices: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    // List all available capture devices
    for (int i = 0; i < num_devices; i++) {
        const char* device_name = SDL_GetAudioDeviceName(i, 1);
        if (device_name) {
            printf("%d: %s\n", i, device_name);
        } else {
            printf("%d: Unknown Device\n", i);
        }
    }

    SDL_Quit();
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    // Parameter validation
    whisper_params params;
    if (!whisper_params_parse(argc, argv, params)) {
        return 1;
    }

    // Handle device listing request
    if (params.list_devices) {
        list_audio_devices();
        return 0;
    }

    // Set whisper logging callback with configurable log level
    g_whisper_log_level = params.whisper_log_level;
    whisper_log_set(whisper_log_callback_filtered, nullptr);

    // Initialize whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    // Initialize Silero VAD context (CPU only - GPU VAD disabled in whisper.cpp for performance)
    // NOTE: GPU support is hardcoded to false in whisper_vad_init_context() in src/whisper.cpp
    // Check that function if whisper.cpp is updated to see if GPU VAD support is re-enabled
    struct whisper_vad_context_params vad_cparams = whisper_vad_default_context_params();
    vad_cparams.n_threads = params.n_threads;
    vad_cparams.use_gpu = false;
    struct whisper_vad_context * vad_ctx = whisper_vad_init_from_file_with_params(params.vad_model.c_str(), vad_cparams);
    if (vad_ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize VAD context from %s\n", params.vad_model.c_str());
        whisper_free(ctx);
        return 3;
    }

    // Audio buffer allocation
    std::vector<float> pcmf32_segment; // Audio for current speech segment
    const int n_samples_buffer = (params.audio_buffer_ms * WHISPER_SAMPLE_RATE) / 1000;
    std::vector<float> pcmf32_buffer(n_samples_buffer, 0.0f);
    const int n_samples_vad = (params.silence_ms * WHISPER_SAMPLE_RATE) / 1000;
    std::vector<float> pcmf32_vad(n_samples_vad, 0.0f);

    // Initialize audio
    audio_async audio(params.audio_buffer_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "error: failed to initialize audio\n");
        return 1;
    }
    audio.resume();
    auto last_audio_get_time = std::chrono::high_resolution_clock::now();

    // Print processing info
    if (params.verbose) {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en") {
                params.language = "en";
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing audio (len = %.1f sec), %d threads, lang = %s\n",
                __func__,
                float(params.audio_buffer_ms) / 1000.0f,
                params.n_threads,
                params.language.c_str());
        
        fprintf(stderr, "%s: Using Silero VAD model: %s\n", __func__, params.vad_model.c_str());
        fprintf(stderr, "%s: Silence threshold = %d ms\n", __func__, params.silence_ms);
        
        fprintf(stderr, "%s: GPU = %s, flash attention = %s\n",
                __func__, params.use_gpu ? "true" : "false", params.flash_attn ? "true" : "false");
        
        fprintf(stderr, "%s: model = %s\n", __func__, params.model.c_str());
        
        fprintf(stderr, "%s: Ready for transcription. Listening for speech...\n", __func__);
        fprintf(stderr, "\n");
    }

    bool in_speech = false;

    // Main processing loop
    while (true) {
        // Handle Ctrl + C
        if (!sdl_poll_events()) break;

        // Don't collect audio more frequently than every min_step_ms.
        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_audio_get_time).count();
        auto sleep_duration = params.min_step_ms - time_since_last;
        if (sleep_duration > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration));
        }

        // Fetch audio
        now = std::chrono::high_resolution_clock::now();
        pcmf32_buffer.clear();
        audio.get(params.audio_buffer_ms, pcmf32_buffer);
        auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_audio_get_time).count();
        last_audio_get_time = now;
        
        // Determine if the last param.silence_ms contain any speech.
        bool voice_detected = false;
        if (pcmf32_buffer.size() >= static_cast<size_t>(n_samples_vad)) {
            // Replace VAD buffer with the most recent samples from main buffer
            std::copy(pcmf32_buffer.end() - n_samples_vad, pcmf32_buffer.end(), pcmf32_vad.begin());
            voice_detected = detect_voice_activity(vad_ctx, pcmf32_vad, params.vad_thold);
        }

        if (in_speech) {
            // Accumulate audio to speech segment.
            size_t new_samples = (elapsed_time_ms * WHISPER_SAMPLE_RATE) / 1000;
            new_samples = std::min(new_samples, pcmf32_buffer.size());
            if (new_samples > 0) {
                pcmf32_segment.insert(pcmf32_segment.end(), pcmf32_buffer.end() - new_samples, pcmf32_buffer.end());
            }
        }

        if (voice_detected && !in_speech) {
            // Start of new speech segment
            if (params.verbose) {
                fprintf(stderr, "\n[DEBUG] Speech started, beginning new segment\n");
            }
            in_speech = true;

            // Initialize for new segment
            pcmf32_segment.clear();
            // Include the last vad interval so we don't truncate the first word or two.
            pcmf32_segment.insert(pcmf32_segment.end(), pcmf32_buffer.end() - n_samples_vad, pcmf32_buffer.end());
        }

        if (!voice_detected && in_speech) {
            // End of speech segment and transcribe
            if (params.verbose) {
                fprintf(stderr, "[DEBUG] Speech ended, transcribing segment");
            }

            // Transcribe the audio segment
            std::string transcribed_text = transcribe_audio_segment(ctx, pcmf32_segment, params);
            
            // Output the transcribed text with a space separator
            if (!transcribed_text.empty()) {
                printf("%s\n", transcribed_text.c_str());
                fflush(stdout);
            }

            // Reset for next speech segment
            in_speech = false;
            pcmf32_segment.clear();
        }
    }

    audio.pause();
    whisper_vad_free(vad_ctx);
    whisper_free(ctx);
    return 0;
}
