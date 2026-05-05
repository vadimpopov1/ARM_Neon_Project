#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "implot/implot.h"
#include <GLFW/glfw3.h>
#include <arm_neon.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>

static const int NUM = 50;
static size_t bench_ns[NUM];
static char bench_labels[NUM][16];
static const char* bench_label_ptrs[NUM];

void init_bench_sizes() {
    double log_min = log10(10.0);
    double log_max = log10(100000000.0);
    for (int i = 0; i < NUM; ++i) {
        double t = (double)i / (NUM - 1);
        bench_ns[i] = (size_t)pow(10.0, log_min + t * (log_max - log_min));
        if (bench_ns[i] < 1) bench_ns[i] = 1;
        if (bench_ns[i] >= 1000000)
            snprintf(bench_labels[i], 16, "%.1fM", bench_ns[i] / 1000000.0);
        else if (bench_ns[i] >= 1000)
            snprintf(bench_labels[i], 16, "%.0fK", bench_ns[i] / 1000.0);
        else
            snprintf(bench_labels[i], 16, "%zu", bench_ns[i]);
        bench_label_ptrs[i] = bench_labels[i];
    }
}

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

int64_t process_array_neon(const int32_t* __restrict__ data, size_t n) {
    int64_t sum = 0;
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        int32x4_t vec = vld1q_s32(data + i);
        uint32x4_t mask_pos = vcgtq_s32(vec, vdupq_n_s32(0));
        uint32x4_t mask_neg = vcltq_s32(vec, vdupq_n_s32(0));
        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);
        int32x4_t pos_part = vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(vec), mask_pos));
        int32x4_t neg_part = vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(abs_val), mask_neg));
        acc = vaddq_s32(acc, vorrq_s32(pos_part, neg_part));
    }
    int32x2_t lo = vget_low_s32(acc);
    int32x2_t hi = vget_high_s32(acc);
    int32x2_t pair = vpadd_s32(lo, hi);
    sum = vget_lane_s32(pair, 0) + vget_lane_s32(pair, 1);
    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

struct BenchResult {
    double scalar_s;
    double neon_s;
    double speedup;
    const char* label;
    double x;
};

void run_all(std::vector<BenchResult>& results, int iters) {
    results.resize(NUM);
    for (int i = 0; i < NUM; ++i) {
        size_t n = bench_ns[i];
        std::vector<int32_t> data(n);
        for (size_t j = 0; j < n; ++j) data[j] = (int32_t)(j % 7) - 3;

        double sum_s = 0, sum_n = 0;
        for (int k = 0; k < iters; ++k) {
            auto t1 = std::chrono::high_resolution_clock::now();
            volatile int64_t r1 = process_array_scalar(data.data(), n);
            auto t2 = std::chrono::high_resolution_clock::now();
            volatile int64_t r2 = process_array_neon(data.data(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            (void)r1; (void)r2;
            sum_s += std::chrono::duration<double>(t2 - t1).count();
            sum_n += std::chrono::duration<double>(t3 - t2).count();
        }
        double s = sum_s / iters;
        double ne = sum_n / iters;
        if (ne < 0.001) ne = 0.001;
        results[i] = { s, ne, s / ne, bench_label_ptrs[i], (double)bench_ns[i] };
    }
}

int main() {
    init_bench_sizes();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1000, 800, "NEON Benchmark", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    std::vector<BenchResult> results;
    int iterations = 3;
    bool need_run = true;
    int chart_type = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({1000, 800});
        ImGui::Begin("NEON Benchmark", nullptr,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Settings");
        ImGui::Separator();
        ImGui::SliderInt("Iterations", &iterations, 1, 20);
        ImGui::SameLine();
        if (ImGui::Button("Run")) need_run = true;
        ImGui::SameLine();
        ImGui::RadioButton("Bar", &chart_type, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Line", &chart_type, 1);
        ImGui::Spacing();

        if (need_run) {
            run_all(results, iterations);
            need_run = false;
        }

        if (!results.empty()) {
            double xs[NUM], ys_s[NUM], ys_n[NUM], ys_sp[NUM];
            for (int i = 0; i < NUM; ++i) {
                xs[i] = results[i].x;
                ys_s[i] = results[i].scalar_s;
                ys_n[i] = results[i].neon_s;
                ys_sp[i] = results[i].speedup;
            }

            ImGui::Text("Time (s) — %d iteration(s)", iterations);
            ImGui::Separator();

            if (ImPlot::BeginPlot("##time", {-1, 220})) {
                ImPlot::SetupAxes("Array size", "Time (s)");
                ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_AutoFit);
                if (chart_type == 0) {
                    ImPlot::PlotBars("Scalar", xs, ys_s, NUM, 0.35);
                    ImPlot::PlotBars("NEON", xs, ys_n, NUM, 0.35);
                } else {
                    ImPlot::PlotLine("Scalar", xs, ys_s, NUM);
                    ImPlot::PlotLine("NEON", xs, ys_n, NUM);
                }
                ImPlot::EndPlot();
            }

            ImGui::Spacing();
            ImGui::Text("Speedup (x)");
            ImGui::Separator();

            if (ImPlot::BeginPlot("##speedup", {-1, 150})) {
                ImPlot::SetupAxes("Array size", "Speedup");
                ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_AutoFit);
                // ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 10, ImPlotCond_Always);
                if (chart_type == 0) {
                    ImPlot::PlotBars("Speedup", xs, ys_sp, NUM, 0.5);
                } else {
                    ImPlot::PlotLine("Speedup", xs, ys_sp, NUM);
                }
                ImPlot::EndPlot();
            }

            ImGui::Spacing();
            ImGui::Text("Results");
            ImGui::Separator();

            ImGui::BeginChild("##table_scroll", {0, 180}, true);
            ImGui::Columns(4, "table");
            ImGui::SetColumnWidth(0, 80);
            ImGui::SetColumnWidth(1, 160);
            ImGui::SetColumnWidth(2, 160);
            ImGui::SetColumnWidth(3, 120);
            ImGui::Text("Size"); 
            ImGui::NextColumn();
            ImGui::Text("Scalar (s)"); 
            ImGui::NextColumn();
            ImGui::Text("NEON (s)");
            ImGui::NextColumn();
            ImGui::Text("Speedup");
            ImGui::NextColumn();
            ImGui::Separator();

            for (int i = 0; i < NUM; ++i) {
                ImGui::Text("%s", results[i].label);
                ImGui::NextColumn();
                ImGui::Text("%.4f", results[i].scalar_s);
                ImGui::NextColumn();
                ImGui::Text("%.4f", results[i].neon_s);
                ImGui::NextColumn();
                ImGui::TextColored({1.f, 0.85f, 0.2f, 1.f}, "%.2fx", results[i].speedup); ImGui::NextColumn();
            }
            ImGui::Columns(1);
            ImGui::EndChild();
        }

        ImGui::End();

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}