#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <arm_neon.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>

__attribute__((optimize("no-tree-vectorize")))
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
    size_t n;
    double scalar_ms;
    double neon_ms;
    double speedup;
    const char* label;
};

BenchResult run_bench(size_t n, const char* label) {
    std::vector<int32_t> data(n);
    for (size_t i = 0; i < n; ++i)
        data[i] = (int32_t)(i % 7) - 3;

    auto t1 = std::chrono::high_resolution_clock::now();
    volatile int64_t r1 = process_array_scalar(data.data(), n);
    auto t2 = std::chrono::high_resolution_clock::now();
    volatile int64_t r2 = process_array_neon(data.data(), n);
    auto t3 = std::chrono::high_resolution_clock::now();
    (void)r1; (void)r2;

    double s  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double ne = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (ne < 0.001) ne = 0.001;
    return { n, s, ne, s / ne, label };
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(900, 620, "NEON Benchmark", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    std::vector<BenchResult> results;
    results.push_back(run_bench(10,        "10"));
    results.push_back(run_bench(100,       "100"));
    results.push_back(run_bench(1000,      "1K"));
    results.push_back(run_bench(10000,     "10K"));
    results.push_back(run_bench(100000,    "100K"));
    results.push_back(run_bench(1000000,   "1M"));
    results.push_back(run_bench(10000000,  "10M"));
    results.push_back(run_bench(100000000, "100M"));

    int n = (int)results.size();

    std::vector<float> scalar_ms(n), neon_ms(n), speedups(n);
    float max_val = 0.001f;
    for (int i = 0; i < n; ++i) {
        scalar_ms[i] = (float)results[i].scalar_ms;
        neon_ms[i]   = (float)results[i].neon_ms;
        speedups[i]  = (float)results[i].speedup;
        if (scalar_ms[i] > max_val) max_val = scalar_ms[i];
        if (neon_ms[i]   > max_val) max_val = neon_ms[i];
    }
    max_val *= 1.2f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({900, 620});
        ImGui::Begin("NEON Benchmark", nullptr,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Scalar vs NEON — time (ms)");
        ImGui::Separator();

        ImDrawList* draw = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        float W = 860, H = 200;
        float bar_w = W / (n * 3 + 1);

        draw->AddRectFilled(p, {p.x + W, p.y + H}, IM_COL32(25, 25, 25, 255));

        for (int i = 0; i < n; ++i) {
            float x = p.x + bar_w + i * (bar_w * 3);

            float hs = (scalar_ms[i] / max_val) * H;
            if (hs < 1) hs = 1;
            draw->AddRectFilled({x, p.y + H - hs}, {x + bar_w, p.y + H}, IM_COL32(100, 160, 255, 255));

            float hn = (neon_ms[i] / max_val) * H;
            if (hn < 1) hn = 1;
            draw->AddRectFilled({x + bar_w, p.y + H - hn}, {x + bar_w * 2, p.y + H}, IM_COL32(80, 220, 120, 255));

            draw->AddText({x, p.y + H + 4}, IM_COL32(200, 200, 200, 255), results[i].label);
        }

        ImGui::Dummy({W, H + 20});

        ImGui::TextColored({0.4f, 0.63f, 1.f, 1.f}, "  Scalar");
        ImGui::SameLine();
        ImGui::TextColored({0.31f, 0.86f, 0.47f, 1.f}, "  NEON");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Results");
        ImGui::Separator();

        ImGui::Columns(4, "table");
        ImGui::SetColumnWidth(0, 80);
        ImGui::SetColumnWidth(1, 160);
        ImGui::SetColumnWidth(2, 160);
        ImGui::SetColumnWidth(3, 120);
        ImGui::Text("Size");      ImGui::NextColumn();
        ImGui::Text("Scalar (ms)"); ImGui::NextColumn();
        ImGui::Text("NEON (ms)");   ImGui::NextColumn();
        ImGui::Text("Speedup");     ImGui::NextColumn();
        ImGui::Separator();

        for (int i = 0; i < n; ++i) {
            ImGui::Text("%s", results[i].label);          ImGui::NextColumn();
            ImGui::Text("%.4f", scalar_ms[i]);            ImGui::NextColumn();
            ImGui::Text("%.4f", neon_ms[i]);              ImGui::NextColumn();
            ImGui::TextColored({1.f, 0.85f, 0.2f, 1.f}, "%.2fx", speedups[i]); ImGui::NextColumn();
        }
        ImGui::Columns(1);
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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}