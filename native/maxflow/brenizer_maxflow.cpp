#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <emscripten/emscripten.h>

#include "vendor/atcoder/maxflow.hpp"

namespace {

using Cap = long long;

constexpr double kCapacityScale = 1048576.0;
constexpr Cap kSoftCapacityCap = 1LL << 50;
constexpr Cap kHardConstraintCap = 1LL << 60;

char g_last_error[256] = "";

void set_last_error(const char* message) {
  if (!message) {
    g_last_error[0] = '\0';
    return;
  }
  std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

Cap quantize_capacity(float value) {
  if (!(value > 0.0f) || !std::isfinite(value)) return 0;
  double scaled = std::ceil(static_cast<double>(value) * kCapacityScale);
  if (scaled < 1.0) scaled = 1.0;
  if (scaled > static_cast<double>(kSoftCapacityCap)) scaled = static_cast<double>(kSoftCapacityCap);
  return static_cast<Cap>(scaled);
}

bool validate_size(int grid_w, int grid_h, int* node_count) {
  if (grid_w <= 0 || grid_h <= 0) {
    set_last_error("grid dimensions must be positive");
    return false;
  }
  long long n = static_cast<long long>(grid_w) * static_cast<long long>(grid_h);
  if (n <= 0 || n > 100000000LL) {
    set_last_error("grid dimensions are out of range");
    return false;
  }
  *node_count = static_cast<int>(n);
  return true;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int solve_grid(
    int grid_w,
    int grid_h,
    const float* data_costs,
    const float* edge_weights_h,
    const float* edge_weights_v,
    const uint8_t* hard_constraints,
    uint8_t* out_labels,
    int32_t* out_stats) {
  set_last_error(nullptr);

  if (!data_costs || !out_labels) {
    set_last_error("data_costs and out_labels are required");
    return 1;
  }

  int node_count = 0;
  if (!validate_size(grid_w, grid_h, &node_count)) {
    return 2;
  }

  const int source = node_count;
  const int sink = node_count + 1;
  atcoder::mf_graph<Cap> graph(node_count + 2);

  for (int i = 0; i < node_count; ++i) {
    Cap cap_source = quantize_capacity(data_costs[i * 2 + 1]);
    Cap cap_sink = quantize_capacity(data_costs[i * 2]);

    if (hard_constraints) {
      if (hard_constraints[i] == 1) {
        cap_source = kHardConstraintCap;
        cap_sink = 0;
      } else if (hard_constraints[i] == 2) {
        cap_source = 0;
        cap_sink = kHardConstraintCap;
      }
    }

    if (cap_source > 0) graph.add_edge(source, i, cap_source);
    if (cap_sink > 0) graph.add_edge(i, sink, cap_sink);
  }

  if (grid_w > 1) {
    for (int y = 0; y < grid_h; ++y) {
      for (int x = 0; x < grid_w - 1; ++x) {
        const int a = y * grid_w + x;
        const int b = a + 1;
        const int edge_idx = y * (grid_w - 1) + x;
        const Cap weight = edge_weights_h ? quantize_capacity(edge_weights_h[edge_idx]) : 1;
        graph.add_edge(a, b, weight);
        graph.add_edge(b, a, weight);
      }
    }
  }

  if (grid_h > 1) {
    for (int y = 0; y < grid_h - 1; ++y) {
      for (int x = 0; x < grid_w; ++x) {
        const int a = y * grid_w + x;
        const int b = a + grid_w;
        const int edge_idx = y * grid_w + x;
        const Cap weight = edge_weights_v ? quantize_capacity(edge_weights_v[edge_idx]) : 1;
        graph.add_edge(a, b, weight);
        graph.add_edge(b, a, weight);
      }
    }
  }

  graph.flow(source, sink);
  const auto cut = graph.min_cut(source);
  for (int i = 0; i < node_count; ++i) {
    out_labels[i] = cut[i] ? 0 : 1;
  }

  if (out_stats) {
    out_stats[0] = 0;
    out_stats[1] = 0;
    out_stats[2] = 0;
    out_stats[3] = 0;
  }

  return 0;
}

EMSCRIPTEN_KEEPALIVE const char* last_error_message() {
  return g_last_error;
}

}  // extern "C"
