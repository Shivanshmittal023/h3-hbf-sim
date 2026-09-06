// ============================================================================
//  llm_prefetch_scheduler.h -- deterministic LLM prefetch hint generator
// ============================================================================
//
//  WHAT THIS IS
//      The component that makes the Latency Hiding Buffer work. LLM inference
//      has a fully deterministic memory access pattern: before a single token
//      is generated we already know every tensor that will be read, in what
//      order, and roughly when. That is what lets a 20 us NAND latency be
//      hidden by prefetching -- there is nothing to predict.
//
//      This class turns a model description (configs/llama_405b_config.yaml)
//      into:
//        1. a concrete ADDRESS LAYOUT   -- every tensor placed in HBF or HBM
//        2. a TIME SCHEDULE             -- when each tensor is needed
//        3. a HINT STREAM               -- issued early enough to hide tR
//
//  WHAT IT CONNECTS TO
//      Downstream : `LatencyHidingBuffer::issue_hint()` (Step 4).
//      Sibling    : `h3_address_router` -- the addresses produced here must
//                   fall in the regions that router decodes, so both read the
//                   same base addresses. A mismatch shows up immediately as
//                   `h3_router_unmapped_requests`.
//      Consumer   : tools/generate_synthetic_trace.py replicates this layout
//                   in Python so generated traces touch the same addresses.
//
//  LEAD TIME
//      hint_issue_time = tensor_needed_time - lead_time_ns - safety_margin_ns
//      clamped at zero. lead_time must cover tR plus the transfer of one LHB
//      half; see configs/llama_405b_config.yaml for the arithmetic.
//
//  TIME UNITS
//      Picoseconds (uint64_t), matching the router and the LHB.
//
//  NO WEIGHTS ARE LOADED. Only sizes are computed; traffic is synthetic.
// ============================================================================
#ifndef LLM_PREFETCH_SCHEDULER_H
#define LLM_PREFETCH_SCHEDULER_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "latency_hiding_buffer.h"

namespace h3 {

// ---------------------------------------------------------------------------
// Where a tensor lives, and what it is
// ---------------------------------------------------------------------------
enum class TensorRegion : int { Hbm = 0, Hbf = 1 };

enum class TensorKind : int {
  WeightQkv = 0,    // q_proj / k_proj / v_proj
  WeightO,          // o_proj
  WeightMlp,        // gate_proj / up_proj / down_proj
  WeightNorm,       // RMSNorm scales
  WeightEmbed,      // token embedding / lm_head
  KvShared,         // shared precomputed KV cache  -> HBF, read-only
  KvGenerated,      // KV produced during inference -> HBM, read/write
  Activation,        // transient activations        -> HBM
};

const char* to_string(TensorRegion r);
const char* to_string(TensorKind k);

struct TensorDesc {
  uint64_t id = 0;
  std::string name;
  int layer = -1;                     // -1 for non-layer tensors
  TensorKind kind = TensorKind::WeightQkv;
  TensorRegion region = TensorRegion::Hbf;
  uint64_t local_addr = 0;            // offset within its region
  uint64_t global_addr = 0;           // region_base + local_addr
  uint64_t size_bytes = 0;
  uint64_t needed_at_ps = 0;          // when the GPU starts reading it
  uint64_t hint_at_ps = 0;            // when its prefetch hint is issued
  bool read_only = true;
};

// ---------------------------------------------------------------------------
// Model configuration (configs/llama_405b_config.yaml)
// ---------------------------------------------------------------------------
struct ModelConfig {
  // model
  std::string name = "llama-3.1-405b";
  uint64_t num_layers = 126;
  uint64_t hidden_dim = 16384;
  uint64_t num_attention_heads = 128;
  uint64_t num_kv_heads = 8;
  uint64_t intermediate_dim = 53248;
  uint64_t vocab_size = 128256;
  uint64_t head_dim = 128;
  uint64_t weight_dtype_bytes = 1;      // FP8

  // kv_cache -- see the long note in the YAML: 128 heads + FP16 reproduces
  // the paper's occupancy figures; 8 heads is the true GQA architecture.
  uint64_t num_cache_heads = 128;
  uint64_t kv_dtype_bytes = 2;          // FP16

  // serving
  uint64_t input_seq_len = 1024;
  uint64_t output_seq_len = 1024;
  uint64_t batch_size = 1;
  uint64_t sequence_length = 1048576;
  uint64_t num_gpus = 8;
  uint64_t tensor_parallel_size = 8;

  // hardware
  double gpu_clock_ghz = 1.98;
  double aggregate_read_bandwidth_gbps = 8000.0;
  uint64_t hbm_capacity_bytes_per_gpu = 192ULL << 30;
  uint64_t hbf_capacity_bytes_per_gpu = 3ULL << 40;

  // address_map
  uint64_t hbm_base_addr = 0x0;
  uint64_t hbf_base_addr = 192ULL << 30;
  uint64_t tensor_alignment_bytes = 4096;

  // prefetch
  double lead_time_ns = 45000.0;
  double safety_margin_ns = 5000.0;
  uint64_t max_outstanding_hints = 8;

  static ModelConfig from_yaml(const std::string& path);
  void validate() const;

  // ---- Derived quantities (all analytic; nothing is loaded) --------------
  uint64_t weight_bytes_per_layer() const;
  uint64_t weight_bytes_all_layers() const;
  uint64_t embedding_bytes() const;
  uint64_t total_weight_bytes() const;         // whole model, all GPUs
  uint64_t total_weight_bytes_per_gpu() const; // this GPU's TP shard
  uint64_t total_parameters() const;

  uint64_t kv_bytes_per_token() const;         // whole model, all GPUs
  uint64_t generated_kv_tokens() const;        // ISL + OSL
  uint64_t shared_kv_tokens() const;           // sequence_length - generated
  uint64_t generated_kv_bytes_per_gpu() const; // -> HBM
  uint64_t shared_kv_bytes_per_gpu() const;    // -> HBF
  uint64_t activation_bytes_per_gpu() const;   // -> HBM

  double hbf_occupancy() const;                // fraction of HBF used
  double hbm_occupancy() const;                // fraction of HBM used

  uint64_t lead_time_ps() const {
    return static_cast<uint64_t>((lead_time_ns + safety_margin_ns) * 1000.0 + 0.5);
  }
};

// ---------------------------------------------------------------------------
// Scheduler statistics
// ---------------------------------------------------------------------------
struct SchedulerStats {
  uint64_t tensors_total = 0;
  uint64_t tensors_hbf = 0;
  uint64_t tensors_hbm = 0;
  uint64_t hints_issued = 0;
  uint64_t hints_pending = 0;
  uint64_t hints_late = 0;      // clamped to t=0: no room to hide tR
  uint64_t bytes_hbf = 0;
  uint64_t bytes_hbm = 0;
};

// ---------------------------------------------------------------------------
// The scheduler
// ---------------------------------------------------------------------------
class LLMPrefetchScheduler {
 public:
  explicit LLMPrefetchScheduler(const ModelConfig& config);

  // Precompute the full tensor layout and access schedule. Called by the
  // constructor; re-callable after mutating the config.
  void build_schedule();

  // Advance to `now_ps`, issuing every hint that has come due. Returns how
  // many were issued this call. `lhb` may be null (schedule-only mode).
  uint64_t tick(uint64_t now_ps, LatencyHidingBuffer* lhb);

  // Reset the hint cursor without rebuilding the layout.
  void rewind();

  const std::vector<TensorDesc>& tensors() const { return m_tensors; }
  const ModelConfig& config() const { return m_config; }
  const SchedulerStats& stats() const { return m_stats; }

  // Total simulated time for one full forward pass, from the schedule.
  uint64_t forward_pass_duration_ps() const { return m_forward_duration_ps; }

  // Look up which tensor owns a global address (-1 if none). Linear scan;
  // intended for diagnostics and tests, not for the per-access hot path.
  int tensor_at(uint64_t global_addr) const;

  void print_layout(std::ostream& os, int max_rows = 24) const;
  void print_summary(std::ostream& os) const;
  void print_schedule(std::ostream& os, int max_rows = 16) const;

 private:
  uint64_t align_up(uint64_t v) const;
  void add_tensor(const std::string& name, int layer, TensorKind kind,
                  TensorRegion region, uint64_t size_bytes);

  ModelConfig m_config;
  std::vector<TensorDesc> m_tensors;
  SchedulerStats m_stats;

  uint64_t m_hbf_cursor = 0;   // next free HBF-local offset
  uint64_t m_hbm_cursor = 0;   // next free HBM-local offset
  size_t m_next_hint = 0;      // index into m_tensors, ordered by hint_at_ps
  std::vector<size_t> m_hint_order;
  uint64_t m_forward_duration_ps = 0;
};

}  // namespace h3

#endif  // LLM_PREFETCH_SCHEDULER_H
