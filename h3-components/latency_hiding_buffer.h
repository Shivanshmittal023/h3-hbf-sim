// ============================================================================
//  latency_hiding_buffer.h -- H3 Latency Hiding Buffer (LHB)
// ============================================================================
//
//  WHAT THIS IS
//      The double-buffered SRAM prefetch buffer in the HBM base die that makes
//      HBF usable despite a ~20 us NAND array latency (H3 paper Section III-C).
//      While the GPU consumes one half, the other half is being filled from
//      HBF. If the fill keeps up, the 20 us is never observed by the GPU.
//
//      Paper Equation (1):
//          Capacity_LHB = 2 x BW_HBF x Latency_HBF
//                       = 2 x 1 TB/s x 20 us
//                       = 40 MB
//
//      NOTE ON SCOPE -- this is a PER-CUBE structure. Eq. (1) only yields
//      40 MB when BW_HBF is 1 TB/s, which is the per-cube bandwidth; the 8 TB/s
//      figure quoted elsewhere in the paper is the per-GPU aggregate over 8
//      cubes. Sizing one 40 MB buffer against 8 TB/s would under-provision by
//      8x and produce a misleadingly poor hit rate. A full GPU therefore has
//      8 x 40 MB = 320 MB of LHB. See `scope` in configs/lhb_config.yaml.
//
//  WHY IT IS A *STREAMING* DOUBLE BUFFER
//      A single Llama-3.1-405B FP8 layer weight tensor is ~870 MB -- vastly
//      larger than a 20 MB half. The LHB therefore does not hold "one tensor";
//      it walks a tensor sequentially in half-sized chunks, alternating halves.
//      The timing works out exactly:
//          drain one half : 20 MB / 1 TB/s = 20 us
//          fill one half  : tR             = 20 us
//      i.e. consumption and refill are balanced by construction, which is the
//      whole point of Eq. (1).
//
//  WHAT IT CONNECTS TO
//      Upstream   : `llm_prefetch_scheduler` (Step 5) supplies PrefetchHints
//                   derived from the deterministic LLM layer schedule.
//      Downstream : an `ILhbFillEngine` performs the actual HBF reads.
//                     * `AnalyticFillEngine` (provided here) models
//                       tR + size/bandwidth. Used by the unit test and usable
//                       as a fast standalone mode.
//                     * Step 6 supplies a Ramulator-backed engine so fills go
//                       through the real HBF timing model.
//      Sibling    : `h3_address_router` decides HBM vs HBF; only HBF-bound
//                   traffic reaches the LHB.
//
//  STATE MACHINE (per buffer half)
//      IDLE -> PREFETCH_ISSUED -> FILLING -> READY -> SERVING -> IDLE
//                ^                                                 |
//                +------------------- swap ------------------------+
//      A SWAP is recorded whenever a half finishes draining and is handed back
//      to the fill engine. `LhbStats::swaps` counts them.
//
//  TIME UNITS
//      Picoseconds (uint64_t) throughout, matching h3_address_router.
//
//  CONFIGURABLE PARAMETERS  (see configs/lhb_config.yaml)
//      enabled                     false = bypass mode, for the baseline
//      buffer_size_mb              total capacity (40)
//      num_buffers                 halves (2 = double buffering)
//      sram_access_latency_ns      latency of a hit (2)
//      hbf_read_latency_ns         tR modeled by AnalyticFillEngine (20000)
//      hbf_bandwidth_gbps          per-cube fill bandwidth (1000)
//      prefetch_hint_lead_time_ns  how far ahead hints are issued
//      max_outstanding_fills       fills in flight (defaults to num_buffers)
//
// ============================================================================
#ifndef LATENCY_HIDING_BUFFER_H
#define LATENCY_HIDING_BUFFER_H

#include <cstdint>
#include <deque>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace h3 {

// ---------------------------------------------------------------------------
// Per-half state machine
// ---------------------------------------------------------------------------
enum class LhbState : int {
  Idle = 0,         // empty, available to be filled
  PrefetchIssued,  // fill requested, engine has not started transferring
  Filling,          // data in flight from HBF
  Ready,            // fully resident, not yet touched by the GPU
  Serving,          // GPU is consuming it
};

const char* to_string(LhbState s);

// ---------------------------------------------------------------------------
// Prefetch hint -- produced by the LLM layer scheduler (Step 5)
// ---------------------------------------------------------------------------
// "At time `needed_at_ps` the GPU will need tensor `tensor_id`, which lives at
// HBF-local address `hbf_addr` and is `size_bytes` long."
struct PrefetchHint {
  uint64_t tensor_id = 0;
  uint64_t hbf_addr = 0;      // HBF region-local address
  uint64_t size_bytes = 0;
  uint64_t needed_at_ps = 0;  // when the GPU expects to start reading it
  int layer = -1;             // provenance, for statistics only
};

// ---------------------------------------------------------------------------
// Result of a GPU access
// ---------------------------------------------------------------------------
enum class LhbOutcome : int {
  Hit = 0,        // resident and complete: only SRAM latency is paid
  MissLate,      // resident but still FILLING: GPU stalls until fill completes
  MissAbsent,    // not in the buffer at all: full demand fetch from HBF
  Bypass,         // LHB disabled: straight to HBF (baseline mode)
};

const char* to_string(LhbOutcome o);

struct LhbAccess {
  LhbOutcome outcome = LhbOutcome::MissAbsent;
  uint64_t ready_time_ps = 0;  // when the data is available to the GPU
  uint64_t stall_ps = 0;       // ready_time_ps - request time (0 on a hit)
  int half = -1;               // which half served it, or -1

  // Set when the fill engine is ASYNCHRONOUS and the half this access needs is
  // still filling. The completion time is not knowable yet, so ready_time_ps
  // and stall_ps are meaningless: the caller must park the request and wait for
  // the buffer's fill-complete callback for `half`.
  //
  // This is request coalescing on an outstanding fill -- the same MSHR merging
  // real hardware does. Issuing a second fetch for data already in flight would
  // both double-count HBF traffic and misreport latency.
  bool waiting_on_fill = false;
};

// ---------------------------------------------------------------------------
// Fill engine interface
// ---------------------------------------------------------------------------
// Abstracts "how long does it take to get `size_bytes` from HBF". Keeping this
// abstract is what lets the LHB be unit-tested without Ramulator, and lets
// Step 6 substitute the real HBF timing model without touching LHB logic.
class ILhbFillEngine {
 public:
  virtual ~ILhbFillEngine() = default;

  // Begin a fill. Returns the absolute time at which the data is complete.
  // Only meaningful for synchronous engines (is_async() == false).
  virtual uint64_t start_fill(uint64_t hbf_addr, uint64_t size_bytes,
                              uint64_t now_ps) = 0;

  // Latency of an un-prefetched demand read of `size_bytes` issued at `now_ps`.
  virtual uint64_t demand_latency_ps(uint64_t size_bytes, uint64_t now_ps) = 0;

  virtual const char* name() const = 0;

  // ---- Asynchronous fills ------------------------------------------------
  // A closed-form engine knows when a fill lands the moment it starts. A real
  // memory model does not -- the answer depends on queueing and arbitration
  // that have not happened yet, and arrives later via callback.
  //
  // Engines that work that way report is_async() == true. The buffer then uses
  // start_fill_async() and marks the half Ready only when the callback fires,
  // rather than at a precomputed time.
  virtual bool is_async() const { return false; }

  using FillCompletionHandler =
      std::function<void(uint64_t token, uint64_t now_ps)>;
  virtual void set_completion_handler(FillCompletionHandler /*handler*/) {}

  // Returns false if the fill could not be issued; the buffer retries later.
  virtual bool start_fill_async(uint64_t /*hbf_addr*/, uint64_t /*size_bytes*/,
                                uint64_t /*now_ps*/, uint64_t /*token*/) {
    return false;
  }

  // Advance the engine's own clock (no-op for closed-form engines).
  virtual void tick(uint64_t /*now_ps*/) {}
};

// Closed-form engine: completion = now + tR + size / bandwidth.
// Adequate for LHB logic tests and for fast parameter sweeps; the Ramulator
// engine in Step 6 replaces it when contention effects matter.
class AnalyticFillEngine : public ILhbFillEngine {
 public:
  AnalyticFillEngine(uint64_t read_latency_ps, double bandwidth_gbps);

  uint64_t start_fill(uint64_t hbf_addr, uint64_t size_bytes,
                      uint64_t now_ps) override;
  uint64_t demand_latency_ps(uint64_t size_bytes, uint64_t now_ps) override;
  const char* name() const override { return "AnalyticFillEngine"; }

  uint64_t transfer_ps(uint64_t size_bytes) const;

 private:
  uint64_t m_read_latency_ps;
  double m_bytes_per_ps;
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct LhbConfig {
  bool enabled = true;                  // false -> bypass mode (baseline)

  uint64_t buffer_size_bytes = 40ULL << 20;  // 40 MB total
  uint32_t num_buffers = 2;                  // double buffering

  double sram_access_latency_ns = 2.0;
  double hbf_read_latency_ns = 20000.0;      // tR = 20 us
  double hbf_bandwidth_gbps = 1000.0;        // 1 TB/s per cube

  double prefetch_hint_lead_time_ns = 25000.0;
  uint32_t max_outstanding_fills = 0;        // 0 -> num_buffers

  // ---- Address interleaving --------------------------------------------
  // GPGPU-Sim spreads consecutive cache lines across all memory partitions, so
  // a partition sees only every Nth line of any contiguous range. A
  // per-partition slice of the LHB must therefore cover the SAME global
  // address span as the whole buffer, while storing only its 1/N share.
  //
  // Without this, a 1 MiB slice covers just 1 MiB of global span, the GPU
  // demands addresses far outside that window (absent misses), and ~(N-1)/N of
  // everything prefetched is never read -- prefetch efficiency collapses to 1/N.
  //
  // 1 = no interleaving (standalone use and the unit tests).
  // The Accel-Sim backend sets this to -gpgpu_n_mem.
  uint32_t address_interleave_factor = 1;

  static LhbConfig from_yaml(const std::string& path);
  void validate() const;

  // Bytes of SRAM in one half.
  uint64_t half_size_bytes() const {
    return num_buffers ? buffer_size_bytes / num_buffers : buffer_size_bytes;
  }
  // Global address span one half covers. Equals half_size_bytes() when there is
  // no interleaving; N x larger when this buffer holds 1/N of each line.
  uint64_t chunk_span_bytes() const {
    const uint32_t f = address_interleave_factor ? address_interleave_factor : 1;
    return half_size_bytes() * f;
  }
  uint64_t sram_access_latency_ps() const {
    return static_cast<uint64_t>(sram_access_latency_ns * 1000.0 + 0.5);
  }
  uint64_t hbf_read_latency_ps() const {
    return static_cast<uint64_t>(hbf_read_latency_ns * 1000.0 + 0.5);
  }
  uint64_t prefetch_hint_lead_time_ps() const {
    return static_cast<uint64_t>(prefetch_hint_lead_time_ns * 1000.0 + 0.5);
  }

  // Eq. (1): the capacity this configuration *should* have, given tR and BW.
  uint64_t required_capacity_bytes() const;
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
struct LhbStats {
  uint64_t accesses = 0;
  uint64_t hits = 0;
  uint64_t misses_late = 0;    // resident but still filling
  uint64_t misses_absent = 0;  // never prefetched
  uint64_t bypasses = 0;

  uint64_t total_stall_ps = 0;
  uint64_t max_stall_ps = 0;

  uint64_t bytes_served = 0;
  uint64_t bytes_prefetched = 0;
  uint64_t bytes_wasted = 0;   // prefetched, evicted before being read

  uint64_t fills_started = 0;
  uint64_t fills_completed = 0;
  uint64_t swaps = 0;
  uint64_t hints_received = 0;
  uint64_t hints_dropped = 0;  // queue overflow

  // Occupancy integral, for time-averaged utilisation.
  uint64_t occupancy_byte_ps = 0;
  uint64_t first_time_ps = 0;
  uint64_t last_time_ps = 0;
  bool has_window = false;

  double hit_rate() const;
  double miss_rate() const;
  double avg_stall_ps_per_miss() const;
  double avg_stall_ns_per_miss() const;
  double avg_stall_cycles_per_miss(double gpu_clock_ghz) const;
  double elapsed_ns() const;
  double avg_occupancy_bytes() const;
  double avg_utilization(uint64_t capacity_bytes) const;
  double prefetch_efficiency() const;   // bytes_served / bytes_prefetched

  void reset();
};

// ---------------------------------------------------------------------------
// The Latency Hiding Buffer
// ---------------------------------------------------------------------------
class LatencyHidingBuffer {
 public:
  LatencyHidingBuffer(const LhbConfig& config, ILhbFillEngine* fill_engine);

  // --- Prefetch path -------------------------------------------------------
  // Accept a hint from the LLM layer scheduler. The tensor is queued and
  // streamed into halves in half-sized chunks as they free up.
  void issue_hint(const PrefetchHint& hint);

  // Advance time: retire completed fills, start new ones into idle halves.
  // Safe (and cheap) to call every GPU cycle; also called internally by
  // access(), so an explicit tick loop is optional.
  void tick(uint64_t now_ps);

  // --- Demand path ---------------------------------------------------------
  // The GPU reads `size_bytes` at HBF-local `addr`.
  LhbAccess access(uint64_t addr, uint64_t size_bytes, uint64_t now_ps);

  // --- Asynchronous fills --------------------------------------------------
  // Invoked when a half's fill lands, so the backend can release any requests
  // it parked on that half (see LhbAccess::waiting_on_fill).
  using FillCompleteHandler = std::function<void(int half, uint64_t now_ps)>;
  void set_fill_complete_handler(FillCompleteHandler handler) {
    m_fill_complete_handler = std::move(handler);
  }

  // Charge a stall to the statistics once its true length is known. Used by
  // the backend for requests that waited on an asynchronous fill.
  void record_deferred_stall(uint64_t stall_ps);

  // --- Introspection -------------------------------------------------------
  LhbState half_state(int half) const;
  bool is_resident(uint64_t addr, uint64_t size_bytes) const;
  uint64_t occupancy_bytes() const;
  uint64_t pending_hints() const { return m_stream_queue.size(); }

  const LhbStats& stats() const { return m_stats; }
  const LhbConfig& config() const { return m_config; }
  void reset_stats() { m_stats.reset(); }

  void print_stats(std::ostream& os, double gpu_clock_ghz = 1.98) const;
  void print_state(std::ostream& os) const;

 private:
  // One half of the double buffer.
  struct Half {
    LhbState state = LhbState::Idle;
    uint64_t tensor_id = 0;
    uint64_t base_addr = 0;      // HBF-local start of the resident chunk
    uint64_t length = 0;         // GLOBAL address span this half covers
    uint64_t stored_bytes = 0;   // SRAM actually occupied (length / interleave)
    uint64_t consumed = 0;       // bytes already read by the GPU
    uint64_t fill_start_ps = 0;
    uint64_t fill_done_ps = 0;   // meaningless while an async fill is pending
    bool async_fill_pending = false;
    uint64_t fill_token = 0;
    int layer = -1;

    bool covers(uint64_t addr, uint64_t size) const {
      return length && addr >= base_addr && (addr + size) <= (base_addr + length);
    }
  };

  // A tensor being streamed in, tracked as a cursor over its address range.
  struct StreamCursor {
    uint64_t tensor_id = 0;
    uint64_t next_addr = 0;
    uint64_t remaining = 0;
    uint64_t needed_at_ps = 0;
    int layer = -1;
  };

  void retire_fills(uint64_t now_ps);
  void launch_fills(uint64_t now_ps);
  int find_half(uint64_t addr, uint64_t size) const;
  int find_idle_half() const;
  uint32_t outstanding_fills() const;
  void accumulate_occupancy(uint64_t now_ps);
  void release_half(Half& h, uint64_t now_ps);

  LhbConfig m_config;
  ILhbFillEngine* m_engine = nullptr;
  std::vector<Half> m_halves;
  std::deque<StreamCursor> m_stream_queue;
  LhbStats m_stats;
  uint64_t m_last_occupancy_update_ps = 0;
  bool m_occupancy_started = false;
  FillCompleteHandler m_fill_complete_handler;
  uint64_t m_next_fill_token = 1;

  void on_fill_complete(uint64_t token, uint64_t now_ps);
};

}  // namespace h3

#endif  // LATENCY_HIDING_BUFFER_H
