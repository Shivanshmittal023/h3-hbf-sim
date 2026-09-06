// ============================================================================
//  h3_address_router.h -- H3 address decoder & router (HBM <-> HBF)
// ============================================================================
//
//  WHAT THIS IS
//      Models the address decoder and router that the H3 paper places inside
//      the HBM base die (Fig. 3b). Every memory request arriving from the GPU
//      is decoded against the H3 address map (Fig. 3c) and steered to either:
//
//          * the HBM core          -- generated KV cache + activations
//          * the HBF base die      -- model weights + shared precomputed KV
//                                     cache, reached over a die-to-die (D2D)
//                                     link that costs an extra hop of latency
//
//  WHAT IT CONNECTS TO
//      Upstream : Accel-Sim / GPGPU-Sim, via `h3_memory_backend` (Step 6),
//                 which converts a `mem_fetch` into an `H3Request`.
//      Downstream: two `IMemoryBackend` implementations, one per memory type.
//                 In the full simulator these wrap Ramulator memory systems
//                 (HBM3e and HBF); in `tests/test_h3_router.cpp` they are
//                 trivial fakes, which is why this header deliberately
//                 depends on NOTHING from Ramulator or GPGPU-Sim.
//      Sits UPSTREAM of per-device address decoding: the router works on the
//      global address, then hands each backend a region-local address starting
//      at zero, because Ramulator's decoder expects a zero-based space.
//
//  TIME UNITS
//      Everything internal is in PICOSECONDS (uint64_t). Rationale: GPGPU-Sim
//      runs at ~1.98 GHz (0.505 ns/cycle), so an integer-nanosecond clock would
//      accumulate rounding error across millions of cycles. Config files
//      express latency in nanoseconds (double) and conversion happens once, in
//      the constructor. The Step 6 backend converts GPU cycles <-> ps.
//
//  CONFIGURABLE PARAMETERS  (see configs/h3_router_config.yaml)
//      hbm_base_addr / hbm_size_bytes    HBM region, default 0 .. 192 GB
//      hbf_base_addr / hbf_size_bytes    HBF region, default 192 GB .. +3 TB
//      d2d_hop_latency_ns                die-to-die routing overhead (25 ns)
//      d2d_hops_to_hbf                   number of hops charged (1)
//      enforce_hbf_read_only             reject writes to the HBF region
//      hbf_write_policy                  what to do with a rejected write
//      hbm_peak_bandwidth_gbps           denominators for utilisation stats
//      hbf_peak_bandwidth_gbps
//
// ============================================================================
#ifndef H3_ADDRESS_ROUTER_H
#define H3_ADDRESS_ROUTER_H

#include <cstdint>
#include <iosfwd>
#include <string>

namespace h3 {

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

enum class MemRegion : int { Hbm = 0, Hbf = 1, Unmapped = 2 };

enum class ReqType : int { Read = 0, Write = 1 };

const char* to_string(MemRegion r);
const char* to_string(ReqType t);

// A memory request as seen by the router. `opaque` carries the originating
// object (a GPGPU-Sim `mem_fetch*`) through the router untouched, so the
// router itself never needs to know about simulator types.
struct H3Request {
  uint64_t addr = 0;           // global address, H3 address space
  ReqType type = ReqType::Read;
  uint32_t size_bytes = 32;    // typically one 32 B sector
  uint64_t time_ps = 0;        // arrival time
  int source_id = -1;          // originating SM / sub-partition
  void* opaque = nullptr;      // passthrough payload (mem_fetch*)
};

enum class RouteStatus : int {
  RoutedHbm = 0,           // accepted, sent to HBM backend
  RoutedHbf = 1,           // accepted, sent to HBF backend (D2D charged)
  RejectedWriteToHbf,    // write to read-only HBF region -- a data
                            // placement bug in the workload, not a HW event
  RejectedUnmapped,        // address in neither region
  RejectedBackendFull,    // backend queue full; caller should retry
};

const char* to_string(RouteStatus s);
bool is_accepted(RouteStatus s);

// Result of routing one request.
struct RouteDecision {
  RouteStatus status = RouteStatus::RejectedUnmapped;
  MemRegion region = MemRegion::Unmapped;
  uint64_t local_addr = 0;    // address relative to the region base
  uint64_t issue_time_ps = 0; // request time + D2D hop latency (HBF only)
  uint32_t d2d_hops = 0;      // hops charged for this request
};

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------
// Implemented in Step 6 by thin wrappers around Ramulator memory systems, and
// in tests/test_h3_router.cpp by fakes. Keeping this abstract is what makes the
// router unit-testable without building the simulator.
class IMemoryBackend {
 public:
  virtual ~IMemoryBackend() = default;

  // Can this backend accept another request right now?
  virtual bool is_full(bool is_write) const = 0;

  // Enqueue. `local_addr` is region-relative; `issue_time_ps` already
  // includes any D2D hop latency. Returns false if the request was refused.
  virtual bool send(const H3Request& req, uint64_t local_addr,
                    uint64_t issue_time_ps) = 0;

  virtual const char* name() const = 0;
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// What to do when a write targets the read-only HBF region.
enum class HbfWritePolicy : int {
  Reject = 0,    // count it, drop it, return REJECTED_WRITE_TO_HBF (default)
  RedirectHbm,  // count it and send it to HBM instead (keeps sims running)
  Allow,         // count it and let it through (for HBF write experiments)
};

struct H3RouterConfig {
  // ---- Address map (H3 paper Fig. 3c) ----
  // Defaults are DERIVED FROM CAPACITY, not copied from the figure's literals:
  // the figure prints 0x030000000 (768 MB) and 0x1E000000000 (~2.06 TB), which
  // do not match its own stated 192 GB / 3 TB. Digits appear to be missing.
  // These defaults are self-consistent with the capacities; override in YAML
  // if you need the literal figure values.
  uint64_t hbm_base_addr = 0x0ULL;
  uint64_t hbm_size_bytes = 192ULL << 30;          // 192 GB  -> ends 0x3000000000
  uint64_t hbf_base_addr = 192ULL << 30;           // 0x3000000000
  uint64_t hbf_size_bytes = 3ULL << 40;            // 3 TB    -> ends 0x33000000000

  // ---- D2D interconnect ----
  // Physical routing overhead of crossing to the HBF base die. This is NOT the
  // NAND array latency (tR = 20 us), which the HBF Ramulator model applies.
  double d2d_hop_latency_ns = 25.0;
  uint32_t d2d_hops_to_hbf = 1;

  // ---- Read-only enforcement ----
  bool enforce_hbf_read_only = true;
  HbfWritePolicy hbf_write_policy = HbfWritePolicy::Reject;

  // ---- Peak bandwidths, used only as utilisation denominators ----
  double hbm_peak_bandwidth_gbps = 8000.0;   // 8 TB/s per GPU
  double hbf_peak_bandwidth_gbps = 8000.0;   // 8 TB/s per GPU

  // ---- Diagnostics ----
  bool warn_on_hbf_write = true;
  uint64_t max_warnings = 16;   // cap log spam from a misplaced tensor

  // Load from configs/h3_router_config.yaml. Throws std::runtime_error on a
  // missing file or an unknown key.
  static H3RouterConfig from_yaml(const std::string& path);

  // Validate ranges: non-zero sizes, no overlap, no wraparound.
  // Throws std::runtime_error describing the first problem found.
  void validate() const;

  uint64_t hbm_end_addr() const { return hbm_base_addr + hbm_size_bytes; }
  uint64_t hbf_end_addr() const { return hbf_base_addr + hbf_size_bytes; }
  uint64_t d2d_hop_latency_ps() const {
    return static_cast<uint64_t>(d2d_hop_latency_ns * 1000.0 + 0.5);
  }
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
struct H3RouterStats {
  uint64_t requests_total = 0;

  uint64_t requests_hbm = 0;
  uint64_t requests_hbf = 0;
  uint64_t reads_hbm = 0;
  uint64_t writes_hbm = 0;
  uint64_t reads_hbf = 0;
  uint64_t writes_hbf = 0;          // only non-zero if policy == ALLOW

  uint64_t bytes_hbm = 0;
  uint64_t bytes_hbf = 0;

  // Correctness counters. `write_attempts_to_hbf` MUST be 0 for a correct
  // data placement -- it is the router's main validation output.
  uint64_t write_attempts_to_hbf = 0;
  uint64_t writes_redirected_to_hbm = 0;
  uint64_t unmapped_requests = 0;
  uint64_t backend_full_rejects = 0;

  uint64_t d2d_hops = 0;

  // Observation window, for bandwidth utilisation.
  uint64_t first_time_ps = 0;
  uint64_t last_time_ps = 0;
  bool has_window = false;

  double elapsed_ns() const;
  double hbm_bandwidth_gbps() const;
  double hbf_bandwidth_gbps() const;
  double hbm_utilization(double peak_gbps) const;
  double hbf_utilization(double peak_gbps) const;
  double hbf_request_fraction() const;

  void reset();
};

// ---------------------------------------------------------------------------
// The router
// ---------------------------------------------------------------------------
class H3AddressRouter {
 public:
  // Backends may be null; a request routed to a null backend is still decoded
  // and counted but not delivered. This makes decode logic testable in
  // isolation from any memory model.
  H3AddressRouter(const H3RouterConfig& config,
                  IMemoryBackend* hbm_backend = nullptr,
                  IMemoryBackend* hbf_backend = nullptr);

  void set_backends(IMemoryBackend* hbm, IMemoryBackend* hbf);

  // Pure decode: which region owns this address, and where in it.
  // No side effects, no statistics -- safe to call from asserts and tests.
  MemRegion decode(uint64_t addr) const;
  uint64_t to_local_addr(uint64_t addr, MemRegion region) const;

  // Full path: decode, enforce read-only, charge D2D latency, forward to the
  // backend, update statistics.
  RouteDecision route(const H3Request& req);

  // Decode + policy WITHOUT forwarding or counting. Used by tests and by the
  // LHB (Step 4) to ask "where does this live?" without issuing anything.
  RouteDecision inspect(const H3Request& req) const;

  const H3RouterStats& stats() const { return m_stats; }
  const H3RouterConfig& config() const { return m_config; }
  void reset_stats() { m_stats.reset(); }

  void print_stats(std::ostream& os) const;
  void print_address_map(std::ostream& os) const;

 private:
  void note_time(uint64_t time_ps);
  void warn(const std::string& msg);

  H3RouterConfig m_config;
  IMemoryBackend* m_hbm = nullptr;
  IMemoryBackend* m_hbf = nullptr;
  H3RouterStats m_stats;
  uint64_t m_warnings_emitted = 0;
};

}  // namespace h3

#endif  // H3_ADDRESS_ROUTER_H
