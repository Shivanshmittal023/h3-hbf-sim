// ============================================================================
//  h3_memory_backend.h -- H3 memory backend for Accel-Sim / GPGPU-Sim
// ============================================================================
//
//  WHAT THIS IS
//      The single object that replaces GPGPU-Sim's `dram_t` when the H3 memory
//      system is enabled. It presents the SAME request/response contract as
//      dram_t and internally owns the whole H3 stack:
//
//          push(req)                        <- from memory_partition_unit
//              |
//              v
//          H3AddressRouter          decode HBM vs HBF, charge D2D, reject
//              |         \                  writes to read-only HBF
//              |          \
//         HBM device       HBF path
//        (IH3MemoryDevice)     |
//                              v
//                     LatencyHidingBuffer  <- fed by LLMPrefetchScheduler
//                              |
//                              v
//                         HBF device (IH3MemoryDevice)
//              |
//              v
//          return queue  -> return_queue_top()/pop() -> memory_partition_unit
//
//  WHY IT TAKES `void*` INSTEAD OF `mem_fetch*`
//      This header deliberately includes NOTHING from GPGPU-Sim. The
//      originating `mem_fetch*` travels through as an opaque pointer. That
//      keeps the entire H3 stack compilable and unit-testable with a bare
//      `g++ -std=c++17`, and means a GPGPU-Sim API change cannot silently
//      break H3 logic. The thin conversion lives in the GPGPU-Sim patch
//      (patches/gpgpu-sim-h3-backend.patch), which is the only file that sees
//      both worlds.
//
//  THE dram_t CONTRACT WE REPRODUCE
//      bool  full(bool is_write) const     backpressure; caller RETRIES
//      void  push(request)                 enqueue
//      void  cycle()                       advance one DRAM clock
//      void* return_queue_top()            completed request, or nullptr
//      void  return_queue_pop()            retire it
//      bool  returnq_full() const
//      set_dram_power_stats(...)           AccelWattch counters
//
//      IMPORTANT: a `full()` rejection is a RETRY signal, not a drop. The
//      memory partition re-offers the request on the next cycle. Dropping it
//      silently loses memory traffic and quietly corrupts every result.
//
//  CLOCK DOMAINS
//      GPGPU-Sim ticks this from the DRAM clock domain and counts in CYCLES.
//      Every H3 component works in PICOSECONDS. `dram_clock_mhz` converts, once,
//      here. Getting this wrong rescales every latency in the system, so it is
//      a single explicit parameter rather than an implicit assumption.
//
//  MEMORY DEVICES
//      `IH3MemoryDevice` abstracts "how long does this access take":
//        * AnalyticMemoryDevice -- latency + bandwidth serialisation. Always
//          available, no dependencies. Good for bring-up and fast sweeps.
//        * RamulatorMemoryDevice -- real Ramulator 2.1 timing. Compiled only
//          when H3_WITH_RAMULATOR is defined, so the H3 stack builds and runs
//          before Ramulator is available.
//
//  CONFIGURABLE PARAMETERS  (configs/h3_backend_config.yaml)
//      All sub-component configs, plus dram_clock_mhz, queue depths, and the
//      device backend selection.
// ============================================================================
#ifndef H3_MEMORY_BACKEND_H
#define H3_MEMORY_BACKEND_H

#include <cstdint>
#include <deque>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "h3_address_router.h"
#include "latency_hiding_buffer.h"
#include "llm_prefetch_scheduler.h"

namespace h3 {

// ---------------------------------------------------------------------------
// A request as it crosses the GPGPU-Sim <-> H3 boundary
// ---------------------------------------------------------------------------
struct H3MemRequest {
  uint64_t addr = 0;         // global address
  bool is_write = false;
  uint32_t size_bytes = 32;  // one sector
  void* opaque = nullptr;    // the originating mem_fetch*
  int source_id = -1;        // sub-partition id
};

// ---------------------------------------------------------------------------
// Memory device abstraction
// ---------------------------------------------------------------------------
class IH3MemoryDevice {
 public:
  virtual ~IH3MemoryDevice() = default;

  // Can this device accept another request now?
  virtual bool can_accept(bool is_write) const = 0;

  // Enqueue at `issue_time_ps`; returns the completion time.
  virtual uint64_t enqueue(uint64_t local_addr, uint32_t size_bytes,
                           bool is_write, uint64_t issue_time_ps) = 0;

  // Advance internal state to `now_ps` (a no-op for closed-form devices).
  virtual void tick(uint64_t /*now_ps*/) {}

  // ---- Asynchronous completion ------------------------------------------
  // Closed-form devices know the completion time at enqueue and return it from
  // enqueue(). A real timing model does not: Ramulator delivers completions via
  // a callback during tick(), after arbitration and queueing have played out.
  // Devices that work that way report is_async() == true, and the backend uses
  // enqueue_async() plus the completion handler instead of enqueue().
  //
  // Pretending a real model is synchronous would mean inventing a completion
  // time at push -- which is precisely the timing the model exists to compute.
  virtual bool is_async() const { return false; }

  // `token` identifies the request; the backend maps it back to its mem_fetch.
  using CompletionHandler = std::function<void(uint64_t token, uint64_t now_ps)>;
  virtual void set_completion_handler(CompletionHandler /*handler*/) {}

  // Returns false if the device refused the request (queue full). The caller
  // must retry -- never drop it.
  virtual bool enqueue_async(uint64_t /*local_addr*/, uint32_t /*size_bytes*/,
                             bool /*is_write*/, uint64_t /*issue_time_ps*/,
                             uint64_t /*token*/) {
    return false;
  }

  virtual const char* name() const = 0;

  // Counters used to populate AccelWattch's DRAM power inputs.
  virtual uint64_t reads() const = 0;
  virtual uint64_t writes() const = 0;
  virtual uint64_t bytes() const = 0;
};

#ifdef H3_WITH_RAMULATOR
// Both defined in h3_ramulator_device.cpp. They wrap a Ramulator 2.1 memory
// system (ExternalFrontEnd + IMemorySystem) behind the two H3 interfaces.
std::unique_ptr<IH3MemoryDevice> make_ramulator_device(const std::string& config_path,
                                                       const char* name);

// An LHB fill engine backed by an existing Ramulator device, so PREFETCH
// traffic is timed by the same model as demand traffic. Without this, the ~99.9%
// of HBF reads that the buffer serves would still be closed-form.
// `demand_handler` is the backend's own completion handler. The engine takes
// over the device's callback (it must see every sector), so it needs somewhere
// to forward completions that belong to demand requests rather than fills.
// Passing it in makes that hand-off explicit and impossible to forget.
std::unique_ptr<ILhbFillEngine> make_ramulator_fill_engine(
    IH3MemoryDevice* device, uint32_t sector_bytes, double fallback_latency_ns,
    double fallback_bw_gbps, IH3MemoryDevice::CompletionHandler demand_handler);
#endif

// Closed-form device: completion = max(now, next_free) + latency, with the
// device serialised at its bandwidth. Enough to reproduce first-order
// latency/bandwidth behaviour without Ramulator.
class AnalyticMemoryDevice : public IH3MemoryDevice {
 public:
  AnalyticMemoryDevice(const char* name, double latency_ns,
                       double bandwidth_gbps, uint32_t queue_depth = 64);

  bool can_accept(bool is_write) const override;
  uint64_t enqueue(uint64_t local_addr, uint32_t size_bytes, bool is_write,
                   uint64_t issue_time_ps) override;
  void tick(uint64_t now_ps) override;
  const char* name() const override { return m_name.c_str(); }
  uint64_t reads() const override { return m_reads; }
  uint64_t writes() const override { return m_writes; }
  uint64_t bytes() const override { return m_bytes; }

 private:
  std::string m_name;
  uint64_t m_latency_ps;
  double m_bytes_per_ps;
  uint32_t m_queue_depth;
  uint64_t m_next_free_ps = 0;
  uint32_t m_in_flight = 0;
  uint64_t m_reads = 0, m_writes = 0, m_bytes = 0;
  std::deque<uint64_t> m_completions;
};

// ---------------------------------------------------------------------------
// Backend configuration
// ---------------------------------------------------------------------------
struct H3BackendConfig {
  bool enabled = true;

  // Clock domain conversion. GPGPU-Sim counts DRAM cycles; H3 counts ps.
  double dram_clock_mhz = 3106.0;   // SM90_H100 default (-gpgpu_clock_domains)

  // Queue depths, mirroring dram_t's request/return queues.
  uint32_t request_queue_size = 64;
  uint32_t return_queue_size = 64;

  // "analytic" or "ramulator"
  std::string device_backend = "analytic";

  // Analytic device parameters (ignored when device_backend == "ramulator").
  //
  // BANDWIDTHS AND LHB CAPACITY ARE PER GPU. GPGPU-Sim instantiates one
  // backend per memory partition (40 on an H100 config), so the backend
  // divides these by the partition count at construction. Configuring
  // per-partition values here would silently give the simulated machine
  // 40x the bandwidth and 40x the buffer.
  double hbm_latency_ns = 350.0;      // typical HBM3e loaded latency
  double hbm_bandwidth_gbps = 8000.0; // PER GPU
  double hbf_latency_ns = 20000.0;    // tR = 20 us
  double hbf_bandwidth_gbps = 8000.0; // PER GPU

  // Outstanding-request limit for the HBF device, per partition.
  //
  // The latency-bandwidth product for HBF is enormous: 20 us x 200 GB/s per
  // partition / 32 B = ~125,000 outstanding sector requests. No controller has
  // that many MSHRs -- which is precisely WHY the Latency Hiding Buffer exists.
  // The LHB turns that demand stream into a handful of large sequential fills.
  // This cap models the real structural limit; raising it would model hardware
  // that cannot be built.
  uint32_t hbf_max_outstanding = 256;

  // HBM outstanding limit. 0 = derive from the latency-bandwidth product,
  // which is what a well-provisioned HBM controller actually sustains.
  uint32_t hbm_max_outstanding = 0;

  // Ramulator config paths (device_backend == "ramulator").
  std::string ramulator_hbm_config = "configs/generated/HBM3E_H3.yaml";
  std::string ramulator_hbf_config = "configs/generated/HBF_H3.yaml";

  // Sub-component config files. Empty means "use built-in defaults".
  std::string router_config = "configs/h3_router_config.yaml";
  std::string lhb_config = "configs/lhb_config.yaml";
  std::string model_config = "configs/llama_405b_config.yaml";

  // Drive the LHB from the deterministic LLM layer schedule. Turn off to
  // measure the LHB with demand traffic only (a useful ablation).
  bool enable_prefetch_scheduler = true;

  static H3BackendConfig from_yaml(const std::string& path);
  void validate() const;

  uint64_t cycles_to_ps(uint64_t cycles) const {
    return static_cast<uint64_t>(static_cast<double>(cycles) * 1e6 / dram_clock_mhz);
  }
};

// ---------------------------------------------------------------------------
// Backend statistics
// ---------------------------------------------------------------------------
struct H3BackendStats {
  uint64_t pushed = 0;
  uint64_t completed = 0;
  uint64_t rejected_full = 0;
  uint64_t rejected_write_to_hbf = 0;
  uint64_t rejected_unmapped = 0;
  uint64_t total_latency_ps = 0;
  uint64_t max_latency_ps = 0;

  double avg_latency_ns() const {
    return completed ? static_cast<double>(total_latency_ps) / completed / 1000.0 : 0.0;
  }
};

// ---------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------
class H3MemoryBackend {
 public:
  // `partition_id` mirrors dram_t's partition id: one backend per memory
  // partition, exactly as GPGPU-Sim instantiates one dram_t per partition.
  //
  // `num_partitions` is -gpgpu_n_mem. Per-GPU quantities in the config
  // (bandwidths, LHB capacity) are divided by it, so the sum across all
  // partitions equals the machine being modeled. Passing the real value from
  // GPGPU-Sim rather than duplicating it in YAML keeps the two from drifting.
  H3MemoryBackend(unsigned partition_id, const H3BackendConfig& config,
                  unsigned num_partitions = 1);
  ~H3MemoryBackend();

  // ---- dram_t-compatible contract ---------------------------------------
  // full() MUST account for device queue occupancy, not just this object's
  // queues. If it does not, the memory partition keeps pushing, the router
  // finds the target device full, and requests get retired without ever
  // being timed -- silently fabricating zero-latency memory accesses.
  bool full(bool is_write) const;
  void push(const H3MemRequest& req);
  void cycle();                       // advance one DRAM clock
  void* return_queue_top() const;
  void return_queue_pop();
  bool returnq_full() const;

  // AccelWattch DRAM power counters.
  void set_dram_power_stats(unsigned& n_cmd, unsigned& n_activity,
                            unsigned& n_nop, unsigned& n_act, unsigned& n_pre,
                            unsigned& n_rd, unsigned& n_wr, unsigned& n_wr_WB,
                            unsigned& n_req) const;

  // ---- Introspection ----------------------------------------------------
  uint64_t current_cycle() const { return m_cycle; }
  uint64_t now_ps() const { return m_config.cycles_to_ps(m_cycle); }

  const H3BackendStats& stats() const { return m_stats; }
  const H3AddressRouter& router() const { return *m_router; }
  const LatencyHidingBuffer& lhb() const { return *m_lhb; }

  // Emits this partition's counters in GPGPU-Sim's `key = value` stats style.
  void print_stats(std::ostream& os) const;

  // ---- Aggregate reporting ----------------------------------------------
  // GPGPU-Sim instantiates one backend PER MEMORY PARTITION (80 on an H100
  // config). Printing per-partition detail would emit 80 duplicate copies of
  // every counter, which breaks get_stats.py's `key = value` parsing and is
  // unreadable besides. Every live backend registers itself, and this prints
  // ONE aggregated block covering the whole GPU.
  //
  // Rates are recomputed from summed totals, never averaged across partitions:
  // averaging ratios would weight an idle partition the same as a saturated one.
  static void print_aggregate_stats(std::ostream& os);
  static size_t live_backends();

 private:
  struct Pending {
    uint64_t ready_ps = 0;
    uint64_t issue_ps = 0;
    void* opaque = nullptr;
    bool is_write = false;
  };

  void on_async_completion(uint64_t token, uint64_t now_ps);
  // Called when an LHB half's asynchronous fill lands: releases every request
  // that coalesced onto it.
  void on_fill_complete(int half, uint64_t now_ps);

  // Adapters letting the router push into IH3MemoryDevice instances.
  class DeviceBackendAdapter;

  void retire_ready(uint64_t now_ps);
  // True only if BOTH devices can take another request. The backend cannot
  // know which region a future request targets (dram_t::full() has no address
  // either), so backpressure is conservative: shared base-die queues in front
  // of both memories. Conservative is correct here; optimistic loses traffic.
  bool devices_can_accept(bool is_write) const;
  bool decode_region_is_hbf(uint64_t addr) const;

  unsigned m_partition_id = 0;
  unsigned m_num_partitions = 1;
  H3BackendConfig m_config;
  uint64_t m_cycle = 0;

  std::unique_ptr<IH3MemoryDevice> m_hbm_device;
  std::unique_ptr<IH3MemoryDevice> m_hbf_device;
  std::unique_ptr<DeviceBackendAdapter> m_hbm_adapter;
  std::unique_ptr<DeviceBackendAdapter> m_hbf_adapter;

  std::unique_ptr<H3AddressRouter> m_router;
  std::unique_ptr<LatencyHidingBuffer> m_lhb;
  std::unique_ptr<ILhbFillEngine> m_fill_engine;
  std::unique_ptr<LLMPrefetchScheduler> m_scheduler;

  std::deque<Pending> m_pending;      // in flight, ordered by ready time
  // Requests handed to an ASYNCHRONOUS device. Their completion time is not
  // known at push, so they cannot live in the ready-time-ordered deque above;
  // the device's callback moves them into m_async_done.
  std::unordered_map<uint64_t, Pending> m_async_pending;
  // Requests that hit a half whose fill was still in flight. They cannot be
  // given a ready time yet, so they wait here, keyed by half index, and are
  // released together when that fill lands. This is MSHR-style coalescing.
  std::unordered_map<int, std::vector<Pending>> m_fill_waiters;
  std::deque<Pending> m_async_done;
  uint64_t m_next_token = 1;
  std::deque<void*> m_return_queue;   // completed, awaiting the partition
  H3BackendStats m_stats;
};

}  // namespace h3

#endif  // H3_MEMORY_BACKEND_H
