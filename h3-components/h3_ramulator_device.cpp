// ============================================================================
//  h3_ramulator_device.cpp -- Ramulator 2.1 timing model behind IH3MemoryDevice
// ============================================================================
//
//  WHAT THIS IS
//      The bridge between the H3 memory backend and a real Ramulator 2.1
//      memory system. It replaces AnalyticMemoryDevice's closed-form
//      `latency + size/bandwidth` with actual command scheduling, bank
//      conflicts, row-buffer locality and queueing.
//
//      One instance models ONE memory (HBM3e or HBF) for ONE GPGPU-Sim memory
//      partition.
//
//  WHY THE INTERFACE IS ASYNCHRONOUS
//      AnalyticMemoryDevice returns a completion time from enqueue(). Ramulator
//      cannot: a request's latency depends on arbitration and queueing that
//      have not happened yet. Completions arrive later, via `req.callback(req)`
//      inside ControllerBase::tick(). So this device reports is_async() == true
//      and delivers completions through IH3MemoryDevice::CompletionHandler.
//      Inventing a completion time at enqueue would discard exactly the timing
//      this model exists to produce.
//
//  WHAT IT CONNECTS TO
//      Upstream   : H3MemoryBackend, via make_ramulator_device()
//      Downstream : Ramulator's ExternalFrontEnd + IMemorySystem, built by
//                   Factory from a machine-generated YAML config
//                   (scripts/export_hbf_yaml.sh)
//
//  BUILD
//      Compiled ONLY when H3_WITH_RAMULATOR is defined:
//          cmake -DH3_WITH_RAMULATOR=ON ...
//      Without it the whole file is empty, so h3-components still builds and
//      unit-tests with a bare C++17 toolchain and no Ramulator checkout.
//
//      NOTE: Ramulator is C++20; the rest of h3-components is C++17. They meet
//      only at the IH3MemoryDevice boundary, and h3-components/CMakeLists.txt
//      compiles this one translation unit with -std=c++20.
//
//  CLOCK DOMAINS
//      GPGPU-Sim ticks the backend once per DRAM clock and hands us absolute
//      picoseconds. Ramulator advances in its own tCK, obtained at runtime from
//      IMemorySystem::get_tCK() (nanoseconds). tick(now_ps) advances Ramulator
//      by however many of its ticks have elapsed, so the two clocks stay
//      aligned no matter how they are configured.
//
//  CONFIGURABLE PARAMETERS
//      The Ramulator YAML itself (configs/generated/*.yaml), selected by
//      `ramulator.hbm_config` / `ramulator.hbf_config` in
//      configs/h3_backend_config.yaml.
// ============================================================================

#include "h3_memory_backend.h"

#ifdef H3_WITH_RAMULATOR

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ramulator/base/base.h"
#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace h3 {
namespace {

class RamulatorMemoryDevice : public IH3MemoryDevice {
 public:
  RamulatorMemoryDevice(const std::string& config_path, const char* name)
      : m_name(name) {
    Ramulator::ConfigNode cfg;
    try {
      cfg = Ramulator::Config::parse_config_file(config_path);
    } catch (const std::exception& e) {
      throw std::runtime_error("RamulatorMemoryDevice(" + m_name +
                               "): cannot load config '" + config_path +
                               "': " + e.what() +
                               "\n  Generate it with ./scripts/export_hbf_yaml.sh");
    }

    m_frontend.reset(Ramulator::Factory::create_frontend(cfg));
    m_memory_system.reset(Ramulator::Factory::create_memory_system(cfg));
    if (!m_frontend || !m_memory_system) {
      throw std::runtime_error("RamulatorMemoryDevice(" + m_name +
                               "): config '" + config_path +
                               "' did not produce a frontend and a memory "
                               "system. The frontend impl must be 'External'.");
    }

    // Same wiring order the reference driver uses (python/bindings.cpp).
    m_frontend->connect_memory_system(m_memory_system.get());
    m_memory_system->connect_frontend(m_frontend.get());

    // tCK in ns -> ps. Anything non-positive means the memory system does not
    // expose a clock, and we cannot align the two domains.
    const float tck_ns = m_memory_system->get_tCK();
    if (!(tck_ns > 0.0f)) {
      throw std::runtime_error("RamulatorMemoryDevice(" + m_name +
                               "): memory system reports no tCK; cannot align "
                               "clock domains");
    }
    m_tck_ps = static_cast<double>(tck_ns) * 1000.0;
    m_tx_bytes = m_memory_system->get_tx_bytes();

    // Ramulator interleaves frontend and memory ticks by their clock ratios.
    m_fe_ratio = std::max(1, m_frontend->get_clock_ratio());
    m_mem_ratio = std::max(1, m_memory_system->get_clock_ratio());
  }

  ~RamulatorMemoryDevice() override {
    try {
      m_frontend->finalize();
      m_memory_system->finalize();
    } catch (...) {
      // Never throw from a destructor during simulator teardown.
    }
  }

  bool is_async() const override { return true; }

  void set_completion_handler(CompletionHandler handler) override {
    m_handler = std::move(handler);
  }

  // Ramulator applies its own backpressure by refusing a send, so the only
  // limit here is our outstanding-request bookkeeping.
  bool can_accept(bool /*is_write*/) const override {
    return m_outstanding < kMaxOutstanding;
  }

  bool enqueue_async(uint64_t local_addr, uint32_t size_bytes, bool is_write,
                     uint64_t issue_time_ps, uint64_t token) override {
    // Requests may be issued "in the future" (the router charges a D2D hop
    // before HBF accesses). Ramulator has no notion of a future arrival, so
    // hold the request until our clock reaches its issue time.
    if (issue_time_ps > m_now_ps) {
      m_deferred.push_back({local_addr, size_bytes, is_write, issue_time_ps, token});
      ++m_outstanding;
      return true;
    }
    return submit(local_addr, size_bytes, is_write, token);
  }

  // The synchronous path is unreachable for this device (is_async() is true),
  // but the interface requires it. Fail loudly rather than fabricate a time.
  uint64_t enqueue(uint64_t /*local_addr*/, uint32_t /*size_bytes*/,
                   bool /*is_write*/, uint64_t issue_time_ps) override {
    throw std::runtime_error(
        "RamulatorMemoryDevice(" + m_name +
        "): synchronous enqueue() called on an asynchronous device. The "
        "backend must use enqueue_async(); this would have invented a "
        "completion time.");
    return issue_time_ps;
  }

  void tick(uint64_t now_ps) override {
    m_now_ps = now_ps;

    // Release requests whose issue time has arrived.
    for (auto it = m_deferred.begin(); it != m_deferred.end();) {
      if (it->issue_ps <= now_ps) {
        if (submit(it->addr, it->size, it->is_write, it->token)) {
          --m_outstanding;   // submit() re-increments; avoid double counting
          it = m_deferred.erase(it);
          continue;
        }
      }
      ++it;
    }

    // Advance Ramulator to match elapsed wall time, interleaving frontend and
    // memory ticks exactly as the reference driver does.
    const uint64_t target = static_cast<uint64_t>(
        static_cast<double>(now_ps) / m_tck_ps);
    uint64_t guard = 0;
    while (m_ticks < target) {
      if (++m_fe_count >= m_mem_ratio) {
        m_fe_count = 0;
        m_frontend->tick();
      }
      if (++m_mem_count >= m_fe_ratio) {
        m_mem_count = 0;
        m_memory_system->tick();
      }
      ++m_ticks;
      // A pathological clock mismatch would otherwise spin here for millions of
      // iterations inside a single GPGPU-Sim cycle.
      if (++guard > kMaxTicksPerCall) {
        m_ticks = target;
        if (!m_warned_tick_flood) {
          m_warned_tick_flood = true;
          std::cerr << "[H3Ramulator] WARNING: " << m_name << " needed more than "
                    << kMaxTicksPerCall << " ticks in one call; check that "
                    << "backend.dram_clock_mhz matches the Ramulator config.\n";
        }
        break;
      }
    }
  }

  const char* name() const override { return m_name.c_str(); }
  uint64_t reads() const override { return m_reads; }
  uint64_t writes() const override { return m_writes; }
  uint64_t bytes() const override { return m_bytes; }

 private:
  struct Deferred {
    uint64_t addr;
    uint32_t size;
    bool is_write;
    uint64_t issue_ps;
    uint64_t token;
  };

  bool submit(uint64_t local_addr, uint32_t size_bytes, bool is_write,
              uint64_t token) {
    const int type_id = is_write ? Ramulator::Request::Type::Write
                                 : Ramulator::Request::Type::Read;
    // Captured by value: the callback outlives this call.
    auto cb = [this, token](Ramulator::Request& /*req*/) {
      if (m_outstanding) --m_outstanding;
      if (m_handler) m_handler(token, m_now_ps);
    };
    const bool ok = m_frontend->receive_external_requests(
        type_id, static_cast<Ramulator::Addr_t>(local_addr), /*source_id=*/0,
        cb, static_cast<int>(size_bytes));
    if (ok) {
      ++m_outstanding;
      if (is_write) ++m_writes; else ++m_reads;
      m_bytes += size_bytes;
    }
    return ok;
  }

  static constexpr uint32_t kMaxOutstanding = 4096;
  static constexpr uint64_t kMaxTicksPerCall = 100000;

  std::string m_name;
  std::unique_ptr<Ramulator::IFrontEnd> m_frontend;
  std::unique_ptr<Ramulator::IMemorySystem> m_memory_system;
  CompletionHandler m_handler;

  double m_tck_ps = 1.0;
  int m_tx_bytes = 32;
  int m_fe_ratio = 1, m_mem_ratio = 1;
  int m_fe_count = 0, m_mem_count = 0;

  uint64_t m_now_ps = 0;
  uint64_t m_ticks = 0;
  uint32_t m_outstanding = 0;
  bool m_warned_tick_flood = false;

  std::vector<Deferred> m_deferred;
  uint64_t m_reads = 0, m_writes = 0, m_bytes = 0;
};

}  // namespace

std::unique_ptr<IH3MemoryDevice> make_ramulator_device(const std::string& config_path,
                                                       const char* name) {
  return std::unique_ptr<IH3MemoryDevice>(
      new RamulatorMemoryDevice(config_path, name));
}

}  // namespace h3

#endif  // H3_WITH_RAMULATOR
