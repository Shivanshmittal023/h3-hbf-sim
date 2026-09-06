// ============================================================================
//  h3_memory_backend.cpp -- implementation of the H3 memory backend
// ============================================================================
//  See h3_memory_backend.h. Depends only on the other h3-components and the
//  C++ standard library, so the whole stack stays unit-testable without
//  GPGPU-Sim or Ramulator.
// ============================================================================

#include "h3_memory_backend.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace h3 {

// ---------------------------------------------------------------------------
// AnalyticMemoryDevice
// ---------------------------------------------------------------------------

AnalyticMemoryDevice::AnalyticMemoryDevice(const char* name, double latency_ns,
                                           double bandwidth_gbps,
                                           uint32_t queue_depth)
    : m_name(name),
      m_latency_ps(static_cast<uint64_t>(latency_ns * 1000.0 + 0.5)),
      m_queue_depth(queue_depth) {
  if (bandwidth_gbps <= 0.0) {
    throw std::runtime_error("AnalyticMemoryDevice: bandwidth must be > 0");
  }
  // GB/s -> bytes per picosecond (1 GB/s == 1 B/ns == 1e-3 B/ps)
  m_bytes_per_ps = bandwidth_gbps * 1e-3;
}

bool AnalyticMemoryDevice::can_accept(bool /*is_write*/) const {
  return m_in_flight < m_queue_depth;
}

uint64_t AnalyticMemoryDevice::enqueue(uint64_t /*local_addr*/,
                                       uint32_t size_bytes, bool is_write,
                                       uint64_t issue_time_ps) {
  // Bandwidth is modeled as serialisation on a single shared data bus: the
  // device cannot start a transfer before it has finished the previous one.
  // This is what makes the analytic device saturate rather than accept
  // unlimited concurrency.
  const uint64_t transfer_ps =
      static_cast<uint64_t>(static_cast<double>(size_bytes) / m_bytes_per_ps + 0.5);
  const uint64_t start = std::max(issue_time_ps, m_next_free_ps);
  m_next_free_ps = start + transfer_ps;

  const uint64_t done = m_next_free_ps + m_latency_ps;
  m_completions.push_back(done);
  ++m_in_flight;

  if (is_write) ++m_writes; else ++m_reads;
  m_bytes += size_bytes;
  return done;
}

void AnalyticMemoryDevice::tick(uint64_t now_ps) {
  while (!m_completions.empty() && m_completions.front() <= now_ps) {
    m_completions.pop_front();
    if (m_in_flight) --m_in_flight;
  }
}

// ---------------------------------------------------------------------------
// Router -> device adapter
// ---------------------------------------------------------------------------
// The router speaks IMemoryBackend; the backend owns IH3MemoryDevice objects.
// This adapter bridges them and records the completion time the router's
// decision produced, which the backend then turns into a Pending entry.
class H3MemoryBackend::DeviceBackendAdapter : public IMemoryBackend {
 public:
  DeviceBackendAdapter(IH3MemoryDevice* device, const char* name)
      : m_device(device), m_name(name) {}

  bool is_full(bool is_write) const override {
    return m_device ? !m_device->can_accept(is_write) : false;
  }

  bool send(const H3Request& req, uint64_t local_addr,
            uint64_t issue_time_ps) override {
    if (!m_device) return false;
    last_completion_ps =
        m_device->enqueue(local_addr, req.size_bytes,
                          req.type == ReqType::Write, issue_time_ps);
    accepted = true;
    return true;
  }

  const char* name() const override { return m_name; }

  // Set by the most recent successful send().
  uint64_t last_completion_ps = 0;
  bool accepted = false;

 private:
  IH3MemoryDevice* m_device;
  const char* m_name;
};

// ---------------------------------------------------------------------------
// Minimal YAML reader (same flat subset used across h3-components)
// ---------------------------------------------------------------------------
namespace {

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}
std::string strip_comment(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) if (s[i] == '#') return s.substr(0, i);
  return s;
}
size_t indent_of(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') ++i;
  return i;
}
std::map<std::string, std::string> read_flat_yaml(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("H3BackendConfig: cannot open " + path);
  std::map<std::string, std::string> out;
  std::string section, line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const std::string raw = strip_comment(line);
    const std::string body = trim(raw);
    if (body.empty()) continue;
    const size_t colon = body.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("H3BackendConfig: " + path + ":" +
                               std::to_string(lineno) + ": expected 'key: value'");
    }
    const std::string key = trim(body.substr(0, colon));
    const std::string value = trim(body.substr(colon + 1));
    if (value.empty()) {
      section = (indent_of(raw) == 0) ? key : section + "." + key;
      continue;
    }
    out[section.empty() ? key : section + "." + key] = value;
  }
  return out;
}
double to_d(const std::string& k, const std::string& v) {
  try { return std::stod(trim(v)); }
  catch (const std::exception&) {
    throw std::runtime_error("H3BackendConfig: bad number for " + k + ": " + v);
  }
}
uint64_t to_u(const std::string& k, const std::string& v) {
  return static_cast<uint64_t>(to_d(k, v));
}
bool to_b(const std::string& k, const std::string& v) {
  std::string s = trim(v);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
  if (s == "false" || s == "0" || s == "no" || s == "off") return false;
  throw std::runtime_error("H3BackendConfig: expected boolean for " + k);
}

}  // namespace

H3BackendConfig H3BackendConfig::from_yaml(const std::string& path) {
  const auto kv = read_flat_yaml(path);
  H3BackendConfig c;

  static const char* known[] = {
      "backend.enabled", "backend.dram_clock_mhz", "backend.request_queue_size",
      "backend.return_queue_size", "backend.device_backend",
      "backend.enable_prefetch_scheduler",
      "analytic.hbm_latency_ns", "analytic.hbm_bandwidth_gbps",
      "analytic.hbf_latency_ns", "analytic.hbf_bandwidth_gbps",
      "analytic.hbf_max_outstanding", "analytic.hbm_max_outstanding",
      "ramulator.hbm_config", "ramulator.hbf_config",
      "configs.router_config", "configs.lhb_config", "configs.model_config",
  };
  for (const auto& e : kv) {
    bool ok = false;
    for (const char* k : known) if (e.first == k) { ok = true; break; }
    if (!ok) throw std::runtime_error("H3BackendConfig: unknown key '" + e.first + "' in " + path);
  }
  auto has = [&](const char* k) { return kv.find(k) != kv.end(); };

  if (has("backend.enabled")) c.enabled = to_b("enabled", kv.at("backend.enabled"));
  if (has("backend.dram_clock_mhz")) c.dram_clock_mhz = to_d("dram_clock_mhz", kv.at("backend.dram_clock_mhz"));
  if (has("backend.request_queue_size")) c.request_queue_size = static_cast<uint32_t>(to_u("request_queue_size", kv.at("backend.request_queue_size")));
  if (has("backend.return_queue_size")) c.return_queue_size = static_cast<uint32_t>(to_u("return_queue_size", kv.at("backend.return_queue_size")));
  if (has("backend.device_backend")) c.device_backend = trim(kv.at("backend.device_backend"));
  if (has("backend.enable_prefetch_scheduler")) c.enable_prefetch_scheduler = to_b("enable_prefetch_scheduler", kv.at("backend.enable_prefetch_scheduler"));
  if (has("analytic.hbm_latency_ns")) c.hbm_latency_ns = to_d("hbm_latency_ns", kv.at("analytic.hbm_latency_ns"));
  if (has("analytic.hbm_bandwidth_gbps")) c.hbm_bandwidth_gbps = to_d("hbm_bandwidth_gbps", kv.at("analytic.hbm_bandwidth_gbps"));
  if (has("analytic.hbf_latency_ns")) c.hbf_latency_ns = to_d("hbf_latency_ns", kv.at("analytic.hbf_latency_ns"));
  if (has("analytic.hbf_bandwidth_gbps")) c.hbf_bandwidth_gbps = to_d("hbf_bandwidth_gbps", kv.at("analytic.hbf_bandwidth_gbps"));
  if (has("analytic.hbf_max_outstanding")) c.hbf_max_outstanding = static_cast<uint32_t>(to_u("hbf_max_outstanding", kv.at("analytic.hbf_max_outstanding")));
  if (has("analytic.hbm_max_outstanding")) c.hbm_max_outstanding = static_cast<uint32_t>(to_u("hbm_max_outstanding", kv.at("analytic.hbm_max_outstanding")));
  if (has("ramulator.hbm_config")) c.ramulator_hbm_config = trim(kv.at("ramulator.hbm_config"));
  if (has("ramulator.hbf_config")) c.ramulator_hbf_config = trim(kv.at("ramulator.hbf_config"));
  if (has("configs.router_config")) c.router_config = trim(kv.at("configs.router_config"));
  if (has("configs.lhb_config")) c.lhb_config = trim(kv.at("configs.lhb_config"));
  if (has("configs.model_config")) c.model_config = trim(kv.at("configs.model_config"));

  c.validate();
  return c;
}

void H3BackendConfig::validate() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("H3BackendConfig: " + m); };
  if (dram_clock_mhz <= 0.0) fail("dram_clock_mhz must be > 0");
  if (request_queue_size == 0) fail("request_queue_size must be > 0");
  if (return_queue_size == 0) fail("return_queue_size must be > 0");
  if (device_backend != "analytic" && device_backend != "ramulator") {
    fail("device_backend must be 'analytic' or 'ramulator', got '" + device_backend + "'");
  }
#ifndef H3_WITH_RAMULATOR
  if (device_backend == "ramulator") {
    fail("device_backend = 'ramulator' but this build was compiled without "
         "H3_WITH_RAMULATOR. Rebuild with -DH3_WITH_RAMULATOR=ON, or use "
         "device_backend = 'analytic'.");
  }
#endif
}

// ---------------------------------------------------------------------------
// H3MemoryBackend
// ---------------------------------------------------------------------------

static std::unique_ptr<IH3MemoryDevice> make_device(const H3BackendConfig& cfg,
                                                    bool is_hbf,
                                                    unsigned num_partitions) {
  if (cfg.device_backend == "analytic") {
    const double lat_ns = is_hbf ? cfg.hbf_latency_ns : cfg.hbm_latency_ns;
    // Per-GPU bandwidth divided across the memory partitions.
    const double bw_gbps =
        (is_hbf ? cfg.hbf_bandwidth_gbps : cfg.hbm_bandwidth_gbps) /
        static_cast<double>(num_partitions ? num_partitions : 1);

    uint32_t depth;
    if (is_hbf) {
      depth = cfg.hbf_max_outstanding;   // structural cap; see the header
    } else if (cfg.hbm_max_outstanding) {
      depth = cfg.hbm_max_outstanding;
    } else {
      // Latency-bandwidth product in 32 B sectors: the concurrency needed to
      // keep the device at full bandwidth. Too small a queue silently caps
      // achievable bandwidth far below the configured value.
      const double lbp = lat_ns * bw_gbps / 32.0;
      depth = static_cast<uint32_t>(std::max(16.0, lbp));
    }
    return std::unique_ptr<IH3MemoryDevice>(new AnalyticMemoryDevice(
        is_hbf ? "HBF-analytic" : "HBM-analytic", lat_ns, bw_gbps, depth));
  }
#ifdef H3_WITH_RAMULATOR
  // See h3_ramulator_device.cpp: wraps Ramulator's ExternalFrontEnd +
  // IMemorySystem so HBM3e and HBF timing come from the real device models.
  return make_ramulator_device(is_hbf ? cfg.ramulator_hbf_config
                                      : cfg.ramulator_hbm_config,
                               is_hbf ? "HBF-ramulator" : "HBM-ramulator");
#else
  throw std::runtime_error(
      "H3MemoryBackend: ramulator backend requested but H3_WITH_RAMULATOR is "
      "not defined");
#endif
}

// Registry of live backends, for aggregate reporting. Not thread-safe, which
// matches GPGPU-Sim: partitions are constructed and cycled from one thread.
static std::vector<const H3MemoryBackend*>& h3_registry() {
  static std::vector<const H3MemoryBackend*> reg;
  return reg;
}

size_t H3MemoryBackend::live_backends() { return h3_registry().size(); }

H3MemoryBackend::H3MemoryBackend(unsigned partition_id,
                                 const H3BackendConfig& config,
                                 unsigned num_partitions)
    : m_partition_id(partition_id),
      m_num_partitions(num_partitions ? num_partitions : 1),
      m_config(config) {
  m_config.validate();
  h3_registry().push_back(this);

  // ---- Devices ----------------------------------------------------------
  m_hbm_device = make_device(m_config, /*is_hbf=*/false, m_num_partitions);
  m_hbf_device = make_device(m_config, /*is_hbf=*/true, m_num_partitions);
  m_hbm_adapter.reset(new DeviceBackendAdapter(m_hbm_device.get(), "HBM"));
  m_hbf_adapter.reset(new DeviceBackendAdapter(m_hbf_device.get(), "HBF"));

  // Asynchronous devices (Ramulator) report completions through this handler
  // during their tick(), which happens inside our cycle().
  auto handler = [this](uint64_t token, uint64_t now) {
    this->on_async_completion(token, now);
  };
  if (m_hbm_device->is_async()) m_hbm_device->set_completion_handler(handler);
  if (m_hbf_device->is_async()) m_hbf_device->set_completion_handler(handler);

  // ---- Router -----------------------------------------------------------
  H3RouterConfig rcfg;
  if (!m_config.router_config.empty()) {
    try {
      rcfg = H3RouterConfig::from_yaml(m_config.router_config);
    } catch (const std::exception& e) {
      // Defaults are already the paper's map; a missing file is not fatal, but
      // it must be visible rather than silently changing the address map.
      std::cerr << "[H3Backend] note: using default router config (" << e.what()
                << ")\n";
    }
  }
  m_router.reset(new H3AddressRouter(rcfg, m_hbm_adapter.get(), m_hbf_adapter.get()));

  // ---- LHB --------------------------------------------------------------
  LhbConfig lcfg;
  if (!m_config.lhb_config.empty()) {
    try {
      lcfg = LhbConfig::from_yaml(m_config.lhb_config);
    } catch (const std::exception& e) {
      std::cerr << "[H3Backend] note: using default LHB config (" << e.what() << ")\n";
    }
  }
  // The LHB is a per-GPU structure in the paper (8 cubes x 40 MB = 320 MB);
  // GPGPU-Sim gives us one backend per partition, so split it. Without this
  // the machine would carry num_partitions x the configured buffer.
  const double part = static_cast<double>(m_num_partitions);
  lcfg.buffer_size_bytes =
      std::max<uint64_t>(lcfg.num_buffers * 4096,
                         static_cast<uint64_t>(lcfg.buffer_size_bytes / part));
  // Keep the halves an exact division, which LhbConfig::validate() requires.
  lcfg.buffer_size_bytes -= lcfg.buffer_size_bytes % lcfg.num_buffers;
  lcfg.hbf_bandwidth_gbps /= part;
  // Each partition holds 1/N of every cache line in its window, so the window
  // must span N times its own capacity to cover the addresses the full
  // per-cube buffer would. See LhbConfig::address_interleave_factor.
  lcfg.address_interleave_factor = m_num_partitions;

  m_fill_engine.reset(new AnalyticFillEngine(lcfg.hbf_read_latency_ps(),
                                             lcfg.hbf_bandwidth_gbps));
  m_lhb.reset(new LatencyHidingBuffer(lcfg, m_fill_engine.get()));

  // ---- Prefetch scheduler ----------------------------------------------
  if (m_config.enable_prefetch_scheduler && !m_config.model_config.empty()) {
    try {
      ModelConfig mcfg = ModelConfig::from_yaml(m_config.model_config);
      m_scheduler.reset(new LLMPrefetchScheduler(mcfg));
    } catch (const std::exception& e) {
      std::cerr << "[H3Backend] note: prefetch scheduler disabled (" << e.what()
                << ")\n";
    }
  }
}

H3MemoryBackend::~H3MemoryBackend() {
  auto& reg = h3_registry();
  reg.erase(std::remove(reg.begin(), reg.end(), this), reg.end());
}

bool H3MemoryBackend::devices_can_accept(bool is_write) const {
  return m_hbm_device->can_accept(is_write) && m_hbf_device->can_accept(is_write);
}

bool H3MemoryBackend::full(bool is_write) const {
  if (m_pending.size() + m_async_pending.size() >= m_config.request_queue_size)
    return true;
  if (m_async_done.size() >= m_config.return_queue_size) return true;
  if (m_return_queue.size() >= m_config.return_queue_size) return true;
  // Device backpressure. Without this the router would reject requests that
  // the partition has already handed over, and push() would have to retire
  // them untimed. Reporting full here makes the partition RETRY next cycle,
  // which is the behaviour dram_t has and the behaviour the GPU should see.
  if (!devices_can_accept(is_write)) return true;
  return false;
}

bool H3MemoryBackend::returnq_full() const {
  return m_return_queue.size() >= m_config.return_queue_size;
}

void H3MemoryBackend::push(const H3MemRequest& req) {
  const uint64_t now = now_ps();

  H3Request r;
  r.addr = req.addr;
  r.type = req.is_write ? ReqType::Write : ReqType::Read;
  r.size_bytes = req.size_bytes;
  r.time_ps = now;
  r.source_id = req.source_id;
  r.opaque = req.opaque;

  m_hbm_adapter->accepted = false;
  m_hbf_adapter->accepted = false;

  const RouteDecision d = m_router->route(r);
  ++m_stats.pushed;

  uint64_t ready_ps = now;

  switch (d.status) {
    case RouteStatus::RoutedHbm:
      ready_ps = m_hbm_adapter->accepted ? m_hbm_adapter->last_completion_ps : now;
      break;

    case RouteStatus::RoutedHbf: {
      // The LHB sits between the router and the HBF device. If the tensor was
      // prefetched, the access costs SRAM latency instead of ~20 us; the
      // router has already charged the D2D hop in d.issue_time_ps.
      const LhbAccess a = m_lhb->access(d.local_addr, r.size_bytes, d.issue_time_ps);
      if (a.outcome == LhbOutcome::Hit) {
        // Served from SRAM: the HBF device is not touched at all.
        ready_ps = a.ready_time_ps;
      } else {
        // Miss (or bypass): the request really goes to HBF.
        ready_ps = m_hbf_adapter->accepted
                       ? std::max(m_hbf_adapter->last_completion_ps, a.ready_time_ps)
                       : a.ready_time_ps;
      }
      break;
    }

    case RouteStatus::RejectedWriteToHbf:
      // A data-placement bug. The router has counted it; complete the request
      // immediately so the simulation continues and the error is visible in
      // the stats rather than deadlocking the memory partition.
      ++m_stats.rejected_write_to_hbf;
      ready_ps = now;
      break;

    case RouteStatus::RejectedUnmapped:
      ++m_stats.rejected_unmapped;
      ready_ps = now;
      break;

    case RouteStatus::RejectedBackendFull:
      // Unreachable in normal operation: full() now reports device occupancy,
      // so the partition retries instead of pushing into a full device. If it
      // still happens, charge the request a full device latency rather than
      // retiring it for free -- a zero-latency completion would silently
      // understate memory cost and corrupt every result.
      ++m_stats.rejected_full;
      ready_ps = now + (decode_region_is_hbf(req.addr)
                            ? m_config.hbf_latency_ns
                            : m_config.hbm_latency_ns) * 1000ULL;
      if (m_stats.rejected_full == 1) {
        std::cerr << "[H3Backend] WARNING: request pushed into a full device "
                     "despite full() reporting space. Charging full latency; "
                     "see h3_backend_rejected_full in the stats.\n";
      }
      break;
  }

  // ---- Asynchronous devices --------------------------------------------
  // A real timing model does not know the completion time yet. Hand the
  // request over with a token and wait for its callback; do NOT invent a
  // ready time here.
  IH3MemoryDevice* target =
      (d.status == RouteStatus::RoutedHbf) ? m_hbf_device.get() : m_hbm_device.get();
  if (is_accepted(d.status) && target->is_async()) {
    // An LHB hit never reaches the device: it was served from SRAM.
    const bool served_by_lhb =
        (d.status == RouteStatus::RoutedHbf) && (ready_ps > now) &&
        (ready_ps - now <= m_lhb->config().sram_access_latency_ps());
    if (!served_by_lhb) {
      Pending ap;
      ap.issue_ps = now;
      ap.ready_ps = 0;            // unknown until the callback fires
      ap.opaque = req.opaque;
      ap.is_write = req.is_write;
      const uint64_t token = m_next_token++;
      if (target->enqueue_async(d.local_addr, req.size_bytes, req.is_write,
                                d.issue_time_ps, token)) {
        m_async_pending.emplace(token, ap);
        return;
      }
      // Refused despite full() reporting space: count it and fall through to
      // the synchronous path so the request is still timed, never dropped.
      ++m_stats.rejected_full;
    }
  }

  Pending p;
  p.issue_ps = now;
  p.ready_ps = std::max(ready_ps, now);
  p.opaque = req.opaque;
  p.is_write = req.is_write;

  // Keep the queue ordered by ready time so retire_ready() only inspects the
  // front. Insertions are almost always at the back, so this is cheap.
  auto it = std::upper_bound(
      m_pending.begin(), m_pending.end(), p.ready_ps,
      [](uint64_t v, const Pending& e) { return v < e.ready_ps; });
  m_pending.insert(it, p);
}

bool H3MemoryBackend::decode_region_is_hbf(uint64_t addr) const {
  return m_router->decode(addr) == MemRegion::Hbf;
}

void H3MemoryBackend::on_async_completion(uint64_t token, uint64_t now_ps) {
  auto it = m_async_pending.find(token);
  if (it == m_async_pending.end()) return;   // already retired, or not ours
  Pending p = it->second;
  m_async_pending.erase(it);
  p.ready_ps = std::max(now_ps, p.issue_ps);
  // Buffered rather than pushed straight to the return queue: the queue may be
  // full, and a device callback must never block or drop a completion.
  m_async_done.push_back(p);
}

void H3MemoryBackend::retire_ready(uint64_t now) {
  // Async completions first -- they are already ordered by completion time.
  while (!m_async_done.empty() &&
         m_return_queue.size() < m_config.return_queue_size) {
    const Pending& p = m_async_done.front();
    const uint64_t lat = p.ready_ps - p.issue_ps;
    m_stats.total_latency_ps += lat;
    m_stats.max_latency_ps = std::max(m_stats.max_latency_ps, lat);
    ++m_stats.completed;
    m_return_queue.push_back(p.opaque);
    m_async_done.pop_front();
  }

  while (!m_pending.empty() && m_pending.front().ready_ps <= now &&
         m_return_queue.size() < m_config.return_queue_size) {
    const Pending& p = m_pending.front();
    const uint64_t lat = p.ready_ps - p.issue_ps;
    m_stats.total_latency_ps += lat;
    m_stats.max_latency_ps = std::max(m_stats.max_latency_ps, lat);
    ++m_stats.completed;
    m_return_queue.push_back(p.opaque);
    m_pending.pop_front();
  }
}

void H3MemoryBackend::cycle() {
  ++m_cycle;
  const uint64_t now = now_ps();

  // 1. Issue any prefetch hints that have come due. This is what keeps the
  //    LHB ahead of demand; without it every HBF access pays the full tR.
  if (m_scheduler) m_scheduler->tick(now, m_lhb.get());

  // 2. Advance the LHB (retire completed fills, launch new ones).
  m_lhb->tick(now);

  // 3. Advance the memory devices.
  m_hbm_device->tick(now);
  m_hbf_device->tick(now);

  // 4. Move completed requests into the return queue.
  retire_ready(now);
}

void* H3MemoryBackend::return_queue_top() const {
  return m_return_queue.empty() ? nullptr : m_return_queue.front();
}

void H3MemoryBackend::return_queue_pop() {
  if (!m_return_queue.empty()) m_return_queue.pop_front();
}

void H3MemoryBackend::set_dram_power_stats(unsigned& n_cmd, unsigned& n_activity,
                                           unsigned& n_nop, unsigned& n_act,
                                           unsigned& n_pre, unsigned& n_rd,
                                           unsigned& n_wr, unsigned& n_wr_WB,
                                           unsigned& n_req) const {
  // AccelWattch expects DRAM command counts. The H3 devices are not
  // command-level models (the analytic ones certainly are not), so reads and
  // writes are reported directly and the row-command counters are derived
  // rather than measured. Treat H3 DRAM power numbers as first-order.
  const uint64_t rd = m_hbm_device->reads() + m_hbf_device->reads();
  const uint64_t wr = m_hbm_device->writes() + m_hbf_device->writes();
  n_rd = static_cast<unsigned>(rd);
  n_wr = static_cast<unsigned>(wr);
  n_wr_WB = 0;
  n_req = static_cast<unsigned>(m_stats.pushed);
  n_act = static_cast<unsigned>(rd + wr);   // one row activation per access
  n_pre = n_act;
  n_cmd = static_cast<unsigned>(m_cycle);
  n_activity = n_rd + n_wr + n_act + n_pre;
  n_nop = n_cmd > n_activity ? n_cmd - n_activity : 0;
}

void H3MemoryBackend::print_stats(std::ostream& os) const {
  // GPGPU-Sim `key = value` style, so util/job_launching/get_stats.py picks
  // these up without any change to its parsing.
  const std::string p = "h3_part" + std::to_string(m_partition_id) + "_";
  os << std::dec;
  os << p << "pushed = " << m_stats.pushed << "\n";
  os << p << "completed = " << m_stats.completed << "\n";
  os << p << "avg_latency_ns = " << std::fixed << std::setprecision(2)
     << m_stats.avg_latency_ns() << "\n";
  os << p << "max_latency_ns = " << std::fixed << std::setprecision(2)
     << static_cast<double>(m_stats.max_latency_ps) / 1000.0 << "\n";
  os << p << "rejected_write_to_hbf = " << m_stats.rejected_write_to_hbf << "\n";
  os << p << "rejected_unmapped = " << m_stats.rejected_unmapped << "\n";
  os << p << "hbm_device_reads = " << m_hbm_device->reads() << "\n";
  os << p << "hbm_device_writes = " << m_hbm_device->writes() << "\n";
  os << p << "hbm_device_bytes = " << m_hbm_device->bytes() << "\n";
  os << p << "hbf_device_reads = " << m_hbf_device->reads() << "\n";
  os << p << "hbf_device_bytes = " << m_hbf_device->bytes() << "\n";
  os << p << "dram_cycles = " << m_cycle << "\n";
  m_router->print_stats(os);
  m_lhb->print_stats(os, 1.98);
}

void H3MemoryBackend::print_aggregate_stats(std::ostream& os) {
  const auto& reg = h3_registry();
  if (reg.empty()) return;

  // ---- Sum raw counters across every memory partition --------------------
  H3BackendStats be{};
  H3RouterStats rt{};
  LhbStats lb{};
  uint64_t hbm_dev_reads = 0, hbm_dev_writes = 0, hbm_dev_bytes = 0;
  uint64_t hbf_dev_reads = 0, hbf_dev_bytes = 0, cycles = 0;
  uint64_t lhb_capacity = 0, lhb_required = 0;
  bool lhb_enabled = false;

  for (const auto* b : reg) {
    const auto& s = b->stats();
    be.pushed += s.pushed;
    be.completed += s.completed;
    be.rejected_full += s.rejected_full;
    be.rejected_write_to_hbf += s.rejected_write_to_hbf;
    be.rejected_unmapped += s.rejected_unmapped;
    be.total_latency_ps += s.total_latency_ps;
    be.max_latency_ps = std::max(be.max_latency_ps, s.max_latency_ps);

    const auto& r = b->router().stats();
    rt.requests_total += r.requests_total;
    rt.requests_hbm += r.requests_hbm;
    rt.requests_hbf += r.requests_hbf;
    rt.reads_hbm += r.reads_hbm;   rt.writes_hbm += r.writes_hbm;
    rt.reads_hbf += r.reads_hbf;   rt.writes_hbf += r.writes_hbf;
    rt.bytes_hbm += r.bytes_hbm;   rt.bytes_hbf += r.bytes_hbf;
    rt.write_attempts_to_hbf += r.write_attempts_to_hbf;
    rt.writes_redirected_to_hbm += r.writes_redirected_to_hbm;
    rt.unmapped_requests += r.unmapped_requests;
    rt.backend_full_rejects += r.backend_full_rejects;
    rt.d2d_hops += r.d2d_hops;
    if (r.has_window) {
      rt.first_time_ps = rt.has_window ? std::min(rt.first_time_ps, r.first_time_ps)
                                       : r.first_time_ps;
      rt.last_time_ps = std::max(rt.last_time_ps, r.last_time_ps);
      rt.has_window = true;
    }

    const auto& l = b->lhb().stats();
    lb.accesses += l.accesses;
    lb.hits += l.hits;
    lb.misses_late += l.misses_late;
    lb.misses_absent += l.misses_absent;
    lb.bypasses += l.bypasses;
    lb.total_stall_ps += l.total_stall_ps;
    lb.max_stall_ps = std::max(lb.max_stall_ps, l.max_stall_ps);
    lb.bytes_served += l.bytes_served;
    lb.bytes_prefetched += l.bytes_prefetched;
    lb.bytes_wasted += l.bytes_wasted;
    lb.fills_started += l.fills_started;
    lb.fills_completed += l.fills_completed;
    lb.swaps += l.swaps;
    lb.hints_received += l.hints_received;
    lb.occupancy_byte_ps += l.occupancy_byte_ps;
    if (l.has_window) {
      lb.first_time_ps = lb.has_window ? std::min(lb.first_time_ps, l.first_time_ps)
                                       : l.first_time_ps;
      lb.last_time_ps = std::max(lb.last_time_ps, l.last_time_ps);
      lb.has_window = true;
    }

    hbm_dev_reads += b->m_hbm_device->reads();
    hbm_dev_writes += b->m_hbm_device->writes();
    hbm_dev_bytes += b->m_hbm_device->bytes();
    hbf_dev_reads += b->m_hbf_device->reads();
    hbf_dev_bytes += b->m_hbf_device->bytes();
    cycles = std::max(cycles, b->current_cycle());

    lhb_capacity += b->lhb().config().buffer_size_bytes;
    lhb_required += b->lhb().config().required_capacity_bytes();
    lhb_enabled = lhb_enabled || b->lhb().config().enabled;
  }

  const auto& cfg0 = reg.front()->router().config();
  auto pct = [](uint64_t num, uint64_t den) {
    return den ? static_cast<double>(num) / static_cast<double>(den) : 0.0;
  };
  const double elapsed_ns = rt.elapsed_ns();
  const double hbm_bw = elapsed_ns > 0 ? rt.bytes_hbm / elapsed_ns : 0.0;
  const double hbf_bw = elapsed_ns > 0 ? rt.bytes_hbf / elapsed_ns : 0.0;
  const uint64_t lhb_reqs = lb.hits + lb.misses_late + lb.misses_absent;

  os << std::dec;
  os << "\n----------- H3 hybrid memory backend (aggregate over "
     << reg.size() << " memory partitions) -----------\n";
  os << "h3_memory_partitions = " << reg.size() << "\n";
  os << "h3_dram_cycles = " << cycles << "\n";
  os << "h3_router_requests_total = " << rt.requests_total << "\n";
  os << "h3_router_requests_hbm = " << rt.requests_hbm << "\n";
  os << "h3_router_requests_hbf = " << rt.requests_hbf << "\n";
  os << "h3_router_reads_hbm = " << rt.reads_hbm << "\n";
  os << "h3_router_writes_hbm = " << rt.writes_hbm << "\n";
  os << "h3_router_reads_hbf = " << rt.reads_hbf << "\n";
  os << "h3_router_writes_hbf = " << rt.writes_hbf << "\n";
  os << "h3_router_bytes_hbm = " << rt.bytes_hbm << "\n";
  os << "h3_router_bytes_hbf = " << rt.bytes_hbf << "\n";
  os << "h3_router_d2d_hops = " << rt.d2d_hops << "\n";
  os << std::fixed << std::setprecision(4);
  os << "h3_router_hbf_request_fraction = "
     << pct(rt.requests_hbf, rt.requests_hbm + rt.requests_hbf) << "\n";
  os << std::setprecision(1);
  os << "h3_router_elapsed_ns = " << elapsed_ns << "\n";
  os << std::setprecision(2);
  os << "h3_router_hbm_bandwidth_gbps = " << hbm_bw << "\n";
  os << "h3_router_hbf_bandwidth_gbps = " << hbf_bw << "\n";
  os << std::setprecision(4);
  os << "h3_router_hbm_utilization = "
     << (cfg0.hbm_peak_bandwidth_gbps > 0 ? hbm_bw / cfg0.hbm_peak_bandwidth_gbps : 0.0) << "\n";
  os << "h3_router_hbf_utilization = "
     << (cfg0.hbf_peak_bandwidth_gbps > 0 ? hbf_bw / cfg0.hbf_peak_bandwidth_gbps : 0.0) << "\n";

  // Correctness counters -- these MUST be zero in a healthy run.
  os << "h3_router_write_attempts_to_hbf = " << rt.write_attempts_to_hbf << "\n";
  os << "h3_router_writes_redirected = " << rt.writes_redirected_to_hbm << "\n";
  os << "h3_router_unmapped_requests = " << rt.unmapped_requests << "\n";
  os << "h3_router_backend_full_rejects = " << rt.backend_full_rejects << "\n";

  os << "lhb_enabled = " << (lhb_enabled ? 1 : 0) << "\n";
  os << "lhb_capacity_bytes = " << lhb_capacity << "\n";
  os << "lhb_required_capacity_eq1_bytes = " << lhb_required << "\n";
  os << "lhb_accesses = " << lb.accesses << "\n";
  os << "lhb_hits = " << lb.hits << "\n";
  os << "lhb_misses_late = " << lb.misses_late << "\n";
  os << "lhb_misses_absent = " << lb.misses_absent << "\n";
  os << "lhb_bypasses = " << lb.bypasses << "\n";
  os << "lhb_hit_rate = " << pct(lb.hits, lhb_reqs) << "\n";
  os << "lhb_miss_rate = " << pct(lb.misses_late + lb.misses_absent, lhb_reqs) << "\n";
  os << std::setprecision(1);
  os << "lhb_total_stall_ns = " << static_cast<double>(lb.total_stall_ps) / 1000.0 << "\n";
  os << "lhb_avg_stall_ns_per_miss = "
     << (lb.misses_late + lb.misses_absent
             ? static_cast<double>(lb.total_stall_ps) / 1000.0 /
                   static_cast<double>(lb.misses_late + lb.misses_absent)
             : 0.0)
     << "\n";
  os << "lhb_max_stall_ns = " << static_cast<double>(lb.max_stall_ps) / 1000.0 << "\n";
  os << "lhb_bytes_served = " << lb.bytes_served << "\n";
  os << "lhb_bytes_prefetched = " << lb.bytes_prefetched << "\n";
  os << "lhb_bytes_wasted = " << lb.bytes_wasted << "\n";
  os << std::setprecision(4);
  os << "lhb_prefetch_efficiency = " << pct(lb.bytes_served, lb.bytes_prefetched) << "\n";
  os << "lhb_fills_started = " << lb.fills_started << "\n";
  os << "lhb_fills_completed = " << lb.fills_completed << "\n";
  os << "lhb_swaps = " << lb.swaps << "\n";
  os << "lhb_hints_received = " << lb.hints_received << "\n";
  os << "lhb_avg_utilization = "
     << (lhb_capacity ? lb.avg_occupancy_bytes() / static_cast<double>(lhb_capacity) : 0.0)
     << "\n";
  os << std::setprecision(2);
  os << "h3_backend_pushed = " << be.pushed << "\n";
  os << "h3_backend_completed = " << be.completed << "\n";
  os << "h3_backend_avg_latency_ns = " << be.avg_latency_ns() << "\n";
  os << "h3_backend_max_latency_ns = "
     << static_cast<double>(be.max_latency_ps) / 1000.0 << "\n";
  os << "h3_hbm_device_reads = " << hbm_dev_reads << "\n";
  os << "h3_hbm_device_writes = " << hbm_dev_writes << "\n";
  os << "h3_hbm_device_bytes = " << hbm_dev_bytes << "\n";
  os << "h3_hbf_device_reads = " << hbf_dev_reads << "\n";
  os << "h3_hbf_device_bytes = " << hbf_dev_bytes << "\n";
  os << "----------- end H3 statistics -----------\n\n";
}

}  // namespace h3
