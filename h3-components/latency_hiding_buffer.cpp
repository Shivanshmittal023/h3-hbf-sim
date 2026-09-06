// ============================================================================
//  latency_hiding_buffer.cpp -- implementation of the H3 Latency Hiding Buffer
// ============================================================================
//  See latency_hiding_buffer.h for the architectural description.
//  Depends only on the C++ standard library, so it is unit-testable without
//  Ramulator or Accel-Sim.
// ============================================================================

#include "latency_hiding_buffer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace h3 {

// ---------------------------------------------------------------------------
// Enum helpers
// ---------------------------------------------------------------------------

const char* to_string(LhbState s) {
  switch (s) {
    case LhbState::Idle: return "IDLE";
    case LhbState::PrefetchIssued: return "PREFETCH_ISSUED";
    case LhbState::Filling: return "FILLING";
    case LhbState::Ready: return "READY";
    case LhbState::Serving: return "SERVING";
    default: return "UNKNOWN";
  }
}

const char* to_string(LhbOutcome o) {
  switch (o) {
    case LhbOutcome::Hit: return "HIT";
    case LhbOutcome::MissLate: return "MISS_LATE";
    case LhbOutcome::MissAbsent: return "MISS_ABSENT";
    case LhbOutcome::Bypass: return "BYPASS";
    default: return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// AnalyticFillEngine
// ---------------------------------------------------------------------------

AnalyticFillEngine::AnalyticFillEngine(uint64_t read_latency_ps,
                                       double bandwidth_gbps)
    : m_read_latency_ps(read_latency_ps) {
  // GB/s -> bytes per picosecond.  1 GB/s == 1 byte/ns == 1e-3 bytes/ps.
  m_bytes_per_ps = bandwidth_gbps * 1e-3;
  if (m_bytes_per_ps <= 0.0) {
    throw std::runtime_error("AnalyticFillEngine: bandwidth must be > 0");
  }
}

uint64_t AnalyticFillEngine::transfer_ps(uint64_t size_bytes) const {
  return static_cast<uint64_t>(static_cast<double>(size_bytes) / m_bytes_per_ps + 0.5);
}

uint64_t AnalyticFillEngine::start_fill(uint64_t /*hbf_addr*/,
                                        uint64_t size_bytes, uint64_t now_ps) {
  // tR is paid once per chunk; the transfer then streams at full bandwidth
  // because plane-level parallelism keeps the pipeline full (see
  // docs/hbf_model.md, "why bandwidth survives a 20 us latency").
  return now_ps + m_read_latency_ps + transfer_ps(size_bytes);
}

uint64_t AnalyticFillEngine::demand_latency_ps(uint64_t size_bytes,
                                               uint64_t /*now_ps*/) {
  return m_read_latency_ps + transfer_ps(size_bytes);
}

// ---------------------------------------------------------------------------
// Minimal YAML reader (same flat subset as h3_address_router.cpp)
// ---------------------------------------------------------------------------
namespace {

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string strip_comment(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '#') return s.substr(0, i);
  }
  return s;
}

size_t indent_of(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') ++i;
  return i;
}

std::map<std::string, std::string> read_flat_yaml(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("LhbConfig: cannot open config file: " + path);

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
      throw std::runtime_error("LhbConfig: " + path + ":" + std::to_string(lineno) +
                               ": expected 'key: value', got: " + body);
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

double parse_double(const std::string& key, const std::string& raw) {
  std::string s;
  for (char c : raw) if (c != '_') s.push_back(c);
  try {
    return std::stod(trim(s));
  } catch (const std::exception&) {
    throw std::runtime_error("LhbConfig: cannot parse number for " + key + ": '" + raw + "'");
  }
}

uint64_t parse_u64(const std::string& key, const std::string& raw) {
  const double d = parse_double(key, raw);
  if (d < 0) throw std::runtime_error("LhbConfig: negative value for " + key);
  return static_cast<uint64_t>(d);
}

bool parse_bool(const std::string& key, const std::string& raw) {
  std::string s = trim(raw);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
  if (s == "false" || s == "0" || s == "no" || s == "off") return false;
  throw std::runtime_error("LhbConfig: expected boolean for " + key + ", got '" + raw + "'");
}

}  // namespace

// ---------------------------------------------------------------------------
// LhbConfig
// ---------------------------------------------------------------------------

LhbConfig LhbConfig::from_yaml(const std::string& path) {
  const auto kv = read_flat_yaml(path);
  LhbConfig c;

  static const char* known[] = {
      "lhb.enabled",
      "lhb.buffer_size_mb",
      "lhb.num_buffers",
      "lhb.sram_access_latency_ns",
      "lhb.max_outstanding_fills",
      "prefetch.prefetch_hint_lead_time_ns",
      "hbf.hbf_read_latency_ns",
      "hbf.hbf_bandwidth_gbps",
  };
  for (const auto& e : kv) {
    bool ok = false;
    for (const char* k : known) {
      if (e.first == k) { ok = true; break; }
    }
    if (!ok) throw std::runtime_error("LhbConfig: unknown key '" + e.first + "' in " + path);
  }

  auto has = [&](const char* k) { return kv.find(k) != kv.end(); };
  auto get = [&](const char* k) { return kv.at(k); };

  if (has("lhb.enabled")) c.enabled = parse_bool("enabled", get("lhb.enabled"));
  if (has("lhb.buffer_size_mb"))
    c.buffer_size_bytes = parse_u64("buffer_size_mb", get("lhb.buffer_size_mb")) << 20;
  if (has("lhb.num_buffers"))
    c.num_buffers = static_cast<uint32_t>(parse_u64("num_buffers", get("lhb.num_buffers")));
  if (has("lhb.sram_access_latency_ns"))
    c.sram_access_latency_ns = parse_double("sram_access_latency_ns", get("lhb.sram_access_latency_ns"));
  if (has("lhb.max_outstanding_fills"))
    c.max_outstanding_fills =
        static_cast<uint32_t>(parse_u64("max_outstanding_fills", get("lhb.max_outstanding_fills")));
  if (has("prefetch.prefetch_hint_lead_time_ns"))
    c.prefetch_hint_lead_time_ns =
        parse_double("prefetch_hint_lead_time_ns", get("prefetch.prefetch_hint_lead_time_ns"));
  if (has("hbf.hbf_read_latency_ns"))
    c.hbf_read_latency_ns = parse_double("hbf_read_latency_ns", get("hbf.hbf_read_latency_ns"));
  if (has("hbf.hbf_bandwidth_gbps"))
    c.hbf_bandwidth_gbps = parse_double("hbf_bandwidth_gbps", get("hbf.hbf_bandwidth_gbps"));

  c.validate();
  return c;
}

void LhbConfig::validate() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("LhbConfig: " + m); };
  if (num_buffers < 1) fail("num_buffers must be >= 1 (2 for double buffering)");
  if (buffer_size_bytes == 0) fail("buffer_size_mb must be > 0");
  if (buffer_size_bytes % num_buffers != 0)
    fail("buffer_size must divide evenly by num_buffers");
  if (sram_access_latency_ns < 0.0) fail("sram_access_latency_ns must be >= 0");
  if (hbf_read_latency_ns < 0.0) fail("hbf_read_latency_ns must be >= 0");
  if (hbf_bandwidth_gbps <= 0.0) fail("hbf_bandwidth_gbps must be > 0");
}

uint64_t LhbConfig::required_capacity_bytes() const {
  // Eq. (1): Capacity = 2 x BW x Latency.
  // GB/s x ns == bytes.
  const double bytes = 2.0 * hbf_bandwidth_gbps * hbf_read_latency_ns;
  return static_cast<uint64_t>(bytes + 0.5);
}

// ---------------------------------------------------------------------------
// LhbStats
// ---------------------------------------------------------------------------

void LhbStats::reset() { *this = LhbStats(); }

double LhbStats::hit_rate() const {
  const uint64_t n = hits + misses_late + misses_absent;
  return n ? static_cast<double>(hits) / static_cast<double>(n) : 0.0;
}

double LhbStats::miss_rate() const {
  const uint64_t n = hits + misses_late + misses_absent;
  return n ? static_cast<double>(misses_late + misses_absent) / static_cast<double>(n) : 0.0;
}

double LhbStats::avg_stall_ps_per_miss() const {
  const uint64_t m = misses_late + misses_absent;
  return m ? static_cast<double>(total_stall_ps) / static_cast<double>(m) : 0.0;
}

double LhbStats::avg_stall_ns_per_miss() const { return avg_stall_ps_per_miss() / 1000.0; }

double LhbStats::avg_stall_cycles_per_miss(double gpu_clock_ghz) const {
  // cycles = ns * GHz
  return avg_stall_ns_per_miss() * gpu_clock_ghz;
}

double LhbStats::elapsed_ns() const {
  if (!has_window || last_time_ps <= first_time_ps) return 0.0;
  return static_cast<double>(last_time_ps - first_time_ps) / 1000.0;
}

double LhbStats::avg_occupancy_bytes() const {
  if (!has_window || last_time_ps <= first_time_ps) return 0.0;
  return static_cast<double>(occupancy_byte_ps) /
         static_cast<double>(last_time_ps - first_time_ps);
}

double LhbStats::avg_utilization(uint64_t capacity_bytes) const {
  return capacity_bytes ? avg_occupancy_bytes() / static_cast<double>(capacity_bytes) : 0.0;
}

double LhbStats::prefetch_efficiency() const {
  return bytes_prefetched ? static_cast<double>(bytes_served) /
                                static_cast<double>(bytes_prefetched)
                          : 0.0;
}

// ---------------------------------------------------------------------------
// LatencyHidingBuffer
// ---------------------------------------------------------------------------

LatencyHidingBuffer::LatencyHidingBuffer(const LhbConfig& config,
                                         ILhbFillEngine* fill_engine)
    : m_config(config), m_engine(fill_engine) {
  m_config.validate();
  if (m_config.max_outstanding_fills == 0) {
    m_config.max_outstanding_fills = m_config.num_buffers;
  }
  m_halves.resize(m_config.num_buffers);
  if (m_engine && m_engine->is_async()) {
    m_engine->set_completion_handler(
        [this](uint64_t token, uint64_t now) { this->on_fill_complete(token, now); });
  }
  if (m_config.enabled && m_engine == nullptr) {
    throw std::runtime_error("LatencyHidingBuffer: enabled but no fill engine supplied");
  }
}

void LatencyHidingBuffer::issue_hint(const PrefetchHint& hint) {
  ++m_stats.hints_received;
  if (!m_config.enabled) return;   // bypass mode ignores hints entirely
  if (hint.size_bytes == 0) return;

  StreamCursor c;
  c.tensor_id = hint.tensor_id;
  c.next_addr = hint.hbf_addr;
  c.remaining = hint.size_bytes;
  c.needed_at_ps = hint.needed_at_ps;
  c.layer = hint.layer;
  m_stream_queue.push_back(c);
}

uint32_t LatencyHidingBuffer::outstanding_fills() const {
  uint32_t n = 0;
  for (const auto& h : m_halves) {
    if (h.state == LhbState::PrefetchIssued || h.state == LhbState::Filling) ++n;
  }
  return n;
}

int LatencyHidingBuffer::find_idle_half() const {
  for (size_t i = 0; i < m_halves.size(); ++i) {
    if (m_halves[i].state == LhbState::Idle) return static_cast<int>(i);
  }
  return -1;
}

int LatencyHidingBuffer::find_half(uint64_t addr, uint64_t size) const {
  for (size_t i = 0; i < m_halves.size(); ++i) {
    const auto& h = m_halves[i];
    if (h.state == LhbState::Idle) continue;
    if (h.covers(addr, size)) return static_cast<int>(i);
  }
  return -1;
}

uint64_t LatencyHidingBuffer::occupancy_bytes() const {
  uint64_t n = 0;
  for (const auto& h : m_halves) {
    // A half occupies SRAM from the moment its fill is issued: the space is
    // reserved, not merely used once the data lands.
    if (h.state != LhbState::Idle) n += h.stored_bytes;
  }
  return n;
}

void LatencyHidingBuffer::accumulate_occupancy(uint64_t now_ps) {
  if (!m_occupancy_started) {
    m_occupancy_started = true;
    m_last_occupancy_update_ps = now_ps;
    m_stats.first_time_ps = now_ps;
    m_stats.last_time_ps = now_ps;
    m_stats.has_window = true;
    return;
  }
  if (now_ps > m_last_occupancy_update_ps) {
    m_stats.occupancy_byte_ps +=
        occupancy_bytes() * (now_ps - m_last_occupancy_update_ps);
    m_last_occupancy_update_ps = now_ps;
  }
  m_stats.last_time_ps = std::max(m_stats.last_time_ps, now_ps);
}

void LatencyHidingBuffer::on_fill_complete(uint64_t token, uint64_t now_ps) {
  for (size_t i = 0; i < m_halves.size(); ++i) {
    Half& h = m_halves[i];
    if (!h.async_fill_pending || h.fill_token != token) continue;
    h.async_fill_pending = false;
    h.fill_done_ps = now_ps;
    h.state = LhbState::Ready;
    ++m_stats.fills_completed;
    // Let the backend release any requests parked on this half.
    if (m_fill_complete_handler) m_fill_complete_handler(static_cast<int>(i), now_ps);
    return;
  }
}

void LatencyHidingBuffer::record_deferred_stall(uint64_t stall_ps) {
  m_stats.total_stall_ps += stall_ps;
  m_stats.max_stall_ps = std::max(m_stats.max_stall_ps, stall_ps);
}

void LatencyHidingBuffer::retire_fills(uint64_t now_ps) {
  for (auto& h : m_halves) {
    // Asynchronous fills complete only via on_fill_complete().
    if (h.async_fill_pending) {
      if (h.state == LhbState::PrefetchIssued && now_ps > h.fill_start_ps) {
        h.state = LhbState::Filling;
      }
      continue;
    }
    if ((h.state == LhbState::PrefetchIssued || h.state == LhbState::Filling) &&
        now_ps >= h.fill_done_ps) {
      h.state = LhbState::Ready;
      ++m_stats.fills_completed;
    } else if (h.state == LhbState::PrefetchIssued && now_ps > h.fill_start_ps) {
      // Once time has advanced past issue, the transfer is in flight.
      h.state = LhbState::Filling;
    }
  }
}

void LatencyHidingBuffer::launch_fills(uint64_t now_ps) {
  while (!m_stream_queue.empty() &&
         outstanding_fills() < m_config.max_outstanding_fills) {
    const int idx = find_idle_half();
    if (idx < 0) break;

    StreamCursor& c = m_stream_queue.front();
    // `span` is the global address range this half covers; `stored` is the SRAM
    // it actually occupies, which is span/N under address interleaving. The
    // fill is timed on `stored`, because that is what this partition's share of
    // the memory system actually transfers.
    const uint64_t span =
        std::min<uint64_t>(c.remaining, m_config.chunk_span_bytes());
    const uint32_t f = m_config.address_interleave_factor
                           ? m_config.address_interleave_factor : 1;
    const uint64_t stored = std::max<uint64_t>(1, span / f);

    Half& h = m_halves[static_cast<size_t>(idx)];
    h.state = LhbState::PrefetchIssued;
    h.tensor_id = c.tensor_id;
    h.base_addr = c.next_addr;
    h.length = span;
    h.stored_bytes = stored;
    h.consumed = 0;
    h.layer = c.layer;
    h.fill_start_ps = now_ps;
    if (m_engine->is_async()) {
      // The completion time is unknown until the engine calls back. Marking a
      // guess here would defeat the point of using a real memory model.
      h.async_fill_pending = true;
      h.fill_token = m_next_fill_token++;
      h.fill_done_ps = 0;
      if (!m_engine->start_fill_async(c.next_addr, stored, now_ps, h.fill_token)) {
        h = Half();          // engine refused; retry on a later tick
        break;
      }
    } else {
      h.async_fill_pending = false;
      h.fill_done_ps = m_engine->start_fill(c.next_addr, stored, now_ps);
    }

    ++m_stats.fills_started;
    m_stats.bytes_prefetched += stored;

    c.next_addr += span;
    c.remaining -= span;
    if (c.remaining == 0) m_stream_queue.pop_front();
  }
}

void LatencyHidingBuffer::release_half(Half& h, uint64_t now_ps) {
  // Waste is measured against SRAM actually held, not the global span.
  if (h.consumed < h.stored_bytes) {
    m_stats.bytes_wasted += (h.stored_bytes - h.consumed);
  }
  h = Half();
  ++m_stats.swaps;
  (void)now_ps;
}

void LatencyHidingBuffer::tick(uint64_t now_ps) {
  if (!m_config.enabled) return;
  if (m_engine) m_engine->tick(now_ps);
  accumulate_occupancy(now_ps);
  retire_fills(now_ps);
  launch_fills(now_ps);
}

LhbAccess LatencyHidingBuffer::access(uint64_t addr, uint64_t size_bytes,
                                      uint64_t now_ps) {
  LhbAccess res;
  ++m_stats.accesses;

  // ---- Bypass mode: baseline with no LHB at all -------------------------
  if (!m_config.enabled) {
    ++m_stats.bypasses;
    const uint64_t lat = m_engine ? m_engine->demand_latency_ps(size_bytes, now_ps)
                                  : m_config.hbf_read_latency_ps();
    res.outcome = LhbOutcome::Bypass;
    res.stall_ps = lat;
    res.ready_time_ps = now_ps + lat;
    m_stats.total_stall_ps += lat;
    m_stats.max_stall_ps = std::max(m_stats.max_stall_ps, lat);
    m_stats.bytes_served += size_bytes;
    return res;
  }

  // Advance the machine to `now` before answering.
  tick(now_ps);

  const int idx = find_half(addr, size_bytes);

  // ---- Not resident: full demand fetch ----------------------------------
  if (idx < 0) {
    ++m_stats.misses_absent;
    const uint64_t lat = m_engine->demand_latency_ps(size_bytes, now_ps);
    res.outcome = LhbOutcome::MissAbsent;
    res.stall_ps = lat;
    res.ready_time_ps = now_ps + lat;
    m_stats.total_stall_ps += lat;
    m_stats.max_stall_ps = std::max(m_stats.max_stall_ps, lat);
    m_stats.bytes_served += size_bytes;
    return res;
  }

  Half& h = m_halves[static_cast<size_t>(idx)];
  res.half = idx;
  const uint64_t sram = m_config.sram_access_latency_ps();

  // ---- Resident but the fill has not landed yet: partial hide -----------
  if (h.state == LhbState::PrefetchIssued || h.state == LhbState::Filling) {
    ++m_stats.misses_late;
    if (h.async_fill_pending) {
      // Coalesce onto the outstanding fill (MSHR merging). The stall length is
      // unknown until the fill lands; the backend parks the request and calls
      // record_deferred_stall() once it does. Issuing a fresh fetch here would
      // double-count HBF traffic and misreport latency.
      res.outcome = LhbOutcome::MissLate;
      res.waiting_on_fill = true;
      res.half = idx;
      res.stall_ps = 0;
      res.ready_time_ps = 0;
      m_stats.bytes_served += size_bytes;
      // CRITICAL: do NOT advance the state here.
      //
      // Setting Serving while the fill is still in flight makes the NEXT
      // access to this half miss the "still filling" test above, fall through
      // to the hit path, and be counted as a HIT on data that has not arrived.
      // That produced 4273 fake hits against 0 completed fills, and an average
      // latency of 46 ns in a system where a miss costs 20 us.
      //
      // The half becomes Ready only in on_fill_complete(). Consumption is
      // tracked now, so the half still drains and swaps correctly once it is.
      h.consumed = std::min<uint64_t>(h.stored_bytes, h.consumed + size_bytes);
      return res;
    }
    const uint64_t stall = h.fill_done_ps > now_ps ? (h.fill_done_ps - now_ps) : 0;
    res.outcome = LhbOutcome::MissLate;
    res.stall_ps = stall;
    res.ready_time_ps = now_ps + stall + sram;
    m_stats.total_stall_ps += stall;
    m_stats.max_stall_ps = std::max(m_stats.max_stall_ps, stall);
  } else {
    // ---- Hit: the 20 us was fully hidden -------------------------------
    ++m_stats.hits;
    res.outcome = LhbOutcome::Hit;
    res.stall_ps = 0;
    res.ready_time_ps = now_ps + sram;
  }

  h.state = LhbState::Serving;
  h.consumed = std::min<uint64_t>(h.stored_bytes, h.consumed + size_bytes);
  m_stats.bytes_served += size_bytes;

  // Fully drained -> swap. The test is against stored_bytes (this partition's
  // share), not the global span the half covers.
  if (h.consumed >= h.stored_bytes) {
    release_half(h, now_ps);
    launch_fills(now_ps);
  }
  return res;
}

LhbState LatencyHidingBuffer::half_state(int half) const {
  if (half < 0 || static_cast<size_t>(half) >= m_halves.size()) return LhbState::Idle;
  return m_halves[static_cast<size_t>(half)].state;
}

bool LatencyHidingBuffer::is_resident(uint64_t addr, uint64_t size_bytes) const {
  const int i = find_half(addr, size_bytes);
  if (i < 0) return false;
  const auto& h = m_halves[static_cast<size_t>(i)];
  return h.state == LhbState::Ready || h.state == LhbState::Serving;
}

void LatencyHidingBuffer::print_state(std::ostream& os) const {
  os << "LHB state (" << (m_config.enabled ? "enabled" : "BYPASS") << ", "
     << m_config.buffer_size_bytes / (1024 * 1024) << " MB in "
     << m_config.num_buffers << " halves of "
     << m_config.half_size_bytes() / (1024 * 1024) << " MB)\n";
  for (size_t i = 0; i < m_halves.size(); ++i) {
    const auto& h = m_halves[i];
    os << "  half[" << i << "] " << std::setw(16) << std::left << to_string(h.state)
       << std::right;
    if (h.state != LhbState::Idle) {
      os << " tensor=" << h.tensor_id << " layer=" << h.layer << " addr=0x"
         << std::hex << h.base_addr << std::dec << " len=" << h.length
         << " consumed=" << h.consumed << " done@" << h.fill_done_ps << "ps";
    }
    os << "\n";
  }
  os << "  queued tensor streams: " << m_stream_queue.size() << "\n";
}

void LatencyHidingBuffer::print_stats(std::ostream& os, double gpu_clock_ghz) const {
  const auto& s = m_stats;
  os << std::dec << "LHB statistics\n";
  os << "  lhb_enabled                       = " << (m_config.enabled ? 1 : 0) << "\n";
  os << "  lhb_capacity_bytes                = " << m_config.buffer_size_bytes << "\n";
  os << "  lhb_address_interleave_factor     = " << m_config.address_interleave_factor << "\n";
  os << "  lhb_chunk_span_bytes              = " << m_config.chunk_span_bytes() << "\n";
  os << "  lhb_required_capacity_eq1_bytes   = " << m_config.required_capacity_bytes() << "\n";
  os << "  lhb_accesses                      = " << s.accesses << "\n";
  os << "  lhb_hits                          = " << s.hits << "\n";
  os << "  lhb_misses_late                   = " << s.misses_late << "\n";
  os << "  lhb_misses_absent                 = " << s.misses_absent << "\n";
  os << "  lhb_bypasses                      = " << s.bypasses << "\n";
  os << "  lhb_hit_rate                      = " << std::fixed << std::setprecision(4)
     << s.hit_rate() << "\n";
  os << "  lhb_miss_rate                     = " << std::fixed << std::setprecision(4)
     << s.miss_rate() << "\n";
  os << "  lhb_total_stall_ns                = " << std::fixed << std::setprecision(1)
     << static_cast<double>(s.total_stall_ps) / 1000.0 << "\n";
  os << "  lhb_avg_stall_ns_per_miss         = " << std::fixed << std::setprecision(1)
     << s.avg_stall_ns_per_miss() << "\n";
  os << "  lhb_avg_stall_cycles_per_miss     = " << std::fixed << std::setprecision(1)
     << s.avg_stall_cycles_per_miss(gpu_clock_ghz) << "\n";
  os << "  lhb_max_stall_ns                  = " << std::fixed << std::setprecision(1)
     << static_cast<double>(s.max_stall_ps) / 1000.0 << "\n";
  os << "  lhb_bytes_served                  = " << s.bytes_served << "\n";
  os << "  lhb_bytes_prefetched              = " << s.bytes_prefetched << "\n";
  os << "  lhb_bytes_wasted                  = " << s.bytes_wasted << "\n";
  os << "  lhb_prefetch_efficiency           = " << std::fixed << std::setprecision(4)
     << s.prefetch_efficiency() << "\n";
  os << "  lhb_fills_started                 = " << s.fills_started << "\n";
  os << "  lhb_fills_completed               = " << s.fills_completed << "\n";
  os << "  lhb_swaps                         = " << s.swaps << "\n";
  os << "  lhb_hints_received                = " << s.hints_received << "\n";
  os << "  lhb_avg_occupancy_bytes           = " << std::fixed << std::setprecision(0)
     << s.avg_occupancy_bytes() << "\n";
  os << "  lhb_avg_utilization               = " << std::fixed << std::setprecision(4)
     << s.avg_utilization(m_config.buffer_size_bytes) << "\n";
  os << "  lhb_elapsed_ns                    = " << std::fixed << std::setprecision(1)
     << s.elapsed_ns() << "\n";
}

}  // namespace h3
