// ============================================================================
//  h3_address_router.cpp -- implementation of the H3 address decoder & router
// ============================================================================
//  See h3_address_router.h for the architectural description. This file holds:
//    * a small dependency-free YAML reader for configs/h3_router_config.yaml
//    * region decode and read-only enforcement
//    * D2D latency accounting and statistics
//
//  Deliberately depends only on the C++ standard library so the router can be
//  unit-tested with a bare `g++ -std=c++17` and no simulator build.
// ============================================================================

#include "h3_address_router.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace h3 {

// ---------------------------------------------------------------------------
// Enum helpers
// ---------------------------------------------------------------------------

const char* to_string(MemRegion r) {
  switch (r) {
    case MemRegion::Hbm: return "HBM";
    case MemRegion::Hbf: return "HBF";
    default: return "UNMAPPED";
  }
}

const char* to_string(ReqType t) {
  return t == ReqType::Read ? "READ" : "WRITE";
}

const char* to_string(RouteStatus s) {
  switch (s) {
    case RouteStatus::RoutedHbm: return "ROUTED_HBM";
    case RouteStatus::RoutedHbf: return "ROUTED_HBF";
    case RouteStatus::RejectedWriteToHbf: return "REJECTED_WRITE_TO_HBF";
    case RouteStatus::RejectedUnmapped: return "REJECTED_UNMAPPED";
    case RouteStatus::RejectedBackendFull: return "REJECTED_BACKEND_FULL";
    default: return "UNKNOWN";
  }
}

bool is_accepted(RouteStatus s) {
  return s == RouteStatus::RoutedHbm || s == RouteStatus::RoutedHbf;
}

// ---------------------------------------------------------------------------
// Minimal YAML reader
// ---------------------------------------------------------------------------
// Supports exactly the subset used by our config files:
//   * `key: value` pairs, optionally nested one level under `section:`
//   * `#` comments (whole-line or trailing)
//   * numeric literals in decimal or 0x hex, with `_` digit separators
//   * size suffixes: KB / MB / GB / TB (binary, i.e. 1 GB = 1<<30)
//   * booleans true/false, and bare strings
// Keys are flattened to "section.key". This avoids a yaml-cpp dependency in
// the standalone unit test; the full simulator links yaml-cpp anyway, and this
// reader can be swapped out without touching the router logic.
namespace {

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string strip_comment(const std::string& s) {
  bool in_quotes = false;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '"') in_quotes = !in_quotes;
    if (s[i] == '#' && !in_quotes) return s.substr(0, i);
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
  if (!in) {
    throw std::runtime_error("H3RouterConfig: cannot open config file: " + path);
  }

  std::map<std::string, std::string> out;
  std::string section;
  std::string line;
  size_t lineno = 0;

  while (std::getline(in, line)) {
    ++lineno;
    const std::string raw = strip_comment(line);
    const std::string body = trim(raw);
    if (body.empty()) continue;

    const size_t colon = body.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("H3RouterConfig: " + path + ":" +
                               std::to_string(lineno) +
                               ": expected 'key: value', got: " + body);
    }

    const std::string key = trim(body.substr(0, colon));
    const std::string value = trim(body.substr(colon + 1));

    if (value.empty()) {          // a section header
      section = (indent_of(raw) == 0) ? key : section + "." + key;
      continue;
    }
    out[section.empty() ? key : section + "." + key] = value;
  }
  return out;
}

// Parse an integer with optional 0x prefix, '_' separators and a binary size
// suffix (KB/MB/GB/TB).
uint64_t parse_u64(const std::string& key, const std::string& raw) {
  std::string s;
  for (char c : raw) {
    if (c != '_' && c != '\'') s.push_back(c);
  }
  s = trim(s);
  if (s.empty()) throw std::runtime_error("H3RouterConfig: empty value for " + key);

  // Split trailing alphabetic suffix.
  size_t end = s.size();
  while (end > 0 && std::isalpha(static_cast<unsigned char>(s[end - 1]))) --end;
  std::string num = trim(s.substr(0, end));
  std::string suffix = s.substr(end);
  for (auto& c : suffix) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

  // A bare hex literal ends in letters too; don't mistake 0x30 for a suffix.
  if (num.size() >= 2 && num[0] == '0' && (num[1] == 'x' || num[1] == 'X')) {
    num = s;      // take the whole token
    suffix.clear();
  }

  uint64_t mult = 1;
  if (suffix == "KB" || suffix == "K") mult = 1ULL << 10;
  else if (suffix == "MB" || suffix == "M") mult = 1ULL << 20;
  else if (suffix == "GB" || suffix == "G") mult = 1ULL << 30;
  else if (suffix == "TB" || suffix == "T") mult = 1ULL << 40;
  else if (!suffix.empty()) {
    throw std::runtime_error("H3RouterConfig: unknown size suffix '" + suffix +
                             "' for key " + key);
  }

  uint64_t base = 10;
  if (num.size() >= 2 && num[0] == '0' && (num[1] == 'x' || num[1] == 'X')) {
    base = 16;
    num = num.substr(2);
  }

  uint64_t v = 0;
  size_t consumed = 0;
  try {
    v = std::stoull(num, &consumed, static_cast<int>(base));
  } catch (const std::exception&) {
    throw std::runtime_error("H3RouterConfig: cannot parse number for " + key +
                             ": '" + raw + "'");
  }
  if (consumed != num.size()) {
    throw std::runtime_error("H3RouterConfig: trailing garbage in value for " +
                             key + ": '" + raw + "'");
  }
  return v * mult;
}

double parse_double(const std::string& key, const std::string& raw) {
  try {
    return std::stod(trim(raw));
  } catch (const std::exception&) {
    throw std::runtime_error("H3RouterConfig: cannot parse number for " + key +
                             ": '" + raw + "'");
  }
}

bool parse_bool(const std::string& key, const std::string& raw) {
  std::string s = trim(raw);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
  if (s == "false" || s == "0" || s == "no" || s == "off") return false;
  throw std::runtime_error("H3RouterConfig: expected boolean for " + key +
                           ", got '" + raw + "'");
}

HbfWritePolicy parse_policy(const std::string& key, const std::string& raw) {
  std::string s = trim(raw);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "reject") return HbfWritePolicy::Reject;
  if (s == "redirect_hbm" || s == "redirect") return HbfWritePolicy::RedirectHbm;
  if (s == "allow") return HbfWritePolicy::Allow;
  throw std::runtime_error("H3RouterConfig: unknown " + key + " '" + raw +
                           "' (expected reject | redirect_hbm | allow)");
}

}  // namespace

// ---------------------------------------------------------------------------
// H3RouterConfig
// ---------------------------------------------------------------------------

H3RouterConfig H3RouterConfig::from_yaml(const std::string& path) {
  const auto kv = read_flat_yaml(path);
  H3RouterConfig c;

  // Every key is optional; anything absent keeps its documented default.
  // Unknown keys are an error, so a typo never silently does nothing.
  static const char* known[] = {
      "address_map.hbm_base_addr",       "address_map.hbm_size_bytes",
      "address_map.hbf_base_addr",       "address_map.hbf_size_bytes",
      "d2d.hop_latency_ns",              "d2d.hops_to_hbf",
      "address_map.allocation_map",       "address_map.unmapped_to_hbm",
      "read_only.enforce_hbf_read_only", "read_only.hbf_write_policy",
      "read_only.warn_on_hbf_write",     "read_only.max_warnings",
      "bandwidth.hbm_peak_gbps",         "bandwidth.hbf_peak_gbps",
  };
  for (const auto& e : kv) {
    bool ok = false;
    for (const char* k : known) {
      if (e.first == k) { ok = true; break; }
    }
    if (!ok) {
      throw std::runtime_error("H3RouterConfig: unknown key '" + e.first +
                               "' in " + path);
    }
  }

  auto has = [&](const char* k) { return kv.find(k) != kv.end(); };
  auto get = [&](const char* k) { return kv.at(k); };

  if (has("address_map.hbm_base_addr"))
    c.hbm_base_addr = parse_u64("hbm_base_addr", get("address_map.hbm_base_addr"));
  if (has("address_map.hbm_size_bytes"))
    c.hbm_size_bytes = parse_u64("hbm_size_bytes", get("address_map.hbm_size_bytes"));
  if (has("address_map.hbf_base_addr"))
    c.hbf_base_addr = parse_u64("hbf_base_addr", get("address_map.hbf_base_addr"));
  if (has("address_map.hbf_size_bytes"))
    c.hbf_size_bytes = parse_u64("hbf_size_bytes", get("address_map.hbf_size_bytes"));

  if (has("address_map.allocation_map"))
    c.allocation_map = get("address_map.allocation_map");
  if (has("address_map.unmapped_to_hbm"))
    c.unmapped_to_hbm = parse_bool("unmapped_to_hbm", get("address_map.unmapped_to_hbm"));

  if (has("d2d.hop_latency_ns"))
    c.d2d_hop_latency_ns = parse_double("hop_latency_ns", get("d2d.hop_latency_ns"));
  if (has("d2d.hops_to_hbf"))
    c.d2d_hops_to_hbf = static_cast<uint32_t>(parse_u64("hops_to_hbf", get("d2d.hops_to_hbf")));

  if (has("read_only.enforce_hbf_read_only"))
    c.enforce_hbf_read_only = parse_bool("enforce_hbf_read_only", get("read_only.enforce_hbf_read_only"));
  if (has("read_only.hbf_write_policy"))
    c.hbf_write_policy = parse_policy("hbf_write_policy", get("read_only.hbf_write_policy"));
  if (has("read_only.warn_on_hbf_write"))
    c.warn_on_hbf_write = parse_bool("warn_on_hbf_write", get("read_only.warn_on_hbf_write"));
  if (has("read_only.max_warnings"))
    c.max_warnings = parse_u64("max_warnings", get("read_only.max_warnings"));

  if (has("bandwidth.hbm_peak_gbps"))
    c.hbm_peak_bandwidth_gbps = parse_double("hbm_peak_gbps", get("bandwidth.hbm_peak_gbps"));
  if (has("bandwidth.hbf_peak_gbps"))
    c.hbf_peak_bandwidth_gbps = parse_double("hbf_peak_gbps", get("bandwidth.hbf_peak_gbps"));

  c.validate();
  return c;
}

void H3RouterConfig::validate() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("H3RouterConfig: " + m); };

  if (hbm_size_bytes == 0) fail("hbm_size_bytes must be non-zero");
  if (hbf_size_bytes == 0) fail("hbf_size_bytes must be non-zero");

  // Overflow / wraparound.
  const uint64_t kMax = std::numeric_limits<uint64_t>::max();
  if (hbm_base_addr > kMax - hbm_size_bytes) fail("HBM region wraps past 2^64");
  if (hbf_base_addr > kMax - hbf_size_bytes) fail("HBF region wraps past 2^64");

  // Overlap: the whole point of the router is an unambiguous decode.
  const uint64_t hbm_end = hbm_end_addr();
  const uint64_t hbf_end = hbf_end_addr();
  const bool overlap = (hbm_base_addr < hbf_end) && (hbf_base_addr < hbm_end);
  if (overlap) {
    std::ostringstream os;
    os << "HBM [0x" << std::hex << hbm_base_addr << ", 0x" << hbm_end
       << ") overlaps HBF [0x" << hbf_base_addr << ", 0x" << hbf_end << ")";
    fail(os.str());
  }

  if (d2d_hop_latency_ns < 0.0) fail("d2d.hop_latency_ns must be >= 0");
  if (hbm_peak_bandwidth_gbps <= 0.0) fail("bandwidth.hbm_peak_gbps must be > 0");
  if (hbf_peak_bandwidth_gbps <= 0.0) fail("bandwidth.hbf_peak_gbps must be > 0");
}

// ---------------------------------------------------------------------------
// H3RouterStats
// ---------------------------------------------------------------------------

void H3RouterStats::reset() { *this = H3RouterStats(); }

double H3RouterStats::elapsed_ns() const {
  if (!has_window || last_time_ps <= first_time_ps) return 0.0;
  return static_cast<double>(last_time_ps - first_time_ps) / 1000.0;
}

static double bw_gbps(uint64_t bytes, double ns) {
  if (ns <= 0.0) return 0.0;
  // bytes / ns == GB/s (1e9 bytes/s), which is the unit the paper quotes.
  return static_cast<double>(bytes) / ns;
}

double H3RouterStats::hbm_bandwidth_gbps() const { return bw_gbps(bytes_hbm, elapsed_ns()); }
double H3RouterStats::hbf_bandwidth_gbps() const { return bw_gbps(bytes_hbf, elapsed_ns()); }

double H3RouterStats::hbm_utilization(double peak_gbps) const {
  return peak_gbps > 0.0 ? hbm_bandwidth_gbps() / peak_gbps : 0.0;
}
double H3RouterStats::hbf_utilization(double peak_gbps) const {
  return peak_gbps > 0.0 ? hbf_bandwidth_gbps() / peak_gbps : 0.0;
}

double H3RouterStats::hbf_request_fraction() const {
  const uint64_t routed = requests_hbm + requests_hbf;
  return routed ? static_cast<double>(requests_hbf) / static_cast<double>(routed) : 0.0;
}

// ---------------------------------------------------------------------------
// H3AddressRouter
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Allocation map
// ---------------------------------------------------------------------------
// Reads the file written by tools/classify_trace.py. Deliberately a small
// hand-rolled reader for one fixed shape rather than a YAML dependency:
//
//   allocations:
//     - orig_addr: 0x7f35ab700000
//       size: 278596
//       region: hbf
//       h3_addr: 0x3000000000
//
void H3AddressRouter::load_allocation_map(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("H3AddressRouter: cannot open allocation map: " + path +
                             "\n  Generate it with tools/classify_trace.py");
  }
  Alloc cur;
  bool have = false;
  std::string line;
  auto flush = [&]() {
    if (have && cur.size) m_allocs.push_back(cur);
    cur = Alloc();
    have = false;
  };
  while (std::getline(in, line)) {
    const std::string body = trim(strip_comment(line));
    if (body.empty() || body == "allocations:") continue;
    std::string kv = body;
    if (kv.rfind("- ", 0) == 0) {           // start of a new entry
      flush();
      have = true;
      kv = trim(kv.substr(2));
    }
    const size_t colon = kv.find(':');
    if (colon == std::string::npos) continue;
    const std::string k = trim(kv.substr(0, colon));
    const std::string v = trim(kv.substr(colon + 1));
    if (k == "orig_addr") { cur.orig_addr = parse_u64(k, v); have = true; }
    else if (k == "size") { cur.size = parse_u64(k, v); }
    else if (k == "region") {
      cur.region = (v == "hbf" || v == "HBF") ? MemRegion::Hbf : MemRegion::Hbm;
    } else if (k == "h3_addr") {
      const uint64_t h3 = parse_u64(k, v);
      const uint64_t base = (cur.region == MemRegion::Hbf) ? m_config.hbf_base_addr
                                                           : m_config.hbm_base_addr;
      cur.local_base = h3 >= base ? h3 - base : 0;
    }
  }
  flush();

  std::sort(m_allocs.begin(), m_allocs.end(),
            [](const Alloc& a, const Alloc& b) { return a.orig_addr < b.orig_addr; });

  // Overlapping entries would make the decode ambiguous.
  for (size_t i = 1; i < m_allocs.size(); ++i) {
    if (m_allocs[i].orig_addr < m_allocs[i - 1].orig_addr + m_allocs[i - 1].size) {
      std::ostringstream os;
      os << "H3AddressRouter: overlapping allocations in " << path << " at 0x"
         << std::hex << m_allocs[i].orig_addr;
      throw std::runtime_error(os.str());
    }
  }
  if (m_allocs.empty()) {
    throw std::runtime_error("H3AddressRouter: allocation map " + path +
                             " contains no entries");
  }
}

const H3AddressRouter::Alloc* H3AddressRouter::find_alloc(uint64_t addr) const {
  // Binary search: last entry whose orig_addr <= addr.
  size_t lo = 0, hi = m_allocs.size();
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (m_allocs[mid].orig_addr <= addr) lo = mid + 1; else hi = mid;
  }
  if (lo == 0) return nullptr;
  const Alloc& a = m_allocs[lo - 1];
  return (addr < a.orig_addr + a.size) ? &a : nullptr;
}

H3AddressRouter::H3AddressRouter(const H3RouterConfig& config,
                                 IMemoryBackend* hbm_backend,
                                 IMemoryBackend* hbf_backend)
    : m_config(config), m_hbm(hbm_backend), m_hbf(hbf_backend) {
  m_config.validate();
  if (!m_config.allocation_map.empty()) {
    load_allocation_map(m_config.allocation_map);
  }
}

void H3AddressRouter::set_backends(IMemoryBackend* hbm, IMemoryBackend* hbf) {
  m_hbm = hbm;
  m_hbf = hbf;
}

MemRegion H3AddressRouter::decode(uint64_t addr) const {
  if (!m_allocs.empty()) {
    const Alloc* a = find_alloc(addr);
    if (a) return a->region;
    // Not in any known buffer: stack, local or constant memory. Small and
    // mutable, so HBM. Reported separately via unallocated_requests.
    return m_config.unmapped_to_hbm ? MemRegion::Hbm : MemRegion::Unmapped;
  }
  if (addr >= m_config.hbm_base_addr && addr < m_config.hbm_end_addr()) {
    return MemRegion::Hbm;
  }
  if (addr >= m_config.hbf_base_addr && addr < m_config.hbf_end_addr()) {
    return MemRegion::Hbf;
  }
  return MemRegion::Unmapped;
}

uint64_t H3AddressRouter::to_local_addr(uint64_t addr, MemRegion region) const {
  if (!m_allocs.empty()) {
    const Alloc* a = find_alloc(addr);
    if (a) return a->local_base + (addr - a->orig_addr);
    return 0;   // unallocated: mapped to the base of HBM
  }
  switch (region) {
    case MemRegion::Hbm: return addr - m_config.hbm_base_addr;
    case MemRegion::Hbf: return addr - m_config.hbf_base_addr;
    default: return addr;
  }
}

RouteDecision H3AddressRouter::inspect(const H3Request& req) const {
  RouteDecision d;
  d.region = decode(req.addr);
  d.issue_time_ps = req.time_ps;

  if (d.region == MemRegion::Unmapped) {
    d.status = RouteStatus::RejectedUnmapped;
    return d;
  }

  d.local_addr = to_local_addr(req.addr, d.region);

  if (d.region == MemRegion::Hbm) {
    d.status = RouteStatus::RoutedHbm;
    return d;
  }

  // --- HBF path ---
  const bool is_write = (req.type == ReqType::Write);
  if (is_write && m_config.enforce_hbf_read_only &&
      m_config.hbf_write_policy != HbfWritePolicy::Allow) {
    if (m_config.hbf_write_policy == HbfWritePolicy::RedirectHbm) {
      // Keep the simulation running, but the address is still an HBF address;
      // report it as an HBM route so the caller sends it to the HBM backend.
      d.status = RouteStatus::RoutedHbm;
      d.region = MemRegion::Hbm;
      d.local_addr = 0;   // caller must not trust this; it is a bug path
      return d;
    }
    d.status = RouteStatus::RejectedWriteToHbf;
    return d;
  }

  // Charge the die-to-die hop. This is physical routing overhead only; the
  // 20 us NAND array latency is applied inside the HBF Ramulator model.
  d.d2d_hops = m_config.d2d_hops_to_hbf;
  d.issue_time_ps = req.time_ps + d.d2d_hops * m_config.d2d_hop_latency_ps();
  d.status = RouteStatus::RoutedHbf;
  return d;
}

RouteDecision H3AddressRouter::route(const H3Request& req) {
  RouteDecision d = inspect(req);

  ++m_stats.requests_total;
  note_time(req.time_ps);
  if (!m_allocs.empty() && find_alloc(req.addr) == nullptr) {
    ++m_stats.unallocated_requests;
  }

  const bool is_write = (req.type == ReqType::Write);

  // --- Unmapped ---
  if (d.status == RouteStatus::RejectedUnmapped) {
    ++m_stats.unmapped_requests;
    std::ostringstream os;
    os << "unmapped address 0x" << std::hex << req.addr
       << " (outside both HBM and HBF regions)";
    warn(os.str());
    return d;
  }

  // --- Write to read-only HBF ---
  if (d.status == RouteStatus::RejectedWriteToHbf) {
    ++m_stats.write_attempts_to_hbf;
    std::ostringstream os;
    os << "WRITE to read-only HBF region at 0x" << std::hex << req.addr
       << " -- data placement bug (weights/shared KV must never be written)";
    warn(os.str());
    return d;
  }

  // A redirected write is an HBF-address write that policy sent to HBM.
  if (is_write && decode(req.addr) == MemRegion::Hbf &&
      m_config.hbf_write_policy == HbfWritePolicy::RedirectHbm) {
    ++m_stats.write_attempts_to_hbf;
    ++m_stats.writes_redirected_to_hbm;
    warn("WRITE to HBF region redirected to HBM (policy=redirect_hbm)");
  }

  IMemoryBackend* backend = (d.status == RouteStatus::RoutedHbm) ? m_hbm : m_hbf;

  // Backpressure: check before counting the request as delivered.
  if (backend != nullptr) {
    if (backend->is_full(is_write)) {
      ++m_stats.backend_full_rejects;
      d.status = RouteStatus::RejectedBackendFull;
      return d;
    }
    if (!backend->send(req, d.local_addr, d.issue_time_ps)) {
      ++m_stats.backend_full_rejects;
      d.status = RouteStatus::RejectedBackendFull;
      return d;
    }
  }

  // --- Accepted: account for it ---
  if (d.status == RouteStatus::RoutedHbm) {
    ++m_stats.requests_hbm;
    m_stats.bytes_hbm += req.size_bytes;
    if (is_write) ++m_stats.writes_hbm; else ++m_stats.reads_hbm;
  } else {
    ++m_stats.requests_hbf;
    m_stats.bytes_hbf += req.size_bytes;
    m_stats.d2d_hops += d.d2d_hops;
    if (is_write) ++m_stats.writes_hbf; else ++m_stats.reads_hbf;
  }
  return d;
}

void H3AddressRouter::note_time(uint64_t time_ps) {
  if (!m_stats.has_window) {
    m_stats.first_time_ps = time_ps;
    m_stats.last_time_ps = time_ps;
    m_stats.has_window = true;
    return;
  }
  m_stats.first_time_ps = std::min(m_stats.first_time_ps, time_ps);
  m_stats.last_time_ps = std::max(m_stats.last_time_ps, time_ps);
}

void H3AddressRouter::warn(const std::string& msg) {
  if (!m_config.warn_on_hbf_write) return;
  if (m_warnings_emitted >= m_config.max_warnings) return;
  ++m_warnings_emitted;
  std::cerr << "[H3Router] WARNING: " << msg << std::dec << "\n";
  if (m_warnings_emitted == m_config.max_warnings) {
    std::cerr << "[H3Router] (further warnings suppressed; see stats for totals)\n";
  }
}

void H3AddressRouter::print_address_map(std::ostream& os) const {
  auto gib = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); };
  os << "H3 address map\n"
     << "  HBM : [0x" << std::hex << m_config.hbm_base_addr << ", 0x"
     << m_config.hbm_end_addr() << ")  " << std::dec << std::fixed
     << std::setprecision(1) << gib(m_config.hbm_size_bytes) << " GB"
     << "   (generated KV cache + activations, read/write)\n"
     << "  HBF : [0x" << std::hex << m_config.hbf_base_addr << ", 0x"
     << m_config.hbf_end_addr() << ")  " << std::dec << std::fixed
     << std::setprecision(1) << gib(m_config.hbf_size_bytes) << " GB"
     << "   (weights + shared precomputed KV cache, read-only)\n"
     << "  D2D : " << m_config.d2d_hop_latency_ns << " ns x "
     << m_config.d2d_hops_to_hbf << " hop(s) on every HBF access\n";
}

void H3AddressRouter::print_stats(std::ostream& os) const {
  const auto& s = m_stats;
  os << std::dec;
  os << "H3 router statistics\n";
  os << "  h3_router_requests_total          = " << s.requests_total << "\n";
  os << "  h3_router_requests_hbm            = " << s.requests_hbm << "\n";
  os << "  h3_router_requests_hbf            = " << s.requests_hbf << "\n";
  os << "  h3_router_reads_hbm               = " << s.reads_hbm << "\n";
  os << "  h3_router_writes_hbm              = " << s.writes_hbm << "\n";
  os << "  h3_router_reads_hbf               = " << s.reads_hbf << "\n";
  os << "  h3_router_writes_hbf              = " << s.writes_hbf << "\n";
  os << "  h3_router_bytes_hbm               = " << s.bytes_hbm << "\n";
  os << "  h3_router_bytes_hbf               = " << s.bytes_hbf << "\n";
  os << "  h3_router_d2d_hops                = " << s.d2d_hops << "\n";
  os << "  h3_router_hbf_request_fraction    = " << std::fixed
     << std::setprecision(4) << s.hbf_request_fraction() << "\n";
  os << "  h3_router_elapsed_ns              = " << std::fixed
     << std::setprecision(1) << s.elapsed_ns() << "\n";
  os << "  h3_router_hbm_bandwidth_gbps      = " << std::fixed
     << std::setprecision(2) << s.hbm_bandwidth_gbps() << "\n";
  os << "  h3_router_hbf_bandwidth_gbps      = " << std::fixed
     << std::setprecision(2) << s.hbf_bandwidth_gbps() << "\n";
  os << "  h3_router_hbm_utilization         = " << std::fixed
     << std::setprecision(4) << s.hbm_utilization(m_config.hbm_peak_bandwidth_gbps) << "\n";
  os << "  h3_router_hbf_utilization         = " << std::fixed
     << std::setprecision(4) << s.hbf_utilization(m_config.hbf_peak_bandwidth_gbps) << "\n";
  // Correctness counters -- these must all be zero in a healthy run.
  os << "  h3_router_write_attempts_to_hbf   = " << s.write_attempts_to_hbf
     << (s.write_attempts_to_hbf ? "   <-- ERROR: data placement bug" : "") << "\n";
  os << "  h3_router_writes_redirected       = " << s.writes_redirected_to_hbm << "\n";
  os << "  h3_router_allocations_loaded      = " << m_allocs.size() << "\n";
  os << "  h3_router_unallocated_requests    = " << s.unallocated_requests << "\n";
  os << "  h3_router_unmapped_requests       = " << s.unmapped_requests
     << (s.unmapped_requests ? "   <-- ERROR: address outside H3 map" : "") << "\n";
  os << "  h3_router_backend_full_rejects    = " << s.backend_full_rejects << "\n";
}

}  // namespace h3
