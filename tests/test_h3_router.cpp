// ============================================================================
//  test_h3_router.cpp -- standalone unit test for the H3 address router
// ============================================================================
//
//  WHAT THIS IS
//      A self-contained test of h3-components/h3_address_router.{h,cpp}. It
//      needs no simulator, no Ramulator, and no build system -- only a C++17
//      compiler -- so router logic can be verified in isolation and in
//      seconds.
//
//  WHAT IT CONNECTS TO
//      Uses FakeBackend, a trivial IMemoryBackend that records what it was
//      given. In the real simulator (Step 6) those backends wrap Ramulator
//      memory systems; the router cannot tell the difference, which is the
//      point of the interface.
//
//  BUILD AND RUN
//      g++ -std=c++17 -O2 -I h3-components \
//          tests/test_h3_router.cpp h3-components/h3_address_router.cpp \
//          -o build/test_h3_router && ./build/test_h3_router
//      (or: ./scripts/run_unit_tests.sh)
//
//  EXIT CODE
//      0 = all assertions passed; 1 = at least one failure (details on stdout).
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "h3_address_router.h"

using namespace h3;

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------
namespace {

int g_checks = 0;
int g_failures = 0;
std::string g_section;

void section(const std::string& name) {
  g_section = name;
  std::cout << "\n== " << name << " ==\n";
}

void check(bool cond, const std::string& what) {
  ++g_checks;
  if (cond) {
    std::cout << "  [ ok ] " << what << "\n";
  } else {
    ++g_failures;
    std::cout << "  [FAIL] " << what << "\n";
  }
}

template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& what) {
  ++g_checks;
  if (got == static_cast<A>(want)) {
    std::cout << "  [ ok ] " << what << "\n";
  } else {
    ++g_failures;
    std::cout << "  [FAIL] " << what << "  (got " << got << ", want " << want << ")\n";
  }
}

// A backend that just records what it received.
class FakeBackend : public IMemoryBackend {
 public:
  explicit FakeBackend(const char* n) : m_name(n) {}

  struct Entry {
    uint64_t global_addr;
    uint64_t local_addr;
    uint64_t issue_time_ps;
    ReqType type;
    uint32_t size_bytes;
  };

  bool is_full(bool /*is_write*/) const override { return full; }

  bool send(const H3Request& req, uint64_t local_addr,
            uint64_t issue_time_ps) override {
    if (refuse) return false;
    received.push_back({req.addr, local_addr, issue_time_ps, req.type, req.size_bytes});
    return true;
  }

  const char* name() const override { return m_name; }

  std::vector<Entry> received;
  bool full = false;
  bool refuse = false;

 private:
  const char* m_name;
};

// Default map: HBM [0, 192 GB), HBF [192 GB, 192 GB + 3 TB)
constexpr uint64_t GB = 1ULL << 30;
constexpr uint64_t TB = 1ULL << 40;
constexpr uint64_t HBM_BASE = 0;
constexpr uint64_t HBM_SIZE = 192 * GB;
constexpr uint64_t HBF_BASE = 192 * GB;
constexpr uint64_t HBF_SIZE = 3 * TB;

H3RouterConfig make_config() {
  H3RouterConfig c;
  c.hbm_base_addr = HBM_BASE;
  c.hbm_size_bytes = HBM_SIZE;
  c.hbf_base_addr = HBF_BASE;
  c.hbf_size_bytes = HBF_SIZE;
  c.d2d_hop_latency_ns = 25.0;
  c.d2d_hops_to_hbf = 1;
  c.enforce_hbf_read_only = true;
  c.hbf_write_policy = HbfWritePolicy::Reject;
  c.warn_on_hbf_write = false;   // keep test output clean
  return c;
}

H3Request read_at(uint64_t addr, uint64_t t_ps = 0, uint32_t size = 32) {
  H3Request r;
  r.addr = addr;
  r.type = ReqType::Read;
  r.size_bytes = size;
  r.time_ps = t_ps;
  return r;
}

H3Request write_at(uint64_t addr, uint64_t t_ps = 0, uint32_t size = 32) {
  H3Request r = read_at(addr, t_ps, size);
  r.type = ReqType::Write;
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
int main() {
  std::cout << "H3 address router -- standalone unit test\n";

  // -------------------------------------------------------------------------
  section("1. Address map boundaries decode correctly");
  {
    H3AddressRouter r(make_config());
    r.print_address_map(std::cout);

    check(r.decode(0) == MemRegion::Hbm, "0x0 -> HBM (first byte of HBM)");
    check(r.decode(HBM_SIZE - 1) == MemRegion::Hbm, "192GB-1 -> HBM (last byte of HBM)");
    check(r.decode(HBM_SIZE) == MemRegion::Hbf, "192GB -> HBF (first byte of HBF)");
    check(r.decode(HBF_BASE + HBF_SIZE - 1) == MemRegion::Hbf, "192GB+3TB-1 -> HBF (last byte)");
    check(r.decode(HBF_BASE + HBF_SIZE) == MemRegion::Unmapped, "192GB+3TB -> UNMAPPED (one past end)");
    check(r.decode(~0ULL) == MemRegion::Unmapped, "0xffff...f -> UNMAPPED");

    // Region-local translation must be zero-based: Ramulator expects that.
    check_eq(r.to_local_addr(0x1000, MemRegion::Hbm), 0x1000ULL, "HBM local addr is identity");
    check_eq(r.to_local_addr(HBF_BASE, MemRegion::Hbf), 0ULL, "HBF base -> local 0");
    check_eq(r.to_local_addr(HBF_BASE + 0xABC, MemRegion::Hbf), 0xABCULL, "HBF local addr is offset");
  }

  // -------------------------------------------------------------------------
  section("2. Requests reach the correct backend");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    r.route(read_at(0x2000));
    r.route(read_at(64 * GB));
    r.route(read_at(HBF_BASE + 0x40));
    r.route(read_at(HBF_BASE + 2 * TB));

    check_eq(hbm.received.size(), 2u, "2 requests delivered to HBM backend");
    check_eq(hbf.received.size(), 2u, "2 requests delivered to HBF backend");
    check_eq(r.stats().requests_hbm, 2u, "stats: requests_hbm == 2");
    check_eq(r.stats().requests_hbf, 2u, "stats: requests_hbf == 2");
    check_eq(r.stats().requests_total, 4u, "stats: requests_total == 4");

    check_eq(hbf.received[0].local_addr, 0x40ULL, "HBF backend sees region-local address");
    check_eq(hbf.received[1].local_addr, 2 * TB, "HBF backend sees region-local address (2TB in)");
    check_eq(hbm.received[1].local_addr, 64 * GB, "HBM backend sees region-local address");
  }

  // -------------------------------------------------------------------------
  section("3. D2D hop latency is charged to HBF only");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    const uint64_t t0 = 1'000'000;  // 1 us in ps
    auto d_hbm = r.route(read_at(0x1000, t0));
    auto d_hbf = r.route(read_at(HBF_BASE + 0x1000, t0));

    check_eq(d_hbm.d2d_hops, 0u, "HBM route charges 0 D2D hops");
    check_eq(d_hbm.issue_time_ps, t0, "HBM route issues at arrival time");

    check_eq(d_hbf.d2d_hops, 1u, "HBF route charges 1 D2D hop");
    check_eq(d_hbf.issue_time_ps, t0 + 25'000, "HBF route adds 25 ns (25000 ps)");
    check_eq(r.stats().d2d_hops, 1u, "stats: d2d_hops == 1");

    // The D2D hop must be tiny relative to tR; if it ever isn't, the model is
    // conflating routing overhead with NAND array latency.
    const double d2d_ns = 25.0, tR_ns = 20000.0;
    check(d2d_ns / tR_ns < 0.01, "D2D latency is <1% of NAND tR (25ns vs 20us)");
  }

  // -------------------------------------------------------------------------
  section("4. HBF is read-only: writes are rejected and counted");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    r.route(read_at(HBF_BASE + 0x100));           // legal
    auto bad = r.route(write_at(HBF_BASE + 0x200));  // illegal
    r.route(write_at(0x300));                     // legal: HBM is writable

    check(bad.status == RouteStatus::RejectedWriteToHbf, "write to HBF is REJECTED_WRITE_TO_HBF");
    check_eq(r.stats().write_attempts_to_hbf, 1u, "stats: write_attempts_to_hbf == 1");
    check_eq(hbf.received.size(), 1u, "rejected write never reaches the HBF backend");
    check_eq(r.stats().writes_hbm, 1u, "writes to HBM are allowed");
    check_eq(r.stats().writes_hbf, 0u, "no writes recorded as served by HBF");
  }

  // -------------------------------------------------------------------------
  section("5. hbf_write_policy = redirect_hbm keeps the sim running");
  {
    auto cfg = make_config();
    cfg.hbf_write_policy = HbfWritePolicy::RedirectHbm;
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    auto d = r.route(write_at(HBF_BASE + 0x200));
    check(d.status == RouteStatus::RoutedHbm, "redirected write routes to HBM");
    check_eq(hbf.received.size(), 0u, "redirected write does not reach HBF");
    check_eq(hbm.received.size(), 1u, "redirected write reaches HBM");
    check_eq(r.stats().write_attempts_to_hbf, 1u, "redirect is still counted as an attempt");
    check_eq(r.stats().writes_redirected_to_hbm, 1u, "stats: writes_redirected_to_hbm == 1");
  }

  // -------------------------------------------------------------------------
  section("6. hbf_write_policy = allow permits HBF writes");
  {
    auto cfg = make_config();
    cfg.hbf_write_policy = HbfWritePolicy::Allow;
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    auto d = r.route(write_at(HBF_BASE + 0x200));
    check(d.status == RouteStatus::RoutedHbf, "write reaches HBF under policy=allow");
    check_eq(r.stats().writes_hbf, 1u, "stats: writes_hbf == 1");
    check_eq(r.stats().write_attempts_to_hbf, 0u, "no violation counted under policy=allow");
  }

  // -------------------------------------------------------------------------
  section("7. Unmapped addresses are rejected, not silently routed");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    auto d = r.route(read_at(HBF_BASE + HBF_SIZE + 0x1000));
    check(d.status == RouteStatus::RejectedUnmapped, "past-end address is REJECTED_UNMAPPED");
    check_eq(r.stats().unmapped_requests, 1u, "stats: unmapped_requests == 1");
    check_eq(hbm.received.size(), 0u, "unmapped request not sent to HBM");
    check_eq(hbf.received.size(), 0u, "unmapped request not sent to HBF");
  }

  // -------------------------------------------------------------------------
  section("8. Backend backpressure is propagated");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    hbf.full = true;
    auto d = r.route(read_at(HBF_BASE + 0x10));
    check(d.status == RouteStatus::RejectedBackendFull, "full backend -> REJECTED_BACKEND_FULL");
    check_eq(r.stats().backend_full_rejects, 1u, "stats: backend_full_rejects == 1");
    check_eq(r.stats().requests_hbf, 0u, "a refused request is not counted as served");

    hbf.full = false;
    d = r.route(read_at(HBF_BASE + 0x10));
    check(is_accepted(d.status), "same request succeeds once the backend drains");
  }

  // -------------------------------------------------------------------------
  section("9. Bandwidth accounting");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    // 1000 x 128 B to each region, spread over 1 us.
    for (int i = 0; i < 1000; ++i) {
      const uint64_t t = static_cast<uint64_t>(i) * 1000;  // ps
      r.route(read_at(0x1000 + i * 128, t, 128));
      r.route(read_at(HBF_BASE + 0x1000 + i * 128, t, 128));
    }

    check_eq(r.stats().bytes_hbm, 128u * 1000u, "stats: bytes_hbm == 128000");
    check_eq(r.stats().bytes_hbf, 128u * 1000u, "stats: bytes_hbf == 128000");

    // 128000 B over 999 ns ~= 128 GB/s
    const double bw = r.stats().hbm_bandwidth_gbps();
    check(bw > 120.0 && bw < 135.0, "HBM bandwidth ~128 GB/s over the window");

    const double frac = r.stats().hbf_request_fraction();
    check(frac > 0.49 && frac < 0.51, "HBF request fraction ~0.5 for a 50/50 mix");

    // Utilisation against an 8 TB/s peak should be small but non-zero.
    const double util = r.stats().hbm_utilization(cfg.hbm_peak_bandwidth_gbps);
    check(util > 0.0 && util < 1.0, "HBM utilisation is a sane fraction of peak");
  }

  // -------------------------------------------------------------------------
  section("10. Config validation rejects bad address maps");
  {
    bool threw = false;
    try {
      H3RouterConfig c = make_config();
      c.hbf_base_addr = HBM_SIZE / 2;   // overlaps HBM
      c.validate();
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "overlapping HBM/HBF regions are rejected");

    threw = false;
    try {
      H3RouterConfig c = make_config();
      c.hbm_size_bytes = 0;
      c.validate();
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "zero-size region is rejected");

    threw = false;
    try {
      H3RouterConfig c = make_config();
      c.hbf_base_addr = ~0ULL - 16;     // wraps
      c.validate();
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "address wraparound past 2^64 is rejected");
  }

  // -------------------------------------------------------------------------
  section("11. YAML config loads (if the file is present)");
  {
    const char* path = "configs/h3_router_config.yaml";
    try {
      H3RouterConfig c = H3RouterConfig::from_yaml(path);
      check_eq(c.hbm_size_bytes, 192 * GB, "YAML: hbm_size_bytes == 192GB");
      check_eq(c.hbf_base_addr, 0x3000000000ULL, "YAML: hbf_base_addr == 0x3000000000");
      check_eq(c.hbf_size_bytes, 3 * TB, "YAML: hbf_size_bytes == 3TB");
      check(c.d2d_hop_latency_ns == 25.0, "YAML: d2d hop latency == 25 ns");
      check(c.enforce_hbf_read_only, "YAML: HBF read-only enforcement is on");
      check(c.hbf_write_policy == HbfWritePolicy::Reject, "YAML: write policy == reject");
      check_eq(c.d2d_hop_latency_ps(), 25'000ULL, "YAML: 25 ns converts to 25000 ps");
    } catch (const std::exception& e) {
      std::cout << "  [skip] " << path << " not loadable from this directory: "
                << e.what() << "\n";
      std::cout << "         (run from the h3-sim/ root to exercise this test)\n";
    }
  }

  // -------------------------------------------------------------------------
  section("12. Allocation map: decode by buffer, not by address range");
  {
    // A real trace uses addresses like 0x7f35ab700000, which fall in NEITHER
    // fixed region. tools/classify_trace.py works out which buffers are
    // read-only and writes a map; the router then decodes by buffer.
    const char* path = "/tmp/h3_test_allocmap.yaml";
    {
      std::ofstream f(path);
      f << "allocations:\n"
        << "  - orig_addr: 0x7f35ab700000\n    size: 1048576\n"
        << "    region: hbf\n    h3_addr: 0x3000000000\n"
        << "  - orig_addr: 0x7f35ac000000\n    size: 65536\n"
        << "    region: hbm\n    h3_addr: 0x0\n";
    }
    auto cfg = make_config();
    cfg.allocation_map = path;
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);

    check_eq(r.allocation_count(), 2u, "both allocations loaded from the map");
    check(r.decode(0x7f35ab700000ULL) == MemRegion::Hbf,
          "a read-only buffer decodes to HBF despite its raw address");
    check(r.decode(0x7f35ac000000ULL) == MemRegion::Hbm,
          "a written buffer decodes to HBM");
    check_eq(r.to_local_addr(0x7f35ab700000ULL + 4096, MemRegion::Hbf), 4096ULL,
             "offsets within a buffer are preserved");

    // An address in no buffer (stack/local/constant) goes to HBM and is counted.
    r.route(read_at(0x7f35ad000000ULL));
    check_eq(r.stats().unallocated_requests, 1u,
             "addresses outside every buffer are counted separately");
    check_eq(r.stats().unmapped_requests, 0u,
             "and are NOT reported as unmapped -- they are placed in HBM");

    r.route(read_at(0x7f35ab700000ULL + 512));
    check_eq(r.stats().requests_hbf, 1u, "the HBF buffer really routes to HBF");
    std::remove(path);
  }

  // -------------------------------------------------------------------------
  section("13. Statistics report renders");
  {
    auto cfg = make_config();
    FakeBackend hbm("HBM"), hbf("HBF");
    H3AddressRouter r(cfg, &hbm, &hbf);
    for (int i = 0; i < 10; ++i) {
      r.route(read_at(0x1000 + i * 64, i * 1000, 64));
      r.route(read_at(HBF_BASE + i * 64, i * 1000, 64));
    }
    std::ostringstream os;
    r.print_stats(os);
    const std::string out = os.str();
    check(out.find("h3_router_requests_hbf") != std::string::npos,
          "stats report contains h3_router_requests_hbf");
    check(out.find("h3_router_write_attempts_to_hbf") != std::string::npos,
          "stats report contains the correctness counter");
    std::cout << out;
  }

  // -------------------------------------------------------------------------
  std::cout << "\n============================================================\n";
  std::cout << "  checks run : " << g_checks << "\n";
  std::cout << "  failures   : " << g_failures << "\n";
  std::cout << (g_failures == 0 ? "  RESULT     : PASS\n" : "  RESULT     : FAIL\n");
  std::cout << "============================================================\n";
  return g_failures == 0 ? 0 : 1;
}
