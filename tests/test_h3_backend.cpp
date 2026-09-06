// ============================================================================
//  test_h3_backend.cpp -- end-to-end invariants of the H3 memory backend
// ============================================================================
//
//  WHY THIS TEST EXISTS
//      The other test files exercise components in isolation. This one drives
//      the assembled backend -- router + LHB + scheduler + devices -- and
//      checks the invariants that only hold when they are wired together
//      correctly.
//
//      The headline one is that an LHB HIT MUST GENERATE NO DEVICE TRAFFIC.
//      That was silently violated for a long time: the router ran before the
//      LHB was consulted, and its adapter enqueued into the HBF device
//      immediately. Every request the buffer then served from SRAM had already
//      been charged to HBF, inflating its traffic by roughly the hit rate
//      (~26x on this workload). Nothing crashed; the numbers were simply wrong.
//
//      It surfaced only when Ramulator was attached, because a real timing
//      model cannot answer synchronously and threw instead of quietly
//      returning a number. The lesson is in the assertion below.
//
//  BUILD AND RUN
//      ./scripts/run_unit_tests.sh     (needs configs/, so run from h3-sim/)
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "h3_memory_backend.h"

using namespace h3;

namespace {
int g_checks = 0, g_failures = 0;
void section(const std::string& n) { std::cout << "\n== " << n << " ==\n"; }
void check(bool c, const std::string& w) {
  ++g_checks;
  if (c) std::cout << "  [ ok ] " << w << "\n";
  else { ++g_failures; std::cout << "  [FAIL] " << w << "\n"; }
}

// Pull a counter out of the backend's `key = value` stats dump.
long stat(const std::string& dump, const std::string& key) {
  const auto p = dump.find(key);
  if (p == std::string::npos) return -1;
  const auto eq = dump.find('=', p);
  if (eq == std::string::npos) return -1;
  return std::atol(dump.c_str() + eq + 1);
}

H3BackendConfig smoke_config() {
  H3BackendConfig c;                       // analytic devices by default
  c.router_config = "configs/h3_router_config.yaml";
  c.lhb_config = "configs/lhb_config.yaml";
  c.model_config = "configs/llama_smoke.yaml";
  return c;
}

constexpr uint64_t HBF_BASE = 0x3000000000ULL;
constexpr unsigned PARTITIONS = 40;        // -gpgpu_n_mem on the H100 config

// Drive `n` sequential reads into a region, draining completions as we go.
std::string drive(H3MemoryBackend& b, uint64_t base, int n, int warm_cycles) {
  for (int i = 0; i < warm_cycles; ++i) b.cycle();
  for (int i = 0; i < n; ++i) {
    H3MemRequest r;
    r.size_bytes = 32;
    r.opaque = reinterpret_cast<void*>(static_cast<intptr_t>(i + 1));
    r.addr = base + static_cast<uint64_t>(i) * 128;
    if (!b.full(false)) b.push(r);
    b.cycle();
    while (b.return_queue_top()) b.return_queue_pop();
  }
  std::ostringstream os;
  b.print_stats(os);
  return os.str();
}
}  // namespace

int main() {
  std::cout << "H3 memory backend -- end-to-end invariants\n";

  // -------------------------------------------------------------------------
  section("1. An LHB hit generates NO device traffic");
  {
    H3MemoryBackend b(0, smoke_config(), PARTITIONS);
    const std::string s = drive(b, HBF_BASE, 3000, 400000);

    const long accesses = stat(s, "lhb_accesses");
    const long hits = stat(s, "lhb_hits");
    const long dev_reads = stat(s, "hbf_device_reads");

    check(accesses > 0, "the workload reached the LHB");
    check(hits > 0, "the prefetch scheduler produced hits");
    check(dev_reads <= accesses - hits,
          "hbf_device_reads <= (accesses - hits): hits never touch the device");
    check(dev_reads < accesses,
          "device traffic is strictly less than LHB traffic (the 26x bug)");
    std::cout << "        (accesses " << accesses << ", hits " << hits
              << ", device reads " << dev_reads << ")\n";
  }

  // -------------------------------------------------------------------------
  section("2. Every request is accounted for -- none dropped, none free");
  {
    H3MemoryBackend b(0, smoke_config(), PARTITIONS);
    const std::string s = drive(b, HBF_BASE, 2000, 400000);

    const long pushed = stat(s, "h3_part0_pushed");
    const long completed = stat(s, "h3_part0_completed");
    check(pushed > 0, "requests were pushed");
    check(completed <= pushed, "completions never exceed pushes");
    check(pushed - completed < pushed / 2,
          "most requests completed (the rest are still in flight)");

    // Correctness counters -- non-zero means a real bug, not slowness.
    check(stat(s, "h3_router_write_attempts_to_hbf") == 0,
          "no writes reached the read-only HBF region");
    check(stat(s, "h3_router_unmapped_requests") == 0,
          "no addresses fell outside the H3 map");
  }

  // -------------------------------------------------------------------------
  section("3. HBM traffic reaches the HBM device");
  {
    H3MemoryBackend b(0, smoke_config(), PARTITIONS);
    const std::string s = drive(b, /*HBM base=*/0x100000ULL, 1000, 1000);

    check(stat(s, "h3_router_requests_hbm") > 0, "requests routed to HBM");
    check(stat(s, "hbm_device_reads") > 0, "and they reached the HBM device");
    check(stat(s, "h3_router_requests_hbf") == 0, "none leaked into HBF");
  }

  // -------------------------------------------------------------------------
  section("4. Per-GPU values are divided across memory partitions");
  {
    // The config carries PER-GPU bandwidth and LHB capacity; the backend must
    // divide by the partition count. Getting this wrong gave the simulated
    // machine 40x the bandwidth and 40x the buffer, with nothing looking odd.
    H3MemoryBackend one(0, smoke_config(), 1);
    H3MemoryBackend many(0, smoke_config(), PARTITIONS);
    const uint64_t cap_one = one.lhb().config().buffer_size_bytes;
    const uint64_t cap_many = many.lhb().config().buffer_size_bytes;

    check(cap_many < cap_one, "a partition's LHB slice is smaller than the whole");
    check(cap_one / cap_many >= PARTITIONS / 2,
          "the slice is ~1/N of the configured capacity");
    check(many.lhb().config().address_interleave_factor == PARTITIONS,
          "the interleave factor equals the partition count");
    std::cout << "        (whole " << cap_one / 1024 << " KiB, slice "
              << cap_many / 1024 << " KiB, factor "
              << many.lhb().config().address_interleave_factor << ")\n";
  }

  std::cout << "\n============================================================\n";
  std::cout << "  checks run : " << g_checks << "\n";
  std::cout << "  failures   : " << g_failures << "\n";
  std::cout << (g_failures == 0 ? "  RESULT     : PASS\n" : "  RESULT     : FAIL\n");
  std::cout << "============================================================\n";
  return g_failures == 0 ? 0 : 1;
}
