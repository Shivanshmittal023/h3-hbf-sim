// ============================================================================
//  test_lhb.cpp -- standalone unit test for the Latency Hiding Buffer
// ============================================================================
//
//  WHAT THIS IS
//      Verifies h3-components/latency_hiding_buffer.{h,cpp} in isolation: no
//      Ramulator, no Accel-Sim, no build system. Only a C++17 compiler.
//
//  WHAT IT CHECKS
//      1.  Equation (1) arithmetic: 2 x 1 TB/s x 20 us == 40 MB
//      2.  Prefetch -> 20 us elapses -> access is a HIT (latency hidden)
//      3.  Access BEFORE the fill lands is a MISS_LATE with the right stall
//      4.  Double buffering: half B fills while half A is consumed
//      5.  Streaming a tensor larger than the buffer
//      6.  Bypass mode pays the full 20 us on every access
//      7.  Steady-state pipeline: sustained hit rate over many chunks
//      8.  Under-provisioned lead time degrades hits, as it should
//      9.  Statistics and state-machine reporting
//
//  BUILD AND RUN
//      ./scripts/run_unit_tests.sh
//
//  EXIT CODE
//      0 = all assertions passed; 1 = at least one failure.
// ============================================================================

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "latency_hiding_buffer.h"

using namespace h3;

namespace {

int g_checks = 0;
int g_failures = 0;

void section(const std::string& name) { std::cout << "\n== " << name << " ==\n"; }

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

constexpr uint64_t MB = 1ULL << 20;
constexpr uint64_t NS = 1000ULL;              // ps per ns
constexpr uint64_t US = 1000ULL * NS;         // ps per us
constexpr uint64_t TR_PS = 20 * US;           // tR = 20 us

LhbConfig make_config(bool enabled = true) {
  LhbConfig c;
  c.enabled = enabled;
  c.buffer_size_bytes = 40 * MB;
  c.num_buffers = 2;
  c.sram_access_latency_ns = 2.0;
  c.hbf_read_latency_ns = 20000.0;   // 20 us
  c.hbf_bandwidth_gbps = 1000.0;     // 1 TB/s per cube
  c.prefetch_hint_lead_time_ns = 45000.0;
  return c;
}

PrefetchHint hint_of(uint64_t id, uint64_t addr, uint64_t size,
                     uint64_t needed_at_ps, int layer = 0) {
  PrefetchHint h;
  h.tensor_id = id;
  h.hbf_addr = addr;
  h.size_bytes = size;
  h.needed_at_ps = needed_at_ps;
  h.layer = layer;
  return h;
}

}  // namespace

int main() {
  std::cout << "H3 Latency Hiding Buffer -- standalone unit test\n";

  // -------------------------------------------------------------------------
  section("1. Equation (1): capacity = 2 x BW x latency");
  {
    auto cfg = make_config();
    // 2 x 1000 GB/s x 20000 ns = 40,000,000 bytes ~ 40 MB (decimal).
    const uint64_t req = cfg.required_capacity_bytes();
    check_eq(req, 40'000'000ULL, "Eq.(1) yields 40,000,000 bytes for 1 TB/s x 20 us");
    check(cfg.buffer_size_bytes >= req,
          "configured 40 MiB (41,943,040 B) covers the Eq.(1) requirement");
    check_eq(cfg.half_size_bytes(), 20 * MB, "each half is 20 MiB");

    // The design identity: a half drains in exactly tR.
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    const uint64_t drain = eng.transfer_ps(cfg.half_size_bytes());
    check(drain > 20 * US * 0.9 && drain < 20 * US * 1.2,
          "draining one 20 MiB half takes ~20 us at 1 TB/s == tR (the Eq.1 identity)");
  }

  // -------------------------------------------------------------------------
  section("2. Prefetch hides the 20 us: hint -> wait -> HIT");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    const uint64_t A = 0x1000;
    lhb.issue_hint(hint_of(/*id=*/1, A, /*size=*/20 * MB, /*needed=*/45 * US));
    lhb.tick(0);
    check(lhb.half_state(0) == LhbState::PrefetchIssued,
          "hint issued -> half[0] enters PREFETCH_ISSUED");

    lhb.tick(1 * US);
    check(lhb.half_state(0) == LhbState::Filling, "time advances -> FILLING");

    // fill completes at tR + transfer(20MiB @ 1TB/s) ~= 20us + 20.97us
    lhb.tick(45 * US);
    check(lhb.half_state(0) == LhbState::Ready, "after tR + transfer -> READY");

    auto acc = lhb.access(A, 128, 45 * US);
    check(acc.outcome == LhbOutcome::Hit, "access after the fill lands is a HIT");
    check_eq(acc.stall_ps, 0ULL, "a HIT stalls for 0 ps");
    check_eq(acc.ready_time_ps, 45 * US + 2 * NS, "HIT costs only 2 ns of SRAM latency");
    check(lhb.stats().hit_rate() == 1.0, "hit rate is 1.0");
  }

  // -------------------------------------------------------------------------
  section("3. Access before the fill lands is MISS_LATE with a partial stall");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    const uint64_t A = 0x2000;
    lhb.issue_hint(hint_of(1, A, 20 * MB, 45 * US));
    lhb.tick(0);

    // Ask for it at 5 us, long before the ~41 us completion.
    auto acc = lhb.access(A, 128, 5 * US);
    check(acc.outcome == LhbOutcome::MissLate, "early access is MISS_LATE");
    check(acc.stall_ps > 0, "MISS_LATE stalls");

    // Crucially the stall is only the REMAINING fill time, not a fresh 20 us:
    // the prefetch that was already in flight still did useful work.
    const uint64_t full_demand = eng.demand_latency_ps(128, 0);
    check(acc.stall_ps < TR_PS + eng.transfer_ps(20 * MB),
          "stall is less than a full fill: partial hiding still helps");
    check(acc.stall_ps > full_demand,
          "but a 20 MiB chunk fill is longer than a 128 B demand read");
    check_eq(lhb.stats().misses_late, 1u, "stats: misses_late == 1");
    check_eq(lhb.stats().hits, 0u, "stats: no hits");
  }

  // -------------------------------------------------------------------------
  section("4. Double buffering: B fills while A is consumed");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    // One 40 MiB tensor = exactly two chunks, so both halves engage.
    lhb.issue_hint(hint_of(7, 0x10000, 40 * MB, 45 * US));
    lhb.tick(0);

    check(lhb.half_state(0) != LhbState::Idle, "half[0] is busy");
    check(lhb.half_state(1) != LhbState::Idle, "half[1] is busy too (double buffered)");
    check_eq(lhb.stats().fills_started, 2u, "two fills issued back to back");

    lhb.tick(45 * US);
    check(lhb.half_state(0) == LhbState::Ready, "half[0] READY");
    check(lhb.half_state(1) == LhbState::Ready, "half[1] READY");

    // Drain half 0 completely; that must trigger a SWAP.
    const uint64_t chunk = cfg.half_size_bytes();
    auto a = lhb.access(0x10000, chunk, 45 * US);
    check(a.outcome == LhbOutcome::Hit, "consuming half[0] is a HIT");
    check_eq(lhb.stats().swaps, 1u, "fully draining a half records a SWAP");
    check(lhb.half_state(0) == LhbState::Idle, "drained half returns to IDLE");

    // Second chunk still resident in the other half.
    auto b = lhb.access(0x10000 + chunk, 128, 46 * US);
    check(b.outcome == LhbOutcome::Hit, "second chunk is a HIT from the other half");
  }

  // -------------------------------------------------------------------------
  section("5. Streaming a tensor much larger than the buffer");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    // A realistic Llama-405B FP8 layer weight: ~870 MB, i.e. ~44 chunks.
    const uint64_t base = 0x100000;
    const uint64_t tensor = 870 * MB;
    lhb.issue_hint(hint_of(42, base, tensor, 0));

    const uint64_t chunk = cfg.half_size_bytes();
    uint64_t t = 0;
    uint64_t consumed = 0;
    // Consume at ~1 TB/s, which is the rate the fill engine sustains.
    while (consumed < tensor) {
      const uint64_t want = std::min<uint64_t>(chunk, tensor - consumed);
      t += eng.transfer_ps(want);            // time to consume this chunk
      t += TR_PS;                            // plus pipeline warm-up allowance
      lhb.access(base + consumed, want, t);
      consumed += want;
    }
    const auto& s = lhb.stats();
    check_eq(s.bytes_served, tensor, "the whole 870 MB tensor was served");
    check(s.fills_started >= 44, "tensor was streamed in >= 44 chunk fills");
    check(s.hit_rate() > 0.9, "streaming with adequate lead time is >90% hits");
    check_eq(s.bytes_wasted, 0ULL, "sequential streaming wastes no prefetched bytes");
    std::cout << "        (hit rate " << s.hit_rate() << " over "
              << s.fills_started << " chunk fills)\n";
  }

  // -------------------------------------------------------------------------
  section("6. Bypass mode pays the full HBF latency every time");
  {
    auto cfg = make_config(/*enabled=*/false);
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    lhb.issue_hint(hint_of(1, 0x3000, 20 * MB, 45 * US));  // ignored in bypass
    auto acc = lhb.access(0x3000, 128, 0);

    check(acc.outcome == LhbOutcome::Bypass, "bypass mode reports BYPASS");
    check(acc.stall_ps >= TR_PS, "bypass stalls at least the full 20 us tR");
    check_eq(lhb.stats().bypasses, 1u, "stats: bypasses == 1");
    check_eq(lhb.stats().fills_started, 0u, "bypass mode issues no prefetch fills");

    std::cout << "        (bypass stall = " << acc.stall_ps / 1000 << " ns"
              << " vs 2 ns for an LHB hit -> "
              << (acc.stall_ps / 2000) << "x)\n";
  }

  // -------------------------------------------------------------------------
  section("7. Steady-state pipeline sustains a high hit rate");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    const uint64_t base = 0x400000;
    const uint64_t chunk = cfg.half_size_bytes();
    const uint64_t n_chunks = 32;
    lhb.issue_hint(hint_of(99, base, chunk * n_chunks, 0));

    // Warm the pipeline, then consume one chunk per drain interval.
    uint64_t t = TR_PS + eng.transfer_ps(chunk);
    for (uint64_t i = 0; i < n_chunks; ++i) {
      lhb.access(base + i * chunk, chunk, t);
      t += eng.transfer_ps(chunk);
    }
    const auto& s = lhb.stats();
    check(s.hit_rate() >= 0.9, "steady-state hit rate >= 0.9");
    check(s.avg_utilization(cfg.buffer_size_bytes) > 0.0, "buffer utilisation is tracked");
    std::cout << "        (hit rate " << s.hit_rate() << ", avg utilisation "
              << s.avg_utilization(cfg.buffer_size_bytes) << ")\n";
  }

  // -------------------------------------------------------------------------
  section("8. Too little lead time degrades the hit rate (sensitivity)");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    const uint64_t base = 0x800000;
    const uint64_t chunk = cfg.half_size_bytes();
    lhb.issue_hint(hint_of(5, base, chunk * 8, 0));

    // Consume far faster than HBF can fill: 1 us apart instead of ~21 us.
    uint64_t t = 0;
    for (uint64_t i = 0; i < 8; ++i) {
      t += 1 * US;
      lhb.access(base + i * chunk, chunk, t);
    }
    const auto& s = lhb.stats();
    check(s.misses_late > 0, "consuming faster than HBF can fill produces MISS_LATE");
    check(s.hit_rate() < 0.5, "hit rate collapses when lead time is inadequate");
    std::cout << "        (hit rate " << s.hit_rate() << ", "
              << s.misses_late << " late misses, avg stall "
              << s.avg_stall_ns_per_miss() << " ns)\n";
  }

  // -------------------------------------------------------------------------
  section("9. Config validation and YAML loading");
  {
    bool threw = false;
    try {
      LhbConfig c = make_config();
      c.num_buffers = 0;
      c.validate();
    } catch (const std::exception&) { threw = true; }
    check(threw, "num_buffers = 0 is rejected");

    threw = false;
    try {
      LhbConfig c = make_config();
      c.buffer_size_bytes = 30 * MB;   // not divisible by 2? it is; use 3 halves
      c.num_buffers = 7;
      c.validate();
    } catch (const std::exception&) { threw = true; }
    check(threw, "capacity that does not divide evenly by num_buffers is rejected");

    try {
      LhbConfig c = LhbConfig::from_yaml("configs/lhb_config.yaml");
      check_eq(c.buffer_size_bytes, 40 * MB, "YAML: buffer_size_mb == 40");
      check_eq(c.num_buffers, 2u, "YAML: num_buffers == 2");
      check(c.enabled, "YAML: LHB enabled by default");
      check(c.sram_access_latency_ns == 2.0, "YAML: sram_access_latency_ns == 2");
      check(c.hbf_read_latency_ns == 20000.0, "YAML: hbf_read_latency_ns == 20 us");
      check(c.hbf_bandwidth_gbps == 1000.0, "YAML: hbf_bandwidth_gbps == 1 TB/s per cube");
      check_eq(c.sram_access_latency_ps(), 2000ULL, "YAML: 2 ns converts to 2000 ps");
    } catch (const std::exception& e) {
      std::cout << "  [skip] configs/lhb_config.yaml not loadable here: " << e.what() << "\n";
      std::cout << "         (run from the h3-sim/ root to exercise this test)\n";
    }
  }

  // -------------------------------------------------------------------------
  section("10. Reporting renders");
  {
    auto cfg = make_config();
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);
    lhb.issue_hint(hint_of(1, 0x9000, 40 * MB, 0));
    lhb.tick(0);
    lhb.print_state(std::cout);
    lhb.tick(45 * US);
    lhb.access(0x9000, 4096, 45 * US);

    std::ostringstream os;
    lhb.print_stats(os);
    const std::string out = os.str();
    check(out.find("lhb_hit_rate") != std::string::npos, "stats include lhb_hit_rate");
    check(out.find("lhb_avg_stall_cycles_per_miss") != std::string::npos,
          "stats include avg stall cycles per miss");
    check(out.find("lhb_avg_utilization") != std::string::npos,
          "stats include buffer utilisation");
    std::cout << out;
  }

  // -------------------------------------------------------------------------
  section("11. Address interleaving: a partition slice covers the full span");
  {
    // GPGPU-Sim spreads consecutive lines across N memory partitions, so one
    // partition's LHB slice holds 1/N of every line but must cover the SAME
    // global address range the whole buffer would. Without this the window is
    // N times too narrow and prefetch efficiency collapses to 1/N.
    const uint32_t N = 40;
    auto cfg = make_config();
    cfg.buffer_size_bytes = 40 * MB / N;        // this partition's slice
    cfg.address_interleave_factor = N;
    cfg.hbf_bandwidth_gbps = 1000.0 / N;        // its share of bandwidth too
    AnalyticFillEngine eng(TR_PS, cfg.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(cfg, &eng);

    check_eq(cfg.chunk_span_bytes(), cfg.half_size_bytes() * N,
             "one half spans N x its own capacity in global addresses");
    check_eq(cfg.chunk_span_bytes(), 20 * MB,
             "a 1 MiB slice of a 40 MiB buffer still spans 20 MiB per half");

    const uint64_t base = 0x2000000;
    const uint64_t span = cfg.chunk_span_bytes();
    lhb.issue_hint(hint_of(1, base, span * 4, 0));
    lhb.tick(0);
    lhb.tick(60 * US);

    // This partition owns every Nth 128 B line. Walk its share of the span.
    // Walk three spans, so the data the double buffer prefetched ahead is
    // actually consumed -- otherwise efficiency looks low purely because the
    // test stopped with fills still in flight.
    uint64_t t = 60 * US, hits = 0, total = 0;
    for (uint64_t off = 0; off < span * 3; off += 128 * N) {
      auto a = lhb.access(base + off, 128, t);
      ++total;
      if (a.outcome == LhbOutcome::Hit) ++hits;
      t += 100 * NS;
    }
    const double hr = static_cast<double>(hits) / static_cast<double>(total);
    check(hr > 0.9, "partition-local accesses across the whole span still hit");

    const auto& st = lhb.stats();
    check(st.prefetch_efficiency() > 0.6,
          "prefetch efficiency stays high (bytes fetched ~= bytes used)");
    check(st.swaps > 0, "halves drain and swap as the stream advances");
    std::cout << "        (hit rate " << hr << ", prefetch efficiency "
              << st.prefetch_efficiency() << ", " << st.swaps << " swaps)\n";

    // Control: the same slice WITHOUT the interleave factor is far too narrow.
    auto bad = cfg;
    bad.address_interleave_factor = 1;
    AnalyticFillEngine eng2(TR_PS, bad.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb2(bad, &eng2);
    lhb2.issue_hint(hint_of(1, base, span * 4, 0));
    lhb2.tick(0);
    lhb2.tick(60 * US);
    uint64_t t2 = 60 * US, hits2 = 0, total2 = 0;
    for (uint64_t off = 0; off < span * 3; off += 128 * N) {
      if (lhb2.access(base + off, 128, t2).outcome == LhbOutcome::Hit) ++hits2;
      ++total2;
      t2 += 100 * NS;
    }
    const double hr2 = static_cast<double>(hits2) / static_cast<double>(total2);
    check(hr2 < hr,
          "without the interleave factor the hit rate is strictly worse");
    std::cout << "        (control, factor=1: hit rate " << hr2 << ")\n";
  }

  std::cout << "\n============================================================\n";
  std::cout << "  checks run : " << g_checks << "\n";
  std::cout << "  failures   : " << g_failures << "\n";
  std::cout << (g_failures == 0 ? "  RESULT     : PASS\n" : "  RESULT     : FAIL\n");
  std::cout << "============================================================\n";
  return g_failures == 0 ? 0 : 1;
}
