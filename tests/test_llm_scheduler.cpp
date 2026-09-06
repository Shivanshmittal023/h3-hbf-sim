// ============================================================================
//  test_llm_scheduler.cpp -- standalone unit test for the LLM prefetch scheduler
// ============================================================================
//  Verifies h3-components/llm_prefetch_scheduler.{h,cpp}: the analytic size
//  model, the HBF/HBM address layout, and hint timing. No simulator needed.
//
//  The headline assertion is that the analytic parameter count reproduces
//  "405B" and that KV occupancy reproduces the paper's ~35% / ~84% figures.
//  If either drifts, the model description is wrong and every downstream
//  experiment is measuring the wrong workload.
//
//  BUILD AND RUN:  ./scripts/run_unit_tests.sh
// ============================================================================

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "llm_prefetch_scheduler.h"

using namespace h3;

namespace {
int g_checks = 0, g_failures = 0;
void section(const std::string& n) { std::cout << "\n== " << n << " ==\n"; }
void check(bool c, const std::string& w) {
  ++g_checks;
  if (c) std::cout << "  [ ok ] " << w << "\n";
  else { ++g_failures; std::cout << "  [FAIL] " << w << "\n"; }
}
template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& w) {
  ++g_checks;
  if (got == static_cast<A>(want)) std::cout << "  [ ok ] " << w << "\n";
  else { ++g_failures; std::cout << "  [FAIL] " << w << " (got " << got << ", want " << want << ")\n"; }
}
void check_near(double got, double want, double tol, const std::string& w) {
  ++g_checks;
  if (std::fabs(got - want) <= tol) std::cout << "  [ ok ] " << w << " (" << got << ")\n";
  else { ++g_failures; std::cout << "  [FAIL] " << w << " (got " << got << ", want " << want << " +/- " << tol << ")\n"; }
}

ModelConfig make_405b() {
  ModelConfig c;   // defaults already describe Llama 3.1 405B
  return c;
}
constexpr uint64_t GiB = 1ULL << 30;
constexpr uint64_t TiB = 1ULL << 40;
}  // namespace

int main() {
  std::cout << "H3 LLM prefetch scheduler -- standalone unit test\n";

  // -------------------------------------------------------------------------
  section("1. Analytic model size reproduces Llama 3.1 405B");
  {
    ModelConfig c = make_405b();
    check_eq(c.weight_bytes_per_layer(), 3'187'703'808ULL,
             "per-layer weights == 3,187,703,808 B");
    check_eq(c.weight_bytes_all_layers(), 3'187'703'808ULL * 126,
             "126 layers == 401.7 GB");
    check_eq(c.embedding_bytes(), 2ULL * 128256 * 16384, "embed + lm_head == 4.2 GB");

    const double params_b = static_cast<double>(c.total_parameters()) / 1e9;
    check_near(params_b, 405.9, 0.5,
               "TOTAL parameter count is ~405B (validates the decomposition)");
  }

  // -------------------------------------------------------------------------
  section("2. KV cache reproduces the paper's occupancy figures");
  {
    // 1M tokens over 8 GPUs -> paper says ~35% of capacity
    ModelConfig c = make_405b();
    c.sequence_length = 1024 * 1024;
    c.num_gpus = 8;
    c.tensor_parallel_size = 8;
    check_eq(c.kv_bytes_per_token(), 8'257'536ULL,
             "MHA 128 heads FP16 -> 8,257,536 B/token");

    const double occ_1m =
        static_cast<double>(c.shared_kv_bytes_per_gpu()) / static_cast<double>(3 * TiB);
    check_near(occ_1m * 100.0, 32.8, 3.0, "1M/8GPU shared-KV occupancy of HBF ~35%");

    // 10M tokens over 32 GPUs -> paper says ~84%
    c.sequence_length = 10ULL * 1024 * 1024;
    c.num_gpus = 32;
    c.tensor_parallel_size = 8;
    const double occ_10m =
        static_cast<double>(c.shared_kv_bytes_per_gpu()) / static_cast<double>(3 * TiB);
    check_near(occ_10m * 100.0, 82.0, 4.0, "10M/32GPU shared-KV occupancy of HBF ~84%");

    // Control: true GQA gives 16x less, and would NOT match the paper.
    c.num_cache_heads = 8;
    c.kv_dtype_bytes = 2;
    const double occ_gqa =
        static_cast<double>(c.shared_kv_bytes_per_gpu()) / static_cast<double>(3 * TiB);
    check(occ_gqa < 0.10, "GQA (8 kv heads) gives <10% occupancy -- does not match the paper");
  }

  // -------------------------------------------------------------------------
  section("3. Data placement: weights and shared KV in HBF, generated KV in HBM");
  {
    ModelConfig c = make_405b();
    c.sequence_length = 65536;   // keep the test quick
    LLMPrefetchScheduler s(c);

    bool all_hbf_ro = true, gen_in_hbm = true, shared_in_hbf = true;
    for (const auto& t : s.tensors()) {
      if (t.region == TensorRegion::Hbf && !t.read_only) all_hbf_ro = false;
      if (t.kind == TensorKind::KvGenerated && t.region != TensorRegion::Hbm) gen_in_hbm = false;
      if (t.kind == TensorKind::KvShared && t.region != TensorRegion::Hbf) shared_in_hbf = false;
    }
    check(all_hbf_ro, "every HBF tensor is marked read-only");
    check(gen_in_hbm, "generated KV cache is in HBM (writable)");
    check(shared_in_hbf, "shared precomputed KV cache is in HBF (read-only)");

    check_eq(s.stats().tensors_hbf, 1 + 126 * 9 + 1 + 126,
             "HBF tensor count == embed + 9/layer + lm_head + KV-shared/layer");
    check_eq(s.stats().tensors_hbm, 126 + 1, "HBM tensors == generated KV/layer + activations");
  }

  // -------------------------------------------------------------------------
  section("4. Address layout is non-overlapping, aligned, and in the right regions");
  {
    ModelConfig c = make_405b();
    c.sequence_length = 65536;
    LLMPrefetchScheduler s(c);

    uint64_t prev_end_hbf = 0, prev_end_hbm = 0;
    bool ordered = true, aligned = true, in_region = true;
    for (const auto& t : s.tensors()) {
      if (t.size_bytes == 0) continue;
      if (t.local_addr % c.tensor_alignment_bytes != 0) aligned = false;
      if (t.region == TensorRegion::Hbf) {
        if (t.local_addr < prev_end_hbf) ordered = false;
        prev_end_hbf = t.local_addr + t.size_bytes;
        if (t.global_addr < c.hbf_base_addr) in_region = false;
      } else {
        if (t.local_addr < prev_end_hbm) ordered = false;
        prev_end_hbm = t.local_addr + t.size_bytes;
        if (t.global_addr < c.hbm_base_addr) in_region = false;
      }
    }
    check(ordered, "tensors do not overlap within their region");
    check(aligned, "every tensor is 4 KiB aligned");
    check(in_region, "global addresses sit at or above their region base");

    // The layout must fit inside the modeled capacities.
    check(prev_end_hbf <= c.hbf_capacity_bytes_per_gpu, "HBF layout fits in 3 TiB");
    check(prev_end_hbm <= c.hbm_capacity_bytes_per_gpu, "HBM layout fits in 192 GiB");

    // Reverse lookup must agree with the layout.
    const auto& t5 = s.tensors()[5];
    check_eq(s.tensor_at(t5.global_addr + 64), static_cast<int>(t5.id),
             "tensor_at() resolves an address back to its tensor");
    check_eq(s.tensor_at(0xFFFFFFFFFFFFULL), -1, "tensor_at() returns -1 for an unmapped address");
  }

  // -------------------------------------------------------------------------
  section("5. Hints lead their tensors by the configured lead time");
  {
    ModelConfig c = make_405b();
    c.sequence_length = 65536;
    LLMPrefetchScheduler s(c);

    const uint64_t lead = c.lead_time_ps();
    check_eq(lead, 50'000'000ULL, "lead time = (45000 + 5000) ns = 50,000,000 ps");

    uint64_t checked = 0, correct = 0, clamped = 0;
    for (const auto& t : s.tensors()) {
      if (t.region != TensorRegion::Hbf) continue;
      ++checked;
      if (t.hint_at_ps == 0 && t.needed_at_ps <= lead) { ++clamped; continue; }
      if (t.needed_at_ps - t.hint_at_ps == lead) ++correct;
    }
    check(checked > 0, "there are HBF tensors to schedule");
    check_eq(correct + clamped, checked, "every hint leads by exactly the lead time, or is clamped at t=0");
    check(clamped > 0, "the first tensors are clamped: no room to hide tR at pass start");
    std::cout << "        (" << clamped << " of " << checked
              << " hints clamped at the start of the forward pass)\n";
  }

  // -------------------------------------------------------------------------
  section("6. Hints are issued in time order as the clock advances");
  {
    ModelConfig c = make_405b();
    c.sequence_length = 65536;
    c.max_outstanding_hints = 1000000;    // do not throttle in this test
    LLMPrefetchScheduler s(c);

    uint64_t issued_total = 0;
    const uint64_t dur = s.forward_pass_duration_ps();
    for (uint64_t t = 0; t <= dur; t += dur / 100 + 1) {
      issued_total += s.tick(t, nullptr);
    }
    // Tick exactly at the end of the pass: the coarse sampling above can step
    // over the final window, whereas the real simulator ticks every cycle.
    issued_total += s.tick(dur, nullptr);
    check_eq(issued_total, s.stats().tensors_hbf,
             "every HBF tensor got exactly one hint over the pass");
    check_eq(s.stats().hints_pending, 0u, "no hints left pending at the end");

    s.rewind();
    check_eq(s.stats().hints_issued, 0u, "rewind() resets the hint cursor");
  }

  // -------------------------------------------------------------------------
  section("7. Scheduler drives the LHB end to end");
  {
    ModelConfig c = make_405b();
    c.sequence_length = 65536;
    c.num_layers = 4;                 // small pass, fast test
    LLMPrefetchScheduler s(c);

    LhbConfig lc;
    lc.enabled = true;
    lc.buffer_size_bytes = 40ULL << 20;
    lc.num_buffers = 2;
    lc.hbf_read_latency_ns = 20000.0;
    lc.hbf_bandwidth_gbps = 1000.0;
    AnalyticFillEngine eng(20'000'000ULL, lc.hbf_bandwidth_gbps);
    LatencyHidingBuffer lhb(lc, &eng);

    const uint64_t dur = s.forward_pass_duration_ps();
    uint64_t served = 0, hits = 0;
    const uint64_t step = dur / 200 + 1;
    for (uint64_t t = 0; t <= dur; t += step) {
      s.tick(t, &lhb);
      lhb.tick(t);
    }
    // Read back the HBF tensors in schedule order.
    for (const auto& td : s.tensors()) {
      if (td.region != TensorRegion::Hbf) continue;
      const uint64_t chunk = std::min<uint64_t>(td.size_bytes, 4096);
      auto a = lhb.access(td.local_addr, chunk, td.needed_at_ps);
      ++served;
      if (a.outcome == LhbOutcome::Hit) ++hits;
    }
    check(served > 0, "HBF tensors were read back through the LHB");
    check(lhb.stats().fills_started > 0, "the scheduler caused prefetch fills to be issued");
    std::cout << "        (" << hits << "/" << served << " accesses hit; "
              << lhb.stats().fills_started << " fills issued)\n";
  }

  // -------------------------------------------------------------------------
  section("8. Config validation and YAML loading");
  {
    bool threw = false;
    try { ModelConfig c = make_405b(); c.head_dim = 64; c.validate(); }
    catch (const std::exception&) { threw = true; }
    check(threw, "head_dim * heads != hidden_dim is rejected");

    threw = false;
    try { ModelConfig c = make_405b(); c.sequence_length = 100; c.validate(); }
    catch (const std::exception&) { threw = true; }
    check(threw, "sequence_length shorter than ISL+OSL is rejected");

    try {
      ModelConfig c = ModelConfig::from_yaml("configs/llama_405b_config.yaml");
      check_eq(c.num_layers, 126u, "YAML: num_layers == 126");
      check_eq(c.hidden_dim, 16384u, "YAML: hidden_dim == 16384");
      check_eq(c.intermediate_dim, 53248u, "YAML: intermediate_dim == 53248");
      check_eq(c.vocab_size, 128256u, "YAML: vocab_size == 128256");
      check_eq(c.weight_dtype_bytes, 1u, "YAML: FP8 weights");
      check_eq(c.num_cache_heads, 128u, "YAML: KV uses 128 heads (paper-matching)");
      check_eq(c.hbf_base_addr, 0x3000000000ULL, "YAML: HBF base matches the router config");
      check_near(static_cast<double>(c.total_parameters()) / 1e9, 405.9, 0.5,
                 "YAML config yields ~405B parameters");

      LLMPrefetchScheduler s(c);
      s.print_summary(std::cout);
      std::cout << "\n";
      s.print_layout(std::cout, 12);
      std::cout << "\n";
      s.print_schedule(std::cout, 8);
    } catch (const std::exception& e) {
      std::cout << "  [skip] configs/llama_405b_config.yaml not loadable here: "
                << e.what() << "\n";
    }
  }

  std::cout << "\n============================================================\n";
  std::cout << "  checks run : " << g_checks << "\n";
  std::cout << "  failures   : " << g_failures << "\n";
  std::cout << (g_failures == 0 ? "  RESULT     : PASS\n" : "  RESULT     : FAIL\n");
  std::cout << "============================================================\n";
  return g_failures == 0 ? 0 : 1;
}
