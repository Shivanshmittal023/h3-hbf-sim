// ============================================================================
//  test_macro_hazard.cpp -- guard against macro collisions with GPGPU-Sim
// ============================================================================
//
//  WHY THIS TEST EXISTS
//      The H3 headers are #included into GPGPU-Sim, a codebase that defines
//      435 object-like macros. `dram.h` alone does:
//
//          #define READ  'R'
//          #define WRITE 'W'
//
//      An early version of h3_address_router.h declared
//      `enum class ReqType { READ, WRITE }`, and the preprocessor rewrote the
//      enumerators to character literals. The build failed deep inside
//      l2cache.cc with "expected identifier before 'R'" -- a confusing error a
//      long way from its cause.
//
//      Every scoped-enum enumerator in h3-components is therefore CamelCase,
//      never SCREAMING_CASE, which is macro territory in C and C++.
//
//  WHAT THIS TEST DOES
//      Defines the real GPGPU-Sim macros BEFORE including the H3 headers, then
//      instantiates the whole stack. If anyone reintroduces a colliding
//      identifier, this fails at COMPILE time with a clear reason, instead of
//      surfacing as a mysterious error in someone else's translation unit.
//
//  MAINTENANCE
//      Refresh the macro list with:
//        grep -rhoE '^\s*#\s*define\s+[A-Z_][A-Z0-9_]*' \
//            gpu-simulator/gpgpu-sim/src gpu-simulator/gpgpu-sim/libcuda \
//            | awk '{print $NF}' | sort -u
// ============================================================================

// ---- GPGPU-Sim's macro environment, reproduced verbatim --------------------
#define READ 'R'          // src/gpgpu-sim/dram.h:48
#define WRITE 'W'         // src/gpgpu-sim/dram.h:49
#define BANK_IDLE 'I'     // src/gpgpu-sim/dram.h:50
#define BANK_ACTIVE 'A'   // src/gpgpu-sim/dram.h:51
#define DRAM 0x08         // clock domain mask
#define NAND 1
#define SWAP(a, b) do { auto t_ = (a); (a) = (b); (b) = t_; } while (0)
// Plausible future hazards -- cheap to guard against now.
#define HIT 1
#define IDLE 0
#define READY 2
#define FILLING 3
#define SERVING 4
#define ALLOW 5
#define REJECT 6
#define ACTIVATION 7

#include <iostream>
#include <string>

#include "h3_address_router.h"
#include "h3_memory_backend.h"
#include "latency_hiding_buffer.h"
#include "llm_prefetch_scheduler.h"

namespace {
int g_checks = 0, g_failures = 0;
void check(bool c, const std::string& w) {
  ++g_checks;
  if (c) std::cout << "  [ ok ] " << w << "\n";
  else { ++g_failures; std::cout << "  [FAIL] " << w << "\n"; }
}
}  // namespace

int main() {
  std::cout << "H3 macro-hazard guard -- GPGPU-Sim macros defined before H3 headers\n\n";

  // Reaching this point at all means every header parsed cleanly with the
  // macros active. The runtime checks below confirm the enumerators still
  // mean what they should.
  check(true, "all four H3 headers parse with GPGPU-Sim's macros defined");

  h3::H3RouterConfig rc;
  h3::H3AddressRouter router(rc);

  h3::H3Request q;
  q.addr = 0x1000;
  q.type = h3::ReqType::Read;
  check(router.inspect(q).status == h3::RouteStatus::RoutedHbm,
        "ReqType::Read / RouteStatus::RoutedHbm survive the preprocessor");

  q.addr = rc.hbf_base_addr + 0x40;
  q.type = h3::ReqType::Write;
  check(router.inspect(q).status == h3::RouteStatus::RejectedWriteToHbf,
        "ReqType::Write still triggers the HBF read-only rejection");

  check(router.decode(0) == h3::MemRegion::Hbm, "MemRegion::Hbm resolves");
  check(router.decode(rc.hbf_base_addr) == h3::MemRegion::Hbf, "MemRegion::Hbf resolves");

  h3::LhbConfig lc;
  h3::AnalyticFillEngine eng(20000000ULL, lc.hbf_bandwidth_gbps);
  h3::LatencyHidingBuffer lhb(lc, &eng);
  check(lhb.half_state(0) == h3::LhbState::Idle, "LhbState::Idle resolves");

  h3::PrefetchHint h;
  h.tensor_id = 1; h.hbf_addr = 0; h.size_bytes = 4096;
  lhb.issue_hint(h);
  lhb.tick(0);
  check(lhb.half_state(0) == h3::LhbState::PrefetchIssued, "LhbState::PrefetchIssued resolves");
  check(lhb.access(0, 64, 0).outcome == h3::LhbOutcome::MissLate, "LhbOutcome::MissLate resolves");

  h3::ModelConfig mc;
  mc.num_layers = 2;
  mc.sequence_length = 65536;
  h3::LLMPrefetchScheduler sched(mc);
  bool found_weight = false, found_kv = false;
  for (const auto& t : sched.tensors()) {
    if (t.kind == h3::TensorKind::WeightMlp) found_weight = true;
    if (t.kind == h3::TensorKind::KvGenerated) found_kv = true;
  }
  check(found_weight, "TensorKind::WeightMlp resolves");
  check(found_kv, "TensorKind::KvGenerated resolves");

  h3::H3BackendConfig bc;
  h3::H3MemoryBackend backend(0, bc);
  backend.cycle();
  check(backend.current_cycle() == 1, "H3MemoryBackend constructs and cycles");

  std::cout << "\n============================================================\n";
  std::cout << "  checks run : " << g_checks << "\n";
  std::cout << "  failures   : " << g_failures << "\n";
  std::cout << (g_failures == 0 ? "  RESULT     : PASS\n" : "  RESULT     : FAIL\n");
  std::cout << "============================================================\n";
  return g_failures == 0 ? 0 : 1;
}
