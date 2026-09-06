// ============================================================================
//  llm_prefetch_scheduler.cpp -- implementation of the LLM prefetch scheduler
// ============================================================================
//  See llm_prefetch_scheduler.h for the architectural description.
//  Standard library only, so it is unit-testable without the simulator.
// ============================================================================

#include "llm_prefetch_scheduler.h"

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

const char* to_string(TensorRegion r) { return r == TensorRegion::Hbm ? "HBM" : "HBF"; }

const char* to_string(TensorKind k) {
  switch (k) {
    case TensorKind::WeightQkv: return "WEIGHT_QKV";
    case TensorKind::WeightO: return "WEIGHT_O";
    case TensorKind::WeightMlp: return "WEIGHT_MLP";
    case TensorKind::WeightNorm: return "WEIGHT_NORM";
    case TensorKind::WeightEmbed: return "WEIGHT_EMBED";
    case TensorKind::KvShared: return "KV_SHARED";
    case TensorKind::KvGenerated: return "KV_GENERATED";
    case TensorKind::Activation: return "ACTIVATION";
    default: return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// Minimal YAML reader (same flat subset used elsewhere in h3-components)
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
  if (!in) throw std::runtime_error("ModelConfig: cannot open config file: " + path);
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
      throw std::runtime_error("ModelConfig: " + path + ":" + std::to_string(lineno) +
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

uint64_t parse_u64(const std::string& key, const std::string& raw) {
  std::string s;
  for (char c : raw) if (c != '_' && c != '\'') s.push_back(c);
  s = trim(s);
  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    base = 16;
    s = s.substr(2);
  }
  size_t consumed = 0;
  uint64_t v = 0;
  try {
    v = std::stoull(s, &consumed, base);
  } catch (const std::exception&) {
    throw std::runtime_error("ModelConfig: cannot parse integer for " + key + ": '" + raw + "'");
  }
  if (consumed != s.size()) {
    throw std::runtime_error("ModelConfig: trailing garbage for " + key + ": '" + raw + "'");
  }
  return v;
}

double parse_double(const std::string& key, const std::string& raw) {
  std::string s;
  for (char c : raw) if (c != '_') s.push_back(c);
  try {
    return std::stod(trim(s));
  } catch (const std::exception&) {
    throw std::runtime_error("ModelConfig: cannot parse number for " + key + ": '" + raw + "'");
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

ModelConfig ModelConfig::from_yaml(const std::string& path) {
  const auto kv = read_flat_yaml(path);
  ModelConfig c;

  static const char* known[] = {
      "model.name", "model.num_layers", "model.hidden_dim",
      "model.num_attention_heads", "model.num_kv_heads",
      "model.intermediate_dim", "model.vocab_size", "model.head_dim",
      "model.weight_dtype_bytes",
      "kv_cache.num_cache_heads", "kv_cache.kv_dtype_bytes",
      "serving.input_seq_len", "serving.output_seq_len", "serving.batch_size",
      "serving.sequence_length", "serving.num_gpus",
      "serving.tensor_parallel_size",
      "hardware.gpu_clock_ghz", "hardware.aggregate_read_bandwidth_gbps",
      "hardware.hbm_capacity_bytes_per_gpu", "hardware.hbf_capacity_bytes_per_gpu",
      "address_map.hbm_base_addr", "address_map.hbf_base_addr",
      "address_map.tensor_alignment_bytes",
      "prefetch.lead_time_ns", "prefetch.safety_margin_ns",
      "prefetch.max_outstanding_hints",
  };
  for (const auto& e : kv) {
    bool ok = false;
    for (const char* k : known) {
      if (e.first == k) { ok = true; break; }
    }
    if (!ok) throw std::runtime_error("ModelConfig: unknown key '" + e.first + "' in " + path);
  }

  auto has = [&](const char* k) { return kv.find(k) != kv.end(); };
  auto U = [&](const char* k) { return parse_u64(k, kv.at(k)); };
  auto D = [&](const char* k) { return parse_double(k, kv.at(k)); };

  if (has("model.name")) c.name = kv.at("model.name");
  if (has("model.num_layers")) c.num_layers = U("model.num_layers");
  if (has("model.hidden_dim")) c.hidden_dim = U("model.hidden_dim");
  if (has("model.num_attention_heads")) c.num_attention_heads = U("model.num_attention_heads");
  if (has("model.num_kv_heads")) c.num_kv_heads = U("model.num_kv_heads");
  if (has("model.intermediate_dim")) c.intermediate_dim = U("model.intermediate_dim");
  if (has("model.vocab_size")) c.vocab_size = U("model.vocab_size");
  if (has("model.head_dim")) c.head_dim = U("model.head_dim");
  if (has("model.weight_dtype_bytes")) c.weight_dtype_bytes = U("model.weight_dtype_bytes");

  if (has("kv_cache.num_cache_heads")) c.num_cache_heads = U("kv_cache.num_cache_heads");
  if (has("kv_cache.kv_dtype_bytes")) c.kv_dtype_bytes = U("kv_cache.kv_dtype_bytes");

  if (has("serving.input_seq_len")) c.input_seq_len = U("serving.input_seq_len");
  if (has("serving.output_seq_len")) c.output_seq_len = U("serving.output_seq_len");
  if (has("serving.batch_size")) c.batch_size = U("serving.batch_size");
  if (has("serving.sequence_length")) c.sequence_length = U("serving.sequence_length");
  if (has("serving.num_gpus")) c.num_gpus = U("serving.num_gpus");
  if (has("serving.tensor_parallel_size")) c.tensor_parallel_size = U("serving.tensor_parallel_size");

  if (has("hardware.gpu_clock_ghz")) c.gpu_clock_ghz = D("hardware.gpu_clock_ghz");
  if (has("hardware.aggregate_read_bandwidth_gbps"))
    c.aggregate_read_bandwidth_gbps = D("hardware.aggregate_read_bandwidth_gbps");
  if (has("hardware.hbm_capacity_bytes_per_gpu"))
    c.hbm_capacity_bytes_per_gpu = U("hardware.hbm_capacity_bytes_per_gpu");
  if (has("hardware.hbf_capacity_bytes_per_gpu"))
    c.hbf_capacity_bytes_per_gpu = U("hardware.hbf_capacity_bytes_per_gpu");

  if (has("address_map.hbm_base_addr")) c.hbm_base_addr = U("address_map.hbm_base_addr");
  if (has("address_map.hbf_base_addr")) c.hbf_base_addr = U("address_map.hbf_base_addr");
  if (has("address_map.tensor_alignment_bytes"))
    c.tensor_alignment_bytes = U("address_map.tensor_alignment_bytes");

  if (has("prefetch.lead_time_ns")) c.lead_time_ns = D("prefetch.lead_time_ns");
  if (has("prefetch.safety_margin_ns")) c.safety_margin_ns = D("prefetch.safety_margin_ns");
  if (has("prefetch.max_outstanding_hints"))
    c.max_outstanding_hints = U("prefetch.max_outstanding_hints");

  c.validate();
  return c;
}

void ModelConfig::validate() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("ModelConfig: " + m); };
  if (num_layers == 0) fail("num_layers must be > 0");
  if (hidden_dim == 0) fail("hidden_dim must be > 0");
  if (num_attention_heads == 0) fail("num_attention_heads must be > 0");
  if (head_dim * num_attention_heads != hidden_dim) {
    fail("head_dim * num_attention_heads must equal hidden_dim");
  }
  if (weight_dtype_bytes == 0 || kv_dtype_bytes == 0) fail("dtype sizes must be > 0");
  if (tensor_parallel_size == 0) fail("tensor_parallel_size must be > 0");
  if (num_gpus == 0) fail("num_gpus must be > 0");
  if (aggregate_read_bandwidth_gbps <= 0.0) fail("aggregate_read_bandwidth_gbps must be > 0");
  if (sequence_length < input_seq_len + output_seq_len) {
    fail("sequence_length must be at least input_seq_len + output_seq_len");
  }
  if (tensor_alignment_bytes == 0) fail("tensor_alignment_bytes must be > 0");
}

// ---- Analytic size model --------------------------------------------------
// Per layer (all values in bytes, whole model before TP sharding):
//   q_proj  hidden x (heads * head_dim)
//   k_proj  hidden x (kv_heads * head_dim)
//   v_proj  hidden x (kv_heads * head_dim)
//   o_proj  (heads * head_dim) x hidden
//   gate / up / down   hidden x intermediate  (x3)
//   2 RMSNorm scales   2 x hidden

uint64_t ModelConfig::weight_bytes_per_layer() const {
  const uint64_t q = hidden_dim * num_attention_heads * head_dim;
  const uint64_t k = hidden_dim * num_kv_heads * head_dim;
  const uint64_t v = k;
  const uint64_t o = num_attention_heads * head_dim * hidden_dim;
  const uint64_t mlp = 3ULL * hidden_dim * intermediate_dim;
  const uint64_t norm = 2ULL * hidden_dim;
  return (q + k + v + o + mlp + norm) * weight_dtype_bytes;
}

uint64_t ModelConfig::weight_bytes_all_layers() const {
  return weight_bytes_per_layer() * num_layers;
}

uint64_t ModelConfig::embedding_bytes() const {
  // token embedding + lm_head (untied)
  return 2ULL * vocab_size * hidden_dim * weight_dtype_bytes;
}

uint64_t ModelConfig::total_weight_bytes() const {
  return weight_bytes_all_layers() + embedding_bytes();
}

uint64_t ModelConfig::total_weight_bytes_per_gpu() const {
  return total_weight_bytes() / tensor_parallel_size;
}

uint64_t ModelConfig::total_parameters() const {
  return total_weight_bytes() / weight_dtype_bytes;
}

uint64_t ModelConfig::kv_bytes_per_token() const {
  // 2 (K and V) x layers x cache_heads x head_dim x dtype
  return 2ULL * num_layers * num_cache_heads * head_dim * kv_dtype_bytes;
}

uint64_t ModelConfig::generated_kv_tokens() const {
  return (input_seq_len + output_seq_len) * batch_size;
}

uint64_t ModelConfig::shared_kv_tokens() const {
  const uint64_t gen = generated_kv_tokens();
  return sequence_length > gen ? sequence_length - gen : 0;
}

uint64_t ModelConfig::generated_kv_bytes_per_gpu() const {
  return kv_bytes_per_token() * generated_kv_tokens() / num_gpus;
}

uint64_t ModelConfig::shared_kv_bytes_per_gpu() const {
  return kv_bytes_per_token() * shared_kv_tokens() / num_gpus;
}

uint64_t ModelConfig::activation_bytes_per_gpu() const {
  // Transient activations for one layer, double-buffered: a small residency
  // compared with weights and KV, but it must live in writable HBM.
  const uint64_t per_token = (hidden_dim + intermediate_dim) * 2ULL * kv_dtype_bytes;
  return per_token * batch_size * 2ULL;
}

double ModelConfig::hbf_occupancy() const {
  const uint64_t used = total_weight_bytes_per_gpu() + shared_kv_bytes_per_gpu();
  return hbf_capacity_bytes_per_gpu
             ? static_cast<double>(used) / static_cast<double>(hbf_capacity_bytes_per_gpu)
             : 0.0;
}

double ModelConfig::hbm_occupancy() const {
  const uint64_t used = generated_kv_bytes_per_gpu() + activation_bytes_per_gpu();
  return hbm_capacity_bytes_per_gpu
             ? static_cast<double>(used) / static_cast<double>(hbm_capacity_bytes_per_gpu)
             : 0.0;
}

// ---------------------------------------------------------------------------
// LLMPrefetchScheduler
// ---------------------------------------------------------------------------

LLMPrefetchScheduler::LLMPrefetchScheduler(const ModelConfig& config)
    : m_config(config) {
  m_config.validate();
  build_schedule();
}

uint64_t LLMPrefetchScheduler::align_up(uint64_t v) const {
  const uint64_t a = m_config.tensor_alignment_bytes;
  return ((v + a - 1) / a) * a;
}

void LLMPrefetchScheduler::add_tensor(const std::string& name, int layer,
                                      TensorKind kind, TensorRegion region,
                                      uint64_t size_bytes) {
  TensorDesc t;
  t.id = m_tensors.size();
  t.name = name;
  t.layer = layer;
  t.kind = kind;
  t.region = region;
  t.size_bytes = size_bytes;

  if (region == TensorRegion::Hbf) {
    t.local_addr = m_hbf_cursor;
    t.global_addr = m_config.hbf_base_addr + t.local_addr;
    m_hbf_cursor = align_up(m_hbf_cursor + size_bytes);
    t.read_only = true;
    ++m_stats.tensors_hbf;
    m_stats.bytes_hbf += size_bytes;
  } else {
    t.local_addr = m_hbm_cursor;
    t.global_addr = m_config.hbm_base_addr + t.local_addr;
    m_hbm_cursor = align_up(m_hbm_cursor + size_bytes);
    t.read_only = false;
    ++m_stats.tensors_hbm;
    m_stats.bytes_hbm += size_bytes;
  }
  m_tensors.push_back(std::move(t));
}

void LLMPrefetchScheduler::build_schedule() {
  m_tensors.clear();
  m_stats = SchedulerStats();
  m_hbf_cursor = 0;
  m_hbm_cursor = 0;
  m_hint_order.clear();
  m_next_hint = 0;

  const uint64_t tp = m_config.tensor_parallel_size;
  const uint64_t dt = m_config.weight_dtype_bytes;
  const uint64_t H = m_config.hidden_dim;
  const uint64_t A = m_config.num_attention_heads;
  const uint64_t KVh = m_config.num_kv_heads;
  const uint64_t hd = m_config.head_dim;
  const uint64_t I = m_config.intermediate_dim;

  // ---- HBF: weights, laid out layer by layer in execution order ----------
  // Sequential layout matters: it is what makes the LHB's streaming prefetch
  // effective. Shuffling this layout would degrade the hit rate, which is
  // itself a worthwhile experiment.
  add_tensor("embed_tokens", -1, TensorKind::WeightEmbed, TensorRegion::Hbf,
             m_config.vocab_size * H * dt / tp);

  for (uint64_t l = 0; l < m_config.num_layers; ++l) {
    const int li = static_cast<int>(l);
    add_tensor("layer" + std::to_string(l) + ".input_norm", li,
               TensorKind::WeightNorm, TensorRegion::Hbf, H * dt);
    add_tensor("layer" + std::to_string(l) + ".q_proj", li,
               TensorKind::WeightQkv, TensorRegion::Hbf, H * A * hd * dt / tp);
    add_tensor("layer" + std::to_string(l) + ".k_proj", li,
               TensorKind::WeightQkv, TensorRegion::Hbf, H * KVh * hd * dt / tp);
    add_tensor("layer" + std::to_string(l) + ".v_proj", li,
               TensorKind::WeightQkv, TensorRegion::Hbf, H * KVh * hd * dt / tp);
    add_tensor("layer" + std::to_string(l) + ".o_proj", li,
               TensorKind::WeightO, TensorRegion::Hbf, A * hd * H * dt / tp);
    add_tensor("layer" + std::to_string(l) + ".post_norm", li,
               TensorKind::WeightNorm, TensorRegion::Hbf, H * dt);
    add_tensor("layer" + std::to_string(l) + ".gate_proj", li,
               TensorKind::WeightMlp, TensorRegion::Hbf, H * I * dt / tp);
    add_tensor("layer" + std::to_string(l) + ".up_proj", li,
               TensorKind::WeightMlp, TensorRegion::Hbf, H * I * dt / tp);
    add_tensor("layer" + std::to_string(l) + ".down_proj", li,
               TensorKind::WeightMlp, TensorRegion::Hbf, I * H * dt / tp);
  }
  add_tensor("lm_head", -1, TensorKind::WeightEmbed, TensorRegion::Hbf,
             m_config.vocab_size * H * dt / tp);

  // ---- HBF: shared precomputed KV cache, one slice per layer -------------
  const uint64_t shared_total = m_config.shared_kv_bytes_per_gpu();
  const uint64_t shared_per_layer = shared_total / m_config.num_layers;
  for (uint64_t l = 0; l < m_config.num_layers; ++l) {
    add_tensor("layer" + std::to_string(l) + ".kv_shared", static_cast<int>(l),
               TensorKind::KvShared, TensorRegion::Hbf, shared_per_layer);
  }

  // ---- HBM: generated KV cache (written during inference) ----------------
  const uint64_t gen_total = m_config.generated_kv_bytes_per_gpu();
  const uint64_t gen_per_layer = gen_total / m_config.num_layers;
  for (uint64_t l = 0; l < m_config.num_layers; ++l) {
    add_tensor("layer" + std::to_string(l) + ".kv_generated", static_cast<int>(l),
               TensorKind::KvGenerated, TensorRegion::Hbm, gen_per_layer);
  }

  // ---- HBM: activations --------------------------------------------------
  add_tensor("activations", -1, TensorKind::Activation, TensorRegion::Hbm,
             m_config.activation_bytes_per_gpu());

  m_stats.tensors_total = m_tensors.size();

  // ---- Time schedule -----------------------------------------------------
  // A tensor is needed once everything before it has been streamed in. This is
  // a bandwidth-bound model: time advances as bytes/BW. It sets WHEN hints are
  // issued; authoritative execution timing still comes from Accel-Sim.
  //   bytes / (GB/s) = ns  ->  x1000 = ps
  const double bytes_per_ps = m_config.aggregate_read_bandwidth_gbps * 1e-3;
  const uint64_t lead_ps = m_config.lead_time_ps();

  uint64_t cursor_bytes = 0;
  for (auto& t : m_tensors) {
    // Activations and generated KV are produced, not streamed in; they do not
    // advance the read cursor and are never prefetched from HBF.
    const bool streamed = (t.region == TensorRegion::Hbf);
    t.needed_at_ps = static_cast<uint64_t>(static_cast<double>(cursor_bytes) / bytes_per_ps);
    if (streamed) cursor_bytes += t.size_bytes;

    if (t.needed_at_ps > lead_ps) {
      t.hint_at_ps = t.needed_at_ps - lead_ps;
    } else {
      t.hint_at_ps = 0;      // no room to hide tR at the very start of the pass
      if (streamed) ++m_stats.hints_late;
    }
  }
  m_forward_duration_ps =
      static_cast<uint64_t>(static_cast<double>(cursor_bytes) / bytes_per_ps);

  // Hint issue order (stable: ties keep layout order).
  for (size_t i = 0; i < m_tensors.size(); ++i) {
    if (m_tensors[i].region == TensorRegion::Hbf) m_hint_order.push_back(i);
  }
  std::stable_sort(m_hint_order.begin(), m_hint_order.end(),
                   [this](size_t a, size_t b) {
                     return m_tensors[a].hint_at_ps < m_tensors[b].hint_at_ps;
                   });
  m_stats.hints_pending = m_hint_order.size();
}

void LLMPrefetchScheduler::rewind() {
  m_next_hint = 0;
  m_stats.hints_issued = 0;
  m_stats.hints_pending = m_hint_order.size();
}

uint64_t LLMPrefetchScheduler::tick(uint64_t now_ps, LatencyHidingBuffer* lhb) {
  uint64_t issued = 0;
  while (m_next_hint < m_hint_order.size()) {
    const TensorDesc& t = m_tensors[m_hint_order[m_next_hint]];
    if (t.hint_at_ps > now_ps) break;
    if (lhb && lhb->pending_hints() >= m_config.max_outstanding_hints) break;

    if (lhb) {
      PrefetchHint h;
      h.tensor_id = t.id;
      h.hbf_addr = t.local_addr;      // LHB works in HBF-local space
      h.size_bytes = t.size_bytes;
      h.needed_at_ps = t.needed_at_ps;
      h.layer = t.layer;
      lhb->issue_hint(h);
    }
    ++m_next_hint;
    ++issued;
    ++m_stats.hints_issued;
    if (m_stats.hints_pending) --m_stats.hints_pending;
  }
  return issued;
}

int LLMPrefetchScheduler::tensor_at(uint64_t global_addr) const {
  for (size_t i = 0; i < m_tensors.size(); ++i) {
    const auto& t = m_tensors[i];
    if (global_addr >= t.global_addr && global_addr < t.global_addr + t.size_bytes) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static std::string human(uint64_t bytes) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2);
  if (bytes >= (1ULL << 40)) os << static_cast<double>(bytes) / (1ULL << 40) << " TiB";
  else if (bytes >= (1ULL << 30)) os << static_cast<double>(bytes) / (1ULL << 30) << " GiB";
  else if (bytes >= (1ULL << 20)) os << static_cast<double>(bytes) / (1ULL << 20) << " MiB";
  else if (bytes >= (1ULL << 10)) os << static_cast<double>(bytes) / (1ULL << 10) << " KiB";
  else os << bytes << " B";
  return os.str();
}

void LLMPrefetchScheduler::print_summary(std::ostream& os) const {
  const auto& c = m_config;
  os << std::dec << std::fixed;
  os << "LLM prefetch scheduler -- " << c.name << "\n";
  os << "  architecture\n";
  os << "    layers / hidden / heads / kv_heads = " << c.num_layers << " / "
     << c.hidden_dim << " / " << c.num_attention_heads << " / " << c.num_kv_heads << "\n";
  os << "    intermediate / vocab / head_dim    = " << c.intermediate_dim << " / "
     << c.vocab_size << " / " << c.head_dim << "\n";
  os << "    weight dtype                       = " << c.weight_dtype_bytes << " B (FP"
     << (c.weight_dtype_bytes * 8) << ")\n";
  os << "  parameters\n";
  os << "    per-layer weights                  = " << human(c.weight_bytes_per_layer()) << "\n";
  os << "    all layers                         = " << human(c.weight_bytes_all_layers()) << "\n";
  os << "    embeddings (embed + lm_head)       = " << human(c.embedding_bytes()) << "\n";
  os << "    TOTAL parameters                   = " << std::setprecision(1)
     << static_cast<double>(c.total_parameters()) / 1e9 << " B\n";
  os << "    weights per GPU (TP=" << c.tensor_parallel_size << ")             = "
     << human(c.total_weight_bytes_per_gpu()) << "\n";
  os << "  kv cache (" << c.num_cache_heads << " heads, FP"
     << (c.kv_dtype_bytes * 8) << ")\n";
  os << "    bytes per token                    = " << human(c.kv_bytes_per_token()) << "\n";
  os << "    sequence length                    = " << c.sequence_length << " tokens\n";
  os << "    generated (ISL+OSL) -> HBM         = " << c.generated_kv_tokens()
     << " tokens, " << human(c.generated_kv_bytes_per_gpu()) << "/GPU\n";
  os << "    shared precomputed  -> HBF         = " << c.shared_kv_tokens()
     << " tokens, " << human(c.shared_kv_bytes_per_gpu()) << "/GPU\n";
  os << "  capacity (" << c.num_gpus << " GPUs)\n";
  os << "    HBF used / capacity                = " << human(c.total_weight_bytes_per_gpu() +
                                                             c.shared_kv_bytes_per_gpu())
     << " / " << human(c.hbf_capacity_bytes_per_gpu) << "  ("
     << std::setprecision(1) << c.hbf_occupancy() * 100.0 << "%)\n";
  os << "    HBM used / capacity                = " << human(c.generated_kv_bytes_per_gpu() +
                                                             c.activation_bytes_per_gpu())
     << " / " << human(c.hbm_capacity_bytes_per_gpu) << "  ("
     << std::setprecision(2) << c.hbm_occupancy() * 100.0 << "%)\n";
  os << "  schedule\n";
  os << "    tensors (HBF / HBM)                = " << m_stats.tensors_hbf << " / "
     << m_stats.tensors_hbm << "\n";
  os << "    forward pass duration              = " << std::setprecision(1)
     << static_cast<double>(m_forward_duration_ps) / 1e6 << " us\n";
  os << "    hints with no room to hide tR      = " << m_stats.hints_late << "\n";
}

void LLMPrefetchScheduler::print_layout(std::ostream& os, int max_rows) const {
  os << std::left << std::setw(30) << "tensor" << std::setw(6) << "reg"
     << std::setw(16) << "kind" << std::right << std::setw(16) << "global addr"
     << std::setw(14) << "size" << "\n";
  os << std::string(82, '-') << "\n";
  int shown = 0;
  for (const auto& t : m_tensors) {
    if (max_rows > 0 && shown >= max_rows) {
      os << "  ... " << (m_tensors.size() - static_cast<size_t>(shown))
         << " more tensors\n";
      break;
    }
    os << std::left << std::setw(30) << t.name << std::setw(6) << to_string(t.region)
       << std::setw(16) << to_string(t.kind) << std::right << "0x" << std::hex
       << std::setw(14) << t.global_addr << std::dec << std::setw(14)
       << human(t.size_bytes) << "\n";
    ++shown;
  }
}

void LLMPrefetchScheduler::print_schedule(std::ostream& os, int max_rows) const {
  os << std::left << std::setw(30) << "tensor" << std::right << std::setw(16)
     << "hint_at (us)" << std::setw(16) << "needed_at (us)" << std::setw(14)
     << "lead (us)" << "\n";
  os << std::string(76, '-') << "\n";
  int shown = 0;
  for (size_t idx : m_hint_order) {
    if (max_rows > 0 && shown >= max_rows) {
      os << "  ... " << (m_hint_order.size() - static_cast<size_t>(shown))
         << " more hints\n";
      break;
    }
    const auto& t = m_tensors[idx];
    os << std::left << std::setw(30) << t.name << std::right << std::fixed
       << std::setprecision(2) << std::setw(16)
       << static_cast<double>(t.hint_at_ps) / 1e6 << std::setw(16)
       << static_cast<double>(t.needed_at_ps) / 1e6 << std::setw(14)
       << static_cast<double>(t.needed_at_ps - t.hint_at_ps) / 1e6 << "\n";
    ++shown;
  }
}

}  // namespace h3
