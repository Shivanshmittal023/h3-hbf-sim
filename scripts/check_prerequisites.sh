#!/usr/bin/env bash
# =============================================================================
#  check_prerequisites.sh -- verify a machine can build the H3 simulator
# =============================================================================
#  Run this BEFORE building. It checks every tool, version, library and
#  resource the build actually needs, and prints exactly what to install for
#  anything missing.
#
#  Requirements are taken from the real build files, not from memory:
#    accel-sim-framework/gpu-simulator/CMakeLists.txt   cmake >= 3.17, C++17
#    ramulator2/CMakeLists.txt                          cmake >= 3.14, C++20
#    Dockerfile.h3sim                                   the apt package list
#    gpu-simulator/version_detection.mk                 invokes nvcc
#
#  Exit code 0 = ready to build, 1 = something is missing.
#
#  USAGE
#      bash scripts/check_prerequisites.sh
#      bash scripts/check_prerequisites.sh --tracing   # also check GPU/NVBit
# =============================================================================
set -uo pipefail

CHECK_TRACING=0
[ "${1:-}" = "--tracing" ] && CHECK_TRACING=1

PASS=0; WARN=0; FAIL=0
ok()   { printf "  \033[32m[ ok ]\033[0m %-34s %s\n" "$1" "${2:-}"; PASS=$((PASS+1)); }
warn() { printf "  \033[33m[warn]\033[0m %-34s %s\n" "$1" "${2:-}"; WARN=$((WARN+1)); }
bad()  { printf "  \033[31m[FAIL]\033[0m %-34s %s\n" "$1" "${2:-}"; FAIL=$((FAIL+1)); }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

# Compare dotted versions: vge 3.20 3.17 -> true
vge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

hdr "System"
if [ -r /etc/os-release ]; then
  . /etc/os-release; ok "OS" "$PRETTY_NAME"
else
  warn "OS" "cannot read /etc/os-release"
fi
ok "kernel / arch" "$(uname -r) $(uname -m)"
[ "$(uname -m)" = "x86_64" ] || bad "architecture" "x86_64 required (CUDA toolchain)"

CORES=$(nproc 2>/dev/null || echo 1)
ok "CPU cores" "$CORES  (build with -j$CORES)"

MEMKB=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
MEMGB=$((MEMKB / 1024 / 1024))
if [ "$MEMGB" -ge 32 ]; then ok "RAM" "${MEMGB} GB"
elif [ "$MEMGB" -ge 16 ]; then warn "RAM" "${MEMGB} GB (16+ works; 32+ better for large traces)"
else bad "RAM" "${MEMGB} GB -- the linker needs ~8 GB, traces need more"; fi

AVAIL=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
AVAIL=${AVAIL:-0}
if [ "$AVAIL" -ge 100 ]; then ok "free disk" "${AVAIL} GB"
elif [ "$AVAIL" -ge 30 ]; then warn "free disk" "${AVAIL} GB (fine to build; traces need much more)"
else bad "free disk" "${AVAIL} GB -- need 30+ to build"; fi

hdr "Compiler and build tools"
if command -v g++ >/dev/null; then
  GV=$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)
  # Ramulator is C++20 -> GCC 10 is the practical floor; 11+ preferred.
  if vge "$GV" 11; then ok "g++" "$GV"
  elif vge "$GV" 10; then warn "g++" "$GV (C++20 support is partial before 11)"
  else bad "g++" "$GV -- need >= 10 for Ramulator's C++20"; fi
else bad "g++" "not found"; fi

if command -v cmake >/dev/null; then
  CV=$(cmake --version | head -1 | awk '{print $3}')
  vge "$CV" 3.17 && ok "cmake" "$CV" || bad "cmake" "$CV -- need >= 3.17"
else bad "cmake" "not found"; fi

for t in make git patch bison flex makedepend pkg-config; do
  command -v $t >/dev/null && ok "$t" "$(command -v $t)" || {
    [ "$t" = makedepend ] && warn "$t" "missing (xutils-dev; only the legacy Makefile path)" \
                          || bad "$t" "not found"; }
done

hdr "CUDA toolkit"
CUDA_HOME_GUESS="${CUDA_INSTALL_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
if [ -x "$CUDA_HOME_GUESS/bin/nvcc" ]; then
  NV=$("$CUDA_HOME_GUESS/bin/nvcc" --version | awk '/release/{print $6}' | tr -d 'V,')
  NVMAJ=${NV%%.*}
  # A NEWER major version is not automatically compatible. GPGPU-Sim's libcuda
  # includes internal CUDA headers (host_defines.h) that NVIDIA has said it will
  # remove, so 13.x may simply fail to compile. Only 12.x is known-good.
  if [ "$NVMAJ" = "12" ] && vge "$NV" 12.8; then
    ok "nvcc" "$NV at $CUDA_HOME_GUESS"
  elif [ "$NVMAJ" = "12" ]; then
    warn "nvcc" "$NV -- project is built and tested on 12.8"
  elif [ "$NVMAJ" -gt 12 ] 2>/dev/null; then
    bad "nvcc" "$NV -- NEWER than the tested 12.8. GPGPU-Sim includes internal
                        CUDA headers that newer toolkits remove. Install CUDA
                        12.8 alongside and point CUDA_INSTALL_PATH at it."
  else
    bad "nvcc" "$NV -- too old, need 12.8"
  fi
  # The specific header that breaks first if the toolkit is too new.
  if [ -f "$CUDA_HOME_GUESS/targets/x86_64-linux/include/host_defines.h" ] \
     || [ -f "$CUDA_HOME_GUESS/include/host_defines.h" ]; then
    ok "  host_defines.h" "present (GPGPU-Sim's libcuda needs it)"
  else
    bad "  host_defines.h" "MISSING -- libcuda/cuda_runtime_api.cc will not compile"
  fi
  [ -n "${CUDA_INSTALL_PATH:-}" ] && ok "CUDA_INSTALL_PATH" "$CUDA_INSTALL_PATH" \
    || warn "CUDA_INSTALL_PATH" "unset -- export CUDA_INSTALL_PATH=$CUDA_HOME_GUESS"
else
  bad "nvcc" "not found at $CUDA_HOME_GUESS (the build reads nvcc --version)"
fi

hdr "Python"
if command -v python3 >/dev/null; then
  PV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  vge "$PV" 3.8 && ok "python3" "$PV" || bad "python3" "$PV -- need >= 3.8"
  python3 -c "import yaml" 2>/dev/null && ok "  pyyaml" "present" \
    || warn "  pyyaml" "missing -- pip3 install pyyaml (needed by the tools)"
  # Headers may belong to a different interpreter than the active one: a conda
  # env can report 3.14 while python3-dev installed headers for the system 3.12.
  # Only needed if Ramulator's Python bindings are enabled, which we build OFF.
  if ls /usr/include/python3*/Python.h >/dev/null 2>&1; then
    ok "  Python.h" "$(ls /usr/include/python3*/Python.h | head -1)"
  else
    warn "  Python.h" "not found (only needed for Ramulator's Python bindings,
                        which this build disables)"
  fi
  if [ -n "${CONDA_PREFIX:-}" ]; then
    warn "  conda env active" "$CONDA_PREFIX
                        Run 'conda deactivate' before building: conda's
                        libstdc++ can shadow the system one and break linking."
  fi
else bad "python3" "not found"; fi

hdr "Libraries (headers must be present, not just the runtime)"
check_hdr() {  # name  header-path  package
  if ls $2 >/dev/null 2>&1; then ok "$1" "$(ls $2 | head -1)"
  else bad "$1" "missing -- apt install $3"; fi
}
check_hdr "zlib"     "/usr/include/zlib.h"                       "zlib1g-dev"
check_hdr "zstd"     "/usr/include/zstd.h"                       "libzstd-dev"
check_hdr "openssl"  "/usr/include/openssl/ssl.h"                "libssl-dev"
check_hdr "libxml2"  "/usr/include/libxml2/libxml/parser.h"      "libxml2-dev"
check_hdr "boost"    "/usr/include/boost/version.hpp"            "libboost-all-dev"
check_hdr "GL/GLU"   "/usr/include/GL/glu.h"                     "libglu1-mesa-dev"

hdr "Network (CMake fetches Ramulator's dependencies at configure time)"
if curl -sI --max-time 12 https://github.com >/dev/null 2>&1; then
  ok "github.com" "reachable (yaml-cpp, fmt are fetched from there)"
else
  bad "github.com" "unreachable -- Ramulator's FetchContent will fail"
fi
curl -sI --max-time 12 https://engineering.purdue.edu >/dev/null 2>&1 \
  && ok "trace server" "reachable" || warn "trace server" "unreachable (only needed for downloads)"

if [ "$CHECK_TRACING" = "1" ]; then
  hdr "GPU and tracing (only needed to RECORD traces, not to simulate)"
  if command -v nvidia-smi >/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU" "$GPU (compute capability $CC)"
    case "$CC" in
      7.*|8.*|9.*) ok "  Accel-Sim SASS support" "compute $CC is decoded (Volta..Hopper)" ;;
      10.*|12.*)   warn "  Accel-Sim SASS support" \
                       "compute $CC is Blackwell; ISA_Def/ stops at Hopper, so traces
                        recorded HERE cannot be decoded. Simulation is unaffected --
                        record traces on a Volta/Ampere/Hopper GPU instead." ;;
      *)           warn "  Accel-Sim SASS support" "compute $CC not recognised" ;;
    esac
  else
    warn "nvidia-smi" "no NVIDIA GPU -- simulation still works, tracing does not"
  fi
fi

hdr "Summary"
printf "  passed %d, warnings %d, failures %d\n" "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo
  echo "  Ready to build. On Debian/Ubuntu, anything flagged above installs with:"
  echo "    sudo apt-get install -y build-essential g++ bison flex xutils-dev \\"
  echo "         zlib1g-dev libzstd-dev libssl-dev libxml2-dev libboost-all-dev \\"
  echo "         libglu1-mesa-dev freeglut3-dev cmake ninja-build git \\"
  echo "         python3 python3-dev python3-pip"
  exit 0
else
  echo
  echo "  Fix the FAIL items above before building. On Debian/Ubuntu:"
  echo "    sudo apt-get install -y build-essential g++ bison flex xutils-dev \\"
  echo "         zlib1g-dev libzstd-dev libssl-dev libxml2-dev libboost-all-dev \\"
  echo "         libglu1-mesa-dev freeglut3-dev cmake ninja-build git \\"
  echo "         python3 python3-dev python3-pip"
  echo "  CUDA 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive"
  exit 1
fi
