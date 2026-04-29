#!/usr/bin/env bash
# agri-gs-slam setup wizard
#
# End-to-end environment bootstrap:
#   1) Sanity-check host prerequisites (docker, compose, NVIDIA toolkit)
#   2) Build the Docker image (docker/docker-compose.yml)
#   3) Start the container
#   4) Compile the C++ modules inside the container (odometry + scancontext, Release)
#   5) Drop into a ready-to-run shell (pipeline.py is one command away)
#
# Usage:
#   scripts/setup.sh              # run all steps (also fetches the HF demo dataset if missing)
#   scripts/setup.sh --no-shell   # skip the final interactive shell
#   scripts/setup.sh --rebuild    # force docker image rebuild
#   scripts/setup.sh --clean-cpp  # wipe src/*/build before recompiling

set -euo pipefail

# ---------- styling ----------
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; RST=$'\e[0m'
else
    BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; BLU=""; RST=""
fi
step()  { echo; echo "${BOLD}${BLU}==>${RST} ${BOLD}$*${RST}"; }
ok()    { echo "${GRN}  ✓${RST} $*"; }
warn()  { echo "${YLW}  !${RST} $*"; }
die()   { echo "${RED}  ✗${RST} $*" >&2; exit 1; }

# ---------- paths ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
COMPOSE_DIR="${ROOT_DIR}/docker"
SERVICE="agri-gs-slam"

# ---------- flags ----------
REBUILD=0
CLEAN_CPP=0
SKIP_SHELL=0
for arg in "$@"; do
    case "$arg" in
        --rebuild)   REBUILD=1 ;;
        --clean-cpp) CLEAN_CPP=1 ;;
        --no-shell)  SKIP_SHELL=1 ;;
        -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
        *) die "unknown flag: $arg" ;;
    esac
done

# ---------- 1) host prerequisites ----------
step "Checking host prerequisites"
command -v docker  >/dev/null || die "docker not found — install Docker Engine first"
docker compose version >/dev/null 2>&1 || die "'docker compose' plugin not found"
ok "docker: $(docker --version)"
ok "compose: $(docker compose version --short 2>/dev/null || docker compose version | head -1)"

if command -v nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found on host — GPU workloads will fail"
fi

# Probe NVIDIA container runtime (non-fatal: build doesn't need GPU, but compose up does)
if ! docker info 2>/dev/null | grep -qi nvidia; then
    warn "NVIDIA container runtime not detected in 'docker info' — install nvidia-container-toolkit if compose up fails"
fi

# Pass host UID/GID/USER into compose build args
export USER_ID="${USER_ID:-$(id -u)}"
export GROUP_ID="${GROUP_ID:-$(id -g)}"
export USER="${USER:-$(id -un)}"
ok "build args: USER=${USER} USER_ID=${USER_ID} GROUP_ID=${GROUP_ID}"

# ---------- 1b) ensure host-side volumes exist ----------
step "Preparing host-side volumes (data/ and results/)"
mkdir -p "${ROOT_DIR}/data" "${ROOT_DIR}/results"
ok "data: ${ROOT_DIR}/data"
ok "results: ${ROOT_DIR}/results"

# ---------- 2) docker image ----------
step "Building Docker image"
cd "${COMPOSE_DIR}"
if [[ $REBUILD -eq 1 ]]; then
    warn "forcing --no-cache rebuild"
    docker compose build --no-cache "${SERVICE}"
else
    docker compose build "${SERVICE}"
fi
ok "image ready: $(docker compose config --format json 2>/dev/null | grep -o '"image":"[^"]*"' | head -1 || echo mirkousuelli/agri-gs-slam:stable)"

# ---------- 3) container up ----------
step "Starting container"
docker compose up -d "${SERVICE}"
# Wait a moment for the container to be healthy
for _ in {1..10}; do
    state=$(docker inspect -f '{{.State.Status}}' agri-gs-slam-container 2>/dev/null || echo "missing")
    [[ "$state" == "running" ]] && break
    sleep 1
done
[[ "$state" == "running" ]] || die "container failed to start (state=$state) — check 'docker compose logs ${SERVICE}'"
ok "container running"

# ---------- helper: exec inside container ----------
dexec() { docker compose exec -T "${SERVICE}" bash -lc "$*"; }

# ---------- 4) C++ modules ----------
build_cpp_module() {
    local mod="$1"
    step "Compiling ${mod} (Release)"
    if [[ $CLEAN_CPP -eq 1 ]]; then
        dexec "rm -rf /agri_gs_slam/src/${mod}/build"
    fi
    # Skip rebuild if a previous build is already in place (install_manifest.txt
    # is written by `cmake --install`, so it's a reliable "fully built" marker).
    # Force a rebuild with --clean-cpp.
    if [[ $CLEAN_CPP -eq 0 ]] && dexec "test -f /agri_gs_slam/src/${mod}/build/install_manifest.txt" 2>/dev/null; then
        ok "${mod} already built — skipping (use --clean-cpp to force rebuild)"
        return 0
    fi
    dexec "set -e; \
           cd /agri_gs_slam/src/${mod} && \
           mkdir -p build && cd build && \
           cmake -DCMAKE_BUILD_TYPE=Release .. && \
           cmake --build . -j\"\$(nproc)\" && \
           sudo cmake --install ."
    ok "${mod} compiled"
}
build_cpp_module odometry
build_cpp_module scancontext

# ---------- 4b) fetch HF demo dataset (skipped if already present) ----------
DEMO_HOST="${ROOT_DIR}/data/agri-gs-slam-dataset-demo"
step "Fetching demo dataset (foxmirko/agri-gs-slam-dataset-demo)"
if [[ -d "${DEMO_HOST}" && -n "$(ls -A "${DEMO_HOST}" 2>/dev/null || true)" ]]; then
    ok "demo dataset already present at ${DEMO_HOST} — skipping download (delete the folder to force re-download)"
else
    # Install huggingface_hub on the fly and download into the mounted host volume
    dexec "python3 -c 'import huggingface_hub' 2>/dev/null || pip install --quiet huggingface_hub"
    dexec "python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='foxmirko/agri-gs-slam-dataset-demo',
    repo_type='dataset',
    local_dir='/agri_gs_slam/data/agri-gs-slam-dataset-demo',
)
print('demo dataset ready at /agri_gs_slam/data/agri-gs-slam-dataset-demo')
PY"
    ok "demo dataset downloaded to ${DEMO_HOST}"
fi

# ---------- 4c) extract PLY archives (runs whenever the dataset is present) ----------
if [[ -d "${DEMO_HOST}" && -n "$(ls -A "${DEMO_HOST}" 2>/dev/null || true)" ]]; then
    step "Extracting PLY archives and removing zip files"
    # Use the dataset's own extract_ply.py to recursively unzip every point-cloud
    # archive into its parent folder, then remove the .zip files (the script keeps
    # them by design). If no .zip files remain, extraction is skipped — but we
    # still --verify so a partially-installed dataset gets caught.
    dexec "set -e; \
           DATA=/agri_gs_slam/data/agri-gs-slam-dataset-demo; \
           mapfile -t zips < <(find \"\$DATA\" -type f -name '*.zip'); \
           if [ \${#zips[@]} -eq 0 ]; then \
               echo 'no .zip archives left — already extracted'; \
           else \
               for z in \"\${zips[@]}\"; do \
                   d=\$(dirname \"\$z\"); \
                   echo \"--- unzip \$z\"; \
                   if unzip -oq \"\$z\" -d \"\$d\" && [ -n \"\$(find \"\$d\" -maxdepth 4 -name '*.ply' -print -quit)\" ]; then \
                       rm -f \"\$z\" && echo \"  removed \$(basename \"\$z\")\"; \
                   else \
                       echo \"  WARNING: extraction of \$(basename \"\$z\") produced no .ply — keeping the archive\"; \
                   fi; \
               done; \
           fi; \
           echo '--- extract_ply.py --verify ---'; \
           python3 \"\$DATA/extract_ply.py\" --base-dir \"\$DATA\" --verify || true"
    ok "archives extracted and removed"
    ok "config/default.yaml dataloader.path already points there"
fi

# ---------- 5) ready ----------
step "Environment ready"
cat <<EOF
${GRN}Everything is set up.${RST}

Run the pipeline (inside the container):
  ${BOLD}docker compose -f ${COMPOSE_DIR}/docker-compose.yml exec ${SERVICE} bash${RST}
  ${DIM}# then:${RST}
  ${BOLD}cd /agri_gs_slam/src && python3 pipeline.py --gs-slam${RST}

Available modalities: --odom | --slam | --gs-odom | --gs-slam   (add --gs-viewer to enable viewer)
Stop the container:   ${BOLD}docker compose -f ${COMPOSE_DIR}/docker-compose.yml down${RST}
EOF

if [[ $SKIP_SHELL -eq 0 ]]; then
    step "Opening interactive shell inside the container"
    exec docker compose -f "${COMPOSE_DIR}/docker-compose.yml" exec "${SERVICE}" bash
fi
