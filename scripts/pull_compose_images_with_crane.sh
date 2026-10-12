#!/usr/bin/env bash
set -euo pipefail

compose_file="${1:-infra/docker-compose.yml}"
crane_bin="${CRANE_BIN:-crane}"
docker_bin="${DOCKER_BIN:-docker}"

if [[ -n "${CRANE_PLATFORM:-}" ]]; then
  crane_platform="$CRANE_PLATFORM"
else
  case "$(uname -m)" in
    x86_64) crane_platform="linux/amd64" ;;
    aarch64 | arm64) crane_platform="linux/arm64" ;;
    *)
      echo "error: unsupported host architecture $(uname -m); set CRANE_PLATFORM explicitly" >&2
      exit 1
      ;;
  esac
fi

if ! command -v "$crane_bin" >/dev/null 2>&1; then
  echo "error: crane is not installed or not in PATH" >&2
  exit 1
fi

if ! command -v "$docker_bin" >/dev/null 2>&1; then
  echo "error: docker is not installed or not in PATH" >&2
  exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf -- "$temp_dir"' EXIT

mapfile -t images < <("$docker_bin" compose -f "$compose_file" config --images)
if ((${#images[@]} == 0)); then
  echo "error: no images found in $compose_file" >&2
  exit 1
fi

for index in "${!images[@]}"; do
  image="${images[$index]}"
  archive="$temp_dir/image-$index.tar"

  if [[ "${CRANE_SKIP_EXISTING:-0}" == "1" ]] && "$docker_bin" image inspect "$image" >/dev/null 2>&1; then
    echo "Using existing local image $image"
    continue
  fi

  echo "Pulling $image for $crane_platform with crane..."
  "$crane_bin" pull --platform "$crane_platform" --format tarball "$image" "$archive"
  "$docker_bin" load --input "$archive"
done
