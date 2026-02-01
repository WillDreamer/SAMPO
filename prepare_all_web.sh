#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'


log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
trap 'die "Error is in Line $LINENO (exit=$?)。"' ERR

as_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    sudo -H bash -lc "$*"
  else
    bash -lc "$*"
  fi
}


log "Run setup game environment "
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
if [[ -z "${CONDA_BASE}" ]]; then
  for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/anaconda3"; do
    [[ -d "$p" ]] && CONDA_BASE="$p" && break
  done
fi
if [[ -z "${CONDA_BASE}" ]]; then
  echo "Conda not found, please confirm it is installed (miniconda or anaconda)." >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"


log "运行 setup_webshop.sh（若存在）"
cd ./agent_system/environments/env_package/webshop/webshop && bash setup.sh || warn "未找到 setup webshop，跳过"

# 再次确保 conda 激活（有些脚本可能改变了环境）
if command -v conda >/dev/null 2>&1; then
  conda activate agentrl_web || true
fi


log "校验登录状态（whoami）"
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || die "whoami 失败"
else
  huggingface-cli whoami || die "whoami 失败"
fi

log "全部完成 🎉"
