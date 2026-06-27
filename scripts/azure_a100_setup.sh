#!/usr/bin/env bash
# Provision a single A100 80GB spot VM on Azure for the self-distillation campaign,
# with a budget guard. Run the steps interactively (review each before continuing) —
# this creates billable cloud resources.
#
# Prereqs:
#   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#   az login --use-device-code
#
# Then:  bash scripts/azure_a100_setup.sh        (or copy/paste blocks as you go)
set -euo pipefail

# ----- knobs (override via env) -----------------------------------------------
RG="${RG:-pe-rg}"
LOCATION="${LOCATION:-eastus}"          # try westus3 / southcentralus if A100 capacity is short
VM="${VM:-pe-a100}"
SIZE="${SIZE:-Standard_NC24ads_A100_v4}"  # 1x A100 80GB
IMAGE="${IMAGE:-Canonical:ubuntu-24_04-lts:server:latest}"
ADMIN="${ADMIN:-azureuser}"
BUDGET="${BUDGET:-80}"                   # alert threshold in USD (credits are $100)
DISK_GB="${DISK_GB:-128}"               # OS disk; 14B weights + caches need room

echo "subscription:"; az account show --query "{name:name, id:id, state:state}" -o table

# ----- 1. quota check (new subscriptions start at 0 GPU vCPUs) -----------------
echo "== A100 (NCADSA100v4) quota in $LOCATION =="
az vm list-usage --location "$LOCATION" -o table | grep -i "NCADSA100v4" || \
  echo "  (family not listed — request a quota increase in Portal -> Quotas before proceeding)"

# ----- 2. budget guard FIRST --------------------------------------------------
SUB_ID="$(az account show --query id -o tsv)"
az consumption budget create \
  --budget-name pe-budget --amount "$BUDGET" --category Cost --time-grain Monthly \
  --scope "/subscriptions/$SUB_ID" 2>/dev/null || \
  echo "  (budget create skipped/unsupported on this offer — set an alert in the Portal)"

# ----- 3. resource group ------------------------------------------------------
az group create --name "$RG" --location "$LOCATION" -o table

# ----- 4. spot A100 VM --------------------------------------------------------
# Spot + Deallocate: evicted when capacity is reclaimed; you keep the disk and resume.
az vm create \
  --resource-group "$RG" --name "$VM" --location "$LOCATION" \
  --size "$SIZE" --image "$IMAGE" --admin-username "$ADMIN" \
  --generate-ssh-keys --os-disk-size-gb "$DISK_GB" \
  --priority Spot --max-price -1 --eviction-policy Deallocate -o table

# ----- 5. NVIDIA driver -------------------------------------------------------
az vm extension set \
  --resource-group "$RG" --vm-name "$VM" \
  --name NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute --version 1.9 -o table

IP="$(az vm show -d --resource-group "$RG" --name "$VM" --query publicIps -o tsv)"
cat <<EOF

VM up at $IP. Next (on the VM):
  ssh $ADMIN@$IP
  nvidia-smi                                  # confirm the A100
  sudo apt-get update && sudo apt-get install -y python3-venv git
  git clone https://github.com/puneethgv/parallel_eagle && cd parallel_eagle
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e ".[train,dev]"
  hf auth login                               # for the Qwen2.5-14B download
  # then run the campaign (see scripts/run_campaign.sh)

STOP BILLING when done (spend is hourly):
  az vm deallocate -g $RG -n $VM     # keeps the disk, stops compute charges
  az group delete -n $RG --yes       # delete everything when fully finished
EOF
