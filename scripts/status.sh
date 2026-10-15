#!/bin/bash
# Status of the unattended training chain.  Usage: scripts/status.sh [run_name]
R=/gpfs/accounts/eecs542f26s001_class_root/eecs542f26s001_class/rueyday/542final
RUN=${1:-main}
cd "$R"

echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.9j %.8T %.10M %.12L %R" 2>/dev/null

echo
echo "=== checkpoint ==="
if [ -f "runs/$RUN/last.pt" ]; then
  source /sw/lmod/lmod/init/bash >/dev/null 2>&1
  module load python/3.11.5 >/dev/null 2>&1
  source .venv/bin/activate
  python -I -c "
import torch
s = torch.load('runs/$RUN/last.pt', map_location='cpu', weights_only=False)
print(f\"  step {s['step']} / {s['args']['steps']}  ({100*s['step']/s['args']['steps']:.1f}%)\")
"
else
  echo "  no checkpoint yet"
fi

echo
echo "=== last training lines ==="
LATEST=$(ls -t logs/rf-train-*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "  ($LATEST)"
  grep -E "^step|^  val|^===|stopping chain|complete" "$LATEST" | tail -8
fi

echo
echo "=== validation history across all chunks ==="
grep -h "^  val" logs/rf-train-*.out 2>/dev/null | tail -12

echo
echo "To stop the chain:  touch runs/$RUN/STOP   (plus scancel any queued job)"
