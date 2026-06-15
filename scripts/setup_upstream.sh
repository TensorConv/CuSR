#!/usr/bin/env bash
# Clone upstream third-party repos at pinned SHAs.
# Rerunning is safe: each repo is fetched + reset only if the SHA differs.
#
# Add a new upstream by appending an entry to the UPSTREAMS array below.
# Format: "name|url|sha"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT/upstream"
mkdir -p "$UPSTREAM_DIR"

UPSTREAMS=(
  "evogp|https://github.com/EMI-Group/evogp.git|b590e9269d878fae7a279deaf9231a640dc75991"
  "pse|https://github.com/intell-sci-comput/PSE.git|de19d442a8aadbff06e69cc21dbbd67e63a62e89"
  "eml_sr|https://github.com/VA00/SymbolicRegressionPackage.git|db31d58ef97e53284efe1da9f22ae2213417079e"
  "oxieml|https://github.com/cool-japan/oxieml.git|ad17db77a0a9d633fad0e3517227537724653da6"
  "pysr|https://github.com/MilesCranmer/PySR.git|c435527ccddb89a7a4e2835fb064679b3ed3e537"
  "sr_jl|https://github.com/MilesCranmer/SymbolicRegression.jl.git|9c3b5e518e19170f656edad8a07e000b626ab589"
  "gsl|https://git.savannah.gnu.org/git/gsl.git|f6de5706f9287e7b9cf5eebf31274caedf6f9603"
  "nlopt|https://github.com/stevengj/nlopt.git|11cff2c773b4b98821915a72179f4667c307ce6d"
  # add more: "name|url|sha"
)

for entry in "${UPSTREAMS[@]}"; do
  IFS='|' read -r name url sha <<< "$entry"
  dest="$UPSTREAM_DIR/$name"

  if [ ! -d "$dest/.git" ]; then
    echo "[$name] cloning $url"
    git clone "$url" "$dest"
  fi

  current=$(git -C "$dest" rev-parse HEAD)
  if [ "$current" != "$sha" ]; then
    echo "[$name] fetching + checking out $sha (was $current)"
    git -C "$dest" fetch --all --tags
    git -C "$dest" checkout "$sha"
  else
    echo "[$name] already at $sha"
  fi
done

echo "Done. Upstream repos in $UPSTREAM_DIR."
