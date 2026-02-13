#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WIDTH="${THESIS_FMT_WIDTH:-100}"

for f in thesis/sections/*.tex; do
  awk -v width="${WIDTH}" '
  function flush_para() {
    if (para != "") {
      cmd = "fmt -w " width
      print para | cmd
      close(cmd)
      para = ""
    }
  }
  {
    line = $0

    if (line ~ /^[[:space:]]*\\begin\{(table|tabular|equation|align|align\*|verbatim|lstlisting)\}/) {
      flush_para()
      in_env = 1
      print line
      next
    }

    if (line ~ /^[[:space:]]*\\end\{(table|tabular|equation|align|align\*|verbatim|lstlisting)\}/) {
      flush_para()
      in_env = 0
      print line
      next
    }

    if (in_env) {
      print line
      next
    }

    if (line ~ /^[[:space:]]*$/) {
      flush_para()
      print ""
      next
    }

    if (line ~ /^[[:space:]]*%/ || line ~ /^[[:space:]]*\\/ || (line ~ /&/ && line ~ /\\\\/)) {
      flush_para()
      print line
      next
    }

    gsub(/[[:space:]]+$/, "", line)
    if (para == "") {
      para = line
    } else {
      para = para " " line
    }
  }
  END {
    flush_para()
  }
  ' "$f" > "$f.tmp"

  mv "$f.tmp" "$f"
done

echo "Formatted thesis sections with width ${WIDTH}."
