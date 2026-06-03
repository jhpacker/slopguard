#!/usr/bin/env bash
# check-ai-metadata.sh — scan images for AI-generation metadata (IPTC + C2PA)

EXIFTOOL=/opt/homebrew/bin/exiftool
C2PATOOL=/opt/homebrew/bin/c2patool

CHECK="✅"
CROSS="❌"

check_image() {
    local file="$1"

    # ── Col 1: IPTC trainedAlgorithmicMedia ──────────────────────────────────
    local exif_all iptc=0
    exif_all=$("$EXIFTOOL" "$file" 2>/dev/null)
    if echo "$exif_all" | grep -qi "trainedAlgorithm"; then
        iptc=1
    fi

    # ── Col 2: Agent Name ────────────────────────────────────────────────────
    # Check "Actions Software Agent" / "Actions Software Agent Name" first,
    # then fall back to "Claim Generator Info Name" (covers Gemini)
    local agent_name=""
    local sa
    sa=$(echo "$exif_all" | grep -i "^Actions Software Agent" | head -1 \
         | sed 's/^[^:]*: *//')
    [[ -n "$sa" ]] && agent_name="$sa"

    if [[ -z "$agent_name" ]]; then
        local cgi
        cgi=$(echo "$exif_all" | grep -i "^Claim Generator Info Name" | head -1 \
              | sed 's/^[^:]*: *//')
        [[ -n "$cgi" ]] && agent_name="$cgi"
    fi

    # ── Col 3: Valid C2PA crypto signature ───────────────────────────────────
    local c2pa_valid=0 c2pa_issuer=""
    if [[ -x "$C2PATOOL" ]]; then
        local c2pa_json
        c2pa_json=$("$C2PATOOL" "$file" 2>/dev/null)
        if [[ -n "$c2pa_json" ]]; then
            local vstate active
            vstate=$(echo "$c2pa_json" | python3 -c \
                "import sys,json; d=json.load(sys.stdin); print(d.get('validation_state',''))" 2>/dev/null)
            if [[ "$vstate" == "Valid" ]]; then
                c2pa_valid=1
                active=$(echo "$c2pa_json" | python3 -c \
                    "import sys,json; d=json.load(sys.stdin); print(d.get('active_manifest',''))" 2>/dev/null)
                c2pa_issuer=$(echo "$c2pa_json" | python3 -c \
                    "import sys,json
d=json.load(sys.stdin)
m=d['manifests'].get('$active',{})
print(m.get('signature_info',{}).get('issuer',''))" 2>/dev/null)
            fi
        fi
    fi

    # Emit 4-column TSV: file | iptc | agent | c2pa
    local c1 c2 c3
    [[ $iptc -eq 1 ]]       && c1="$CHECK" || c1="$CROSS"
    [[ -n "$agent_name" ]]  && c2="$CHECK" || c2="$CROSS"
    [[ $c2pa_valid -eq 1 ]] && c3="$CHECK" || c3="$CROSS"

    printf "%s\t%s\t%s\t%s\n" \
        "$(basename "$file")" \
        "$c1" \
        "${c2}${agent_name:+ $agent_name}" \
        "${c3}${c2pa_issuer:+ $c2pa_issuer}"
}

# ── Collect files ─────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    files=("$@")
else
    files=()
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(find . -maxdepth 1 \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No images found." >&2
    exit 1
fi

# ── Render: pipe TSV into Python for Unicode-aware column alignment ───────────
# python3 -c reads the script from the arg, leaving stdin free for the pipe.
{
    printf "File\tTagged as AI (IPTC)\tAgent Name (Exif)\tValid C2PA sig\n"
    for f in "${files[@]}"; do
        check_image "$f"
    done
} | python3 -c '
import sys, unicodedata

def vlen(s):
    """Visual width: wide/fullwidth chars count as 2, everything else 1."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

def rpad(s, width):
    return s + " " * max(0, width - vlen(s))

rows = [line.rstrip("\n").split("\t") for line in sys.stdin]
widths = [max(vlen(row[i]) for row in rows) for i in range(4)]
sep = "─" * (sum(widths) + 2 * 3)   # 3 two-space gaps between 4 cols

header, *data = rows
print()
print("AI-generated image metadata detection")
print(sep)
print("  ".join(rpad(cell, widths[i]) for i, cell in enumerate(header)).rstrip())
print(sep)
for row in data:
    print("  ".join(rpad(cell, widths[i]) for i, cell in enumerate(row)).rstrip())
print(sep)
print()
'
