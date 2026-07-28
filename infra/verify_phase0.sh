#!/usr/bin/env bash
# Phase 0 verification — proves the auth holes are actually closed.
#
#   BASE=https://<pod-id>-8888.proxy.runpod.net bash infra/verify_phase0.sh
#
# Run this against a RUNNING pod after `resume.sh`. Exits non-zero on any failure,
# so it is safe to wire into CI later.
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8888}"
PASS=0; FAIL=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s (got %s, want %s)\n' "$1" "$2" "$3"; FAIL=$((FAIL+1)); }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$@"; }

echo "== target: $BASE =="
curl -s --max-time 20 "$BASE/health" | grep -q '"ready":true' \
  || { echo "server not ready — run infra/resume.sh first"; exit 1; }

# ── create a session; capture id + secret ────────────────────────────────────
RESP=$(curl -s --max-time 30 -X POST "$BASE/session" -F speaker=Adam -F listener=Sara)
SID=$(printf '%s' "$RESP" | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
SEC=$(printf '%s' "$RESP" | sed -n 's/.*"session_secret":"\([^"]*\)".*/\1/p')
[ -n "$SID" ] && ok "POST /session returns a session_id" || bad "session_id" "empty" "an id"
[ -n "$SEC" ] && ok "POST /session returns a session_secret" || bad "session_secret" "empty" "a secret"
[ -n "$SEC" ] || exit 1

# ── the four endpoints must reject a bare session_id ─────────────────────────
c=$(code -X POST "$BASE/distill" -F "session_id=$SID")
[ "$c" = "403" ] && ok "/distill without bearer -> 403" || bad "/distill without bearer" "$c" "403"

c=$(code -X POST "$BASE/approve" -F "session_id=$SID")
[ "$c" = "403" ] && ok "/approve without bearer -> 403" || bad "/approve without bearer" "$c" "403"

c=$(code -X POST "$BASE/turn" -F "session_id=$SID" -F "audio=@/dev/null;filename=a.m4a")
[ "$c" = "403" ] && ok "/turn without bearer -> 403" || bad "/turn without bearer" "$c" "403"

# ── a wrong secret must fail the same way as a missing one ───────────────────
c=$(code -X POST "$BASE/distill" -F "session_id=$SID" -H "Authorization: Bearer wrong-$SEC")
[ "$c" = "403" ] && ok "wrong bearer -> 403" || bad "wrong bearer" "$c" "403"

# ── an unknown session must not be distinguishable from a bad secret ─────────
c=$(code -X POST "$BASE/distill" -F "session_id=deadbeef" -H "Authorization: Bearer $SEC")
[ "$c" = "403" ] && ok "unknown session -> 403 (not 404: ids stay unconfirmable)" \
                 || bad "unknown session" "$c" "403"

# ── /audio: the original hole. Guessing an id must not work. ────────────────
c=$(code "$BASE/audio/deadbeefdeadbeefdeadbeefdeadbeef")
[ "$c" = "403" ] && ok "/audio bare guess -> 403" || bad "/audio bare guess" "$c" "403"

c=$(code "$BASE/audio/deadbeefdeadbeefdeadbeefdeadbeef?session_id=$SID" \
         -H "Authorization: Bearer $SEC")
[ "$c" = "403" ] && ok "/audio authed but not ours -> 403" \
                 || bad "/audio foreign id" "$c" "403"

c=$(code "$BASE/audio/../../etc/passwd?session_id=$SID" -H "Authorization: Bearer $SEC")
[ "$c" != "200" ] && ok "/audio path traversal blocked ($c)" || bad "traversal" "200" "not 200"

# ── the happy path must still work ──────────────────────────────────────────
c=$(code -X POST "$BASE/distill" -F "session_id=$SID" -H "Authorization: Bearer $SEC")
# 422 is correct here: authorised, but nothing has been said in this session yet.
[ "$c" = "422" ] && ok "/distill WITH bearer -> 422 (authorised, nothing said yet)" \
                 || bad "/distill with bearer" "$c" "422"

# ── CORS wildcard must be gone ──────────────────────────────────────────────
if curl -s -I --max-time 20 -H "Origin: https://evil.example" "$BASE/health" \
     | grep -qi 'access-control-allow-origin'; then
  bad "CORS" "allow-origin header present" "no header"
else
  ok "no Access-Control-Allow-Origin echoed back"
fi

echo ""
printf 'Phase 0: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
