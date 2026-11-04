#!/bin/bash
# grid/notify_tg.sh with stubs for curl (Telegram) and sacct, on a fake grid:
# --test finds and saves the chat id and never puts the token on a command line; the report holds this job's decider
# output only; a failed analysis sends its log instead; a Telegram error fails the job; a bad job id is refused.
#   bash grid/tests/notify_tg_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NOTIFY=$(cd "$HERE/.." && pwd)/notify_tg.sh
T=$(mktemp -d "${TMPDIR:-/tmp}/notify_tg.XXXXXX")
trap 'rm -rf "$T"' EXIT
fail() { echo "NOTIFY TEST FAIL: $*" >&2; exit 1; }
TOKEN=123456:SECRET-token-for-the-test

mkdir -p "$T/bin" "$T/d/grid" "$T/d/logs" "$T/home"
cat > "$T/bin/curl" <<'SH'
#!/bin/bash
# records argv and the URL from -K; replies like Telegram
echo "ARGV $*" >> "$STUB_LOG"
while [ $# -gt 0 ]; do
    case $1 in
        -K) url=$(sed -n 's/^url = "\(.*\)"$/\1/p' "$2"); shift 2 ;;
        --data-urlencode) v=$2; case $v in text@*) echo "TEXT $(cat "${v#text@}" | tr '\n' '|')" >> "$STUB_LOG" ;; *) echo "DATA $v" >> "$STUB_LOG" ;; esac; shift 2 ;;
        -F) echo "FORM $2" >> "$STUB_LOG"; case $2 in document=@*) cp "${2#document=@}" "$STUB_DOC" ;; esac; shift 2 ;;
        *) shift ;;
    esac
done
echo "URL $url" >> "$STUB_LOG"
[ "${STUB_FAIL:-}" = 1 ] && { echo '{"ok":false,"error_code":401,"description":"Unauthorized"}'; exit 0; }
case $url in
    */getUpdates) echo '{"ok":true,"result":[{"update_id":1,"message":{"chat":{"id":777,"type":"private"},"text":"hi"}},{"update_id":2,"message":{"chat":{"id":98765,"type":"private"},"text":"/start"}}]}' ;;
    *) echo '{"ok":true,"result":{}}' ;;
esac
SH
cat > "$T/bin/sacct" <<'SH'
#!/bin/bash
echo "COMPLETED|00:15:02"
SH
chmod +x "$T/bin/curl" "$T/bin/sacct"
export PATH=$T/bin:$PATH STUB_LOG=$T/stub.log STUB_DOC=$T/doc.txt C2S_D=$T/d C2S_ENV=$T/home/c2s.env

printf 'HF_TOKEN=hf_x\nTG_TOKEN=%s\n' "$TOKEN" > "$C2S_ENV"
printf 'lr1e-4_ep2_s42\t/r/a\t0\t0\nlr1e-4_ep1_s42\t/r/b\t24525624\t24525624\nlr1e-4_ep4_s42\t/r/c\t24525625\t24525625\n' > "$T/d/grid/cells.tsv"
cat > "$T/d/grid/chain.log" <<'LOG'
2026-11-04 03:15:00 analysis job 24514310 on r1u01
2026-11-04 03:30:00 decider
STAGE A  learning rate at 2 epochs
  OLD-ANALYSIS-LINE
NEXT: {"action": "train", "cells": [{"lr": 0.0003, "ep": 8, "seed": 42}]}
2026-11-05 10:00:00 analysis job 24525626 on r1u02
violins: 2 run(s) ...
2026-11-05 10:12:00 decider
FORCE_LR 1e-4: stage B runs at this LR (set by the author in grid/FORCE_LR)
STAGE A  learning rate at 2 epochs
  lr1e-4_ep2_s42         violins mismatch 14 (copy of input 22)  eval_loss 2.1838  kept
  -> lr3e-4_ep2_s42: reverse task: significantly better than every other run; energy guard passed
  LR set by the author (grid/FORCE_LR): 1e-4; the rule chose 3e-4
STAGE B  epochs at LR 0.0001
  lr1e-4_ep1_s42         reverse task, human: 70.0%
  lr1e-4_ep4_s42 vs lr1e-4_ep2_s42: only lr1e-4_ep4_s42 right 50, only lr1e-4_ep2_s42 right 20, McNemar p_bonf 0.001
  -> lr1e-4_ep4_s42: reverse task: significantly better than every other run; energy guard passed
NEXT: {"action": "train", "cells": [{"lr": 0.0001, "ep": 8, "seed": 42}], "why": "stage B+: the winner 4 epochs is at the upper edge; try 8"}
STOP: the cap of 0 trainings would be exceeded (8); the rule wants: stage B+: the winner 4 epochs is at the upper edge; try 8
2026-11-05 10:13:00 nothing to submit; see /xdisk/vkalin/dolzhenko/c2s/grid/decisions and /xdisk/vkalin/dolzhenko/c2s/grid/STOPPED
LOG
echo "the cap of 0 trainings would be exceeded (8); the rule wants: stage B+: the winner 4 epochs is at the upper edge; try 8" > "$T/d/grid/STOPPED"

echo "== --test: chat id found, saved, test message sent, token never in argv"
bash "$NOTIFY" --test
grep -qx "TG_CHAT=98765" "$C2S_ENV" || fail "chat id not saved: $(cat "$C2S_ENV")"
[ "$(stat -c %a "$C2S_ENV" 2>/dev/null || stat -f %Lp "$C2S_ENV")" = 600 ] || fail "env file is not 600"
grep -q "URL https://api.telegram.org/bot$TOKEN/getUpdates" "$STUB_LOG" || fail "getUpdates not called"
grep -q "URL https://api.telegram.org/bot$TOKEN/sendMessage" "$STUB_LOG" || fail "sendMessage not called"
grep -q "DATA chat_id=98765" "$STUB_LOG" || fail "test message not sent to the saved chat"
grep "^ARGV" "$STUB_LOG" | grep -q "SECRET" && fail "the token appeared on the curl command line"
: > "$STUB_LOG"
bash "$NOTIFY" --test > /dev/null
grep -q getUpdates "$STUB_LOG" && fail "--test looked up the chat id again although it is saved"
echo "   ok"

echo "== analysis 24525626 finished: message + report file with this job's decision only"
: > "$STUB_LOG"
bash "$NOTIFY" 24525626; grep "^TEXT" "$STUB_LOG" | tr "|" "\n" | sed "s/^/   msg> /"
REP=$T/d/grid/reports/analysis_24525626.txt
[ -s "$REP" ] || fail "no report"
cmp -s "$REP" "$STUB_DOC" || fail "the document sent is not the report"
grep -q "lr1e-4_ep4_s42 vs lr1e-4_ep2_s42" "$REP" || fail "report lacks the McNemar line"
grep -q "STAGE B  epochs at LR 0.0001" "$REP" || fail "report lacks stage B"
grep -q "OLD-ANALYSIS-LINE" "$REP" && fail "report includes an older analysis"
grep -q "lr1e-4_ep4_s42  job 24525625  COMPLETED 00:15:02" "$REP" || fail "report lacks the cell job states"
grep -q "^TEXT C2S grid: analysis 24525626 COMPLETED 00:15:02" "$STUB_LOG" || fail "message header: $(grep ^TEXT "$STUB_LOG")"
grep "^TEXT" "$STUB_LOG" | grep -q "STOPPED: the cap of 0" || fail "message lacks STOPPED"
grep "^TEXT" "$STUB_LOG" | grep -q "NEXT: " || fail "message lacks NEXT"
grep "^TEXT" "$STUB_LOG" | grep -q "LR set by the author" || fail "message lacks the forced LR line"
grep "^TEXT" "$STUB_LOG" | grep -q "reverse task, human: 70.0%" || fail "message lacks the stage B numbers"
grep "^TEXT" "$STUB_LOG" | grep -q "McNemar p_bonf 0.001" || fail "message lacks the McNemar line"
grep "^TEXT" "$STUB_LOG" | grep -q "OLD-ANALYSIS-LINE\|violins mismatch 14" && fail "message carries stage A or an older analysis"
grep -q "FORM document=@$REP" "$STUB_LOG" || fail "report not sent as a document"
grep "^ARGV" "$STUB_LOG" | grep -q "SECRET" && fail "the token appeared on the curl command line"
echo "   ok"; sed 's/^/   | /' "$REP" | head -n 12

echo "== analysis that failed before the decider: its log goes out instead"
printf '2026-11-05 11:00:00 analysis job 24530000 on r1u03\n2026-11-05 11:00:05 FAIL: missing data file x (run deploy.sh)\n' >> "$T/d/grid/chain.log"
printf 'line a\nTraceback: boom\n' > "$T/d/logs/c2s-analyze_24530000.out"
bash "$NOTIFY" 24530000 > /dev/null
R2=$T/d/grid/reports/analysis_24530000.txt
grep -q "the decider did not run" "$R2" || fail "failed analysis not recognised"
grep -q "FAIL: missing data file" "$R2" || fail "chain.log lines of the failed job missing"
grep -q "Traceback: boom" "$R2" || fail "Slurm log tail missing"
echo "   ok"

echo "== Telegram error fails the job; bad job id refused"
if STUB_FAIL=1 bash "$NOTIFY" 24525626 2> "$T/err"; then fail "exit 0 although Telegram said ok:false"; fi
grep -q "Unauthorized" "$T/err" || fail "error text not shown: $(cat "$T/err")"
grep -q "SECRET" "$T/err" && fail "the token leaked into the error message"
if bash "$NOTIFY" '1;rm' 2> /dev/null; then fail "bad job id accepted"; fi
echo "   ok"
echo "NOTIFY OK"
