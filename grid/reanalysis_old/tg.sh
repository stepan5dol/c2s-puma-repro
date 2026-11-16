# Telegram for grid/reanalysis_old, as grid/notify_tg.sh: token and chat id from ~/c2s/c2s.env; the token goes to curl
# only in a config read from a file descriptor, never on a command line or in a log.
# usage: source tg.sh; tg_send <plain text file> [file to attach]
#   the text is sent as one monospace block (HTML <pre>, escaped); if Telegram rejects the formatting, as plain text.
#   Never fails the caller: on any error it says so and returns 0 (the report stays on disk).
tg_send() {
    local txt=$1 doc=${2:-} env=${C2S_ENV:-$HOME/c2s/c2s.env} token chat html out
    token=$(sed -n 's/^TG_TOKEN=//p' "$env" 2>/dev/null | tail -n 1)
    chat=$(sed -n 's/^TG_CHAT=//p' "$env" 2>/dev/null | tail -n 1)
    if [ -z "$token" ] || [ -z "$chat" ]; then echo "no TG_TOKEN/TG_CHAT in $env: not sent"; return 0; fi
    _tg() {  # _tg <method> [curl args]: 0 if Telegram answers "ok":true
        out=$(curl -sS --max-time 60 --retry 3 -K <(printf 'url = "%s/bot%s/%s"\n' "${TG_API:-https://api.telegram.org}" "$token" "$1") "${@:2}") || return 1
        printf '%s' "$out" | grep -q '"ok":true' || { echo "Telegram $1: $(printf '%s' "$out" | grep -o '"description":"[^"]*"')"; return 1; }
    }
    html=$(mktemp "${TMPDIR:-/tmp}/c2s_tg.XXXXXX")
    { printf '<pre>'; head -c 3800 "$txt" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'; printf '</pre>'; } > "$html"
    _tg sendMessage --data-urlencode "chat_id=$chat" --data-urlencode "parse_mode=HTML" --data-urlencode "text@$html" \
        || _tg sendMessage --data-urlencode "chat_id=$chat" --data-urlencode "text@$txt" \
        || echo "Telegram message not sent ($txt stays on disk)"
    rm -f "$html"
    if [ -n "$doc" ] && [ -s "$doc" ]; then
        _tg sendDocument -F "chat_id=$chat" -F "document=@$doc" || echo "Telegram file not sent ($doc stays on disk)"
    fi
    return 0
}
