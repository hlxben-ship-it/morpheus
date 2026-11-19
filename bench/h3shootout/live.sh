# Every 5 s: <run>/stats.html (self-refreshing table) from status.json; videos.html is rebuilt when a clip finishes. Stops when the run has finished.
d="$1"; last=""
while :; do
  { echo '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5">'; .venv/Scripts/python.exe bench/h3shootout/page.py "$d"; } > "$d/stats.tmp" && mv -f "$d/stats.tmp" "$d/stats.html"
  now=$(grep -c '"video"' "$d/status.json")
  [ "$now" != "$last" ] && .venv/Scripts/python.exe bench/h3shootout/videos.py "$d" && last=$now
  grep -q '"finished"' "$d/status.json" && break
  sleep 5
done
