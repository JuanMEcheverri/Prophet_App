# Rutina diaria local de Prophet Trader (Programador de tareas de Windows, L-V 23:30).
# 1) calculo y estado (Python directo, no depende de Claude); 2) commit/push; 3) Claude solo para el Slack.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\FEVER\Documents\GitHubRepos\Inversion"
$logDir = Join-Path $repo "Prophet_App\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("daily_" + (Get-Date -Format "yyyy-MM-dd") + ".log")
function Log($m) { Add-Content -Path $log -Value ("[" + (Get-Date -Format "HH:mm:ss") + "] " + $m) -Encoding utf8 }
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"

# El repo es de JuanMEcheverri; otra cuenta de gh puede estar activa (push 403). Se fuerza la del dueño.
& "C:\Program Files\GitHub CLI\gh.exe" auth switch --user JuanMEcheverri 2>&1 | ForEach-Object { Log $_ }
git pull --rebase -q origin master 2>&1 | ForEach-Object { Log $_ }

Log "daily_update.py"
Push-Location (Join-Path $repo "Prophet_App")
& "C:\Users\FEVER\anaconda3\python.exe" daily_update.py 2>&1 | ForEach-Object { Log $_ }
$ok = $LASTEXITCODE -eq 0
Pop-Location
Log ("daily_update exit ok: " + $ok)

git add Prophet_App/paper_bt15.json Prophet_App/paper_combo.json Prophet_App/positions_log.json 2>&1 | ForEach-Object { Log $_ }
git commit -q -m "Cartera de papel: rutina diaria $(Get-Date -Format yyyy-MM-dd)" 2>&1 | ForEach-Object { Log $_ }
git pull --rebase -q origin master 2>&1 | ForEach-Object { Log $_ }
git push -q origin master 2>&1 | ForEach-Object { Log $_ }

# Slack via Claude Code (requiere sesion iniciada en el CLI; si falla, el calculo y el push ya estan hechos)
$prompt = if ($ok) {
  "Envia por Slack un mensaje directo con slack_send_message, channel_id ""U0BA35NUY5C"", y como message el contenido EXACTO del archivo Prophet_App/slack_summary.md (leelo con Read; no lo resumas ni añadas nada). Termina con una linea diciendo si se envio."
} else {
  "Envia por Slack un mensaje directo con slack_send_message, channel_id ""U0BA35NUY5C"", con el texto: ""Prophet Trader: la rutina diaria del $(Get-Date -Format yyyy-MM-dd) fallo al calcular. Revisa Prophet_App/logs/$(Split-Path $log -Leaf)"". Termina con una linea diciendo si se envio."
}
$prompt | & "C:\Users\FEVER\.local\bin\claude.exe" -p --allowedTools "Read,mcp__claude_ai_Slack__slack_send_message" 2>&1 | ForEach-Object { Log $_ }
Log "fin"
