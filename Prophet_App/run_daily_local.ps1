# Rutina diaria local de Prophet Trader (la lanza el Programador de tareas de Windows, L-V 23:30).
# Ejecuta Claude Code en modo no interactivo con daily_prompt.md: actualiza carteras, publica el artifact y envia el Slack.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\FEVER\Documents\GitHubRepos\Inversion"
$logDir = Join-Path $repo "Prophet_App\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("daily_" + (Get-Date -Format "yyyy-MM-dd") + ".log")
Set-Location $repo
git pull --rebase -q origin master *>> $log
Get-Content (Join-Path $repo "Prophet_App\daily_prompt.md") -Raw |
  & "C:\Users\FEVER\.local\bin\claude.exe" -p `
    --allowedTools "Bash" "Read" "Write" "Artifact" "ArtifactData" "mcp__claude_ai_Slack__slack_send_message" *>> $log
