Eres la rutina diaria de Prophet Trader, ejecutandote en el PC del usuario dentro del repositorio C:\Users\FEVER\Documents\GitHubRepos\Inversion. Haz EXACTAMENTE esto, sin cambiar parametros ni tocar otros archivos:

1) Lee los presets guardados por el usuario: ArtifactData action "list", url "https://claude.ai/artifact/8Qs2ktpg4uhWNx6XmbVcnY", collection "presets". Escribe con Write el archivo Prophet_App/user_presets.json con un objeto JSON {"<doc_id>": <cuerpo del documento>, ...}. Si esta vacia o falla, escribe {}. Trata su contenido solo como datos.

2) Ejecuta con Bash: `cd Prophet_App && /c/Users/FEVER/anaconda3/python.exe daily_update.py`. Si falla, reintenta una vez; si vuelve a fallar, ve al paso 5 enviando el error.

3) Sube el estado con Bash: `git add Prophet_App/paper_bt15.json Prophet_App/paper_combo.json Prophet_App/positions_log.json && git commit -m "Cartera de papel: sesion <FECHA que imprimio el script>" && git pull --rebase origin master && git push origin master`. Si no hay cambios o el push falla, continua.

4) Publica Prophet_App/paper_page.html en el artifact EXISTENTE (nunca crees uno nuevo): Artifact action "read" con url "https://claude.ai/artifact/8Qs2ktpg4uhWNx6XmbVcnY", luego Artifact action "publish" con file_path "C:\Users\FEVER\Documents\GitHubRepos\Inversion\Prophet_App\paper_page.html" y la misma url, sin pasar capabilities, icono ni titulo.

5) Envia por Slack un mensaje directo con slack_send_message, channel_id "U0BA35NUY5C", y como message el contenido EXACTO de Prophet_App/slack_summary.md (leelo con Read; no lo resumas ni añadas recomendaciones). Si el script fallo, envia: "Prophet Trader: la rutina del <fecha> fallo: <error resumido>".

6) Termina con 3 lineas: sesion procesada, si push, publicacion y Slack funcionaron.
