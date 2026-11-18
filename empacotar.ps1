# Builds dist/sumo-neuronio.zip for friends: sumo/ + the desafio2 initializer it imports.
# Excludes runtime data (dados/), the local venv, caches and tests.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$dist = Join-Path $root "dist"
$stage = Join-Path $dist "staging\sumo-neuronio"
$zip = Join-Path $dist "sumo-neuronio.zip"

if (Test-Path (Join-Path $dist "staging")) { Remove-Item -Recurse -Force (Join-Path $dist "staging") }
New-Item -ItemType Directory -Force (Join-Path $stage "sumo") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $stage "desafio2") | Out-Null

$files = "pyproject.toml", "uv.lock", ".python-version", "jogar.bat", "README.md",
         "jogo.py", "regras.py", "neuronio.py", "dados_io.py", "gerar_dados_falsos.py"
foreach ($f in $files) {
    Copy-Item (Join-Path $root "sumo\$f") (Join-Path $stage "sumo\$f")
}
Copy-Item (Join-Path $root "desafio2\desafio2_Leandro_Marques_Miguel_Cesario.py") (Join-Path $stage "desafio2\")

if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Remove-Item -Recurse -Force (Join-Path $dist "staging")
Write-Host "Gerado: $zip ($([math]::Round((Get-Item $zip).Length / 1KB)) KB)"
