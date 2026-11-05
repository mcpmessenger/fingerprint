# Open-source corpus runner: 182 prompts x 3 models via Ollama, sequential.
# Resumable (harness skips done prompt_ids). Run: powershell -ExecutionPolicy Bypass -File run-oss-corpus.ps1
$ErrorActionPreference = "Continue"
$corpusDir = "C:\Users\senti\OneDrive\Desktop\websites\fingerprint\corpus"
$models = @("llama3.1:8b", "mistral:7b", "qwen2.5:7b")

function Wait-Model($name) {
    Write-Host "waiting for $name ..."
    while ($true) {
        $list = ollama list 2>$null
        if ($list -match [regex]::Escape(($name -split ":")[0])) { Write-Host "$name ready"; return }
        Start-Sleep -Seconds 60
    }
}

Set-Location $corpusDir
foreach ($m in $models) {
    Wait-Model $m
    $safe = $m -replace ":", "_"
    Write-Host "=== running $m ==="
    $env:OLLAMA_MODEL = $m
    # only Ollama adapter configured -> only it runs (no API keys set here)
    python harness.py --out "corpus_ollama_$safe.jsonl" --delay 0.2
    Write-Host "=== done $m ==="
}
Write-Host "ALL DONE"
