# Open-source corpus runner v2: 4 more models x 182 prompts via Ollama, sequential.
# v2 fixes: clears API-key env vars so ONLY the Ollama adapter runs
# (v1 accidentally burned 546 Claude calls because keys were system env vars).
$ErrorActionPreference = "Continue"
$corpusDir = "C:\Users\senti\OneDrive\Desktop\websites\fingerprint\corpus"
$models = @("qwen2.5:1.5b", "llama3.2:3b", "mistral-nemo:12b", "gemma2:9b")

# kill API keys for this process tree -> harness only sees Ollama
$env:ANTHROPIC_API_KEY = $null
$env:GEMINI_API_KEY = $null
$env:OPENAI_API_KEY = $null
$env:OPENROUTER_API_KEY = $null
$env:GOOGLE_API_KEY = $null

Set-Location $corpusDir
foreach ($m in $models) {
    $safe = ($m -replace ":", "_") -replace "\.", ""
    Write-Host "=== pulling $m ==="
    ollama pull $m
    Write-Host "=== running $m ==="
    $env:OLLAMA_MODEL = $m
    python harness.py --out "corpus_ollama_$safe.jsonl" --delay 0.2 2>&1 | Tee-Object -FilePath "ollama_$safe.log" -Append
    Write-Host "=== done $m ==="
}
Write-Host "ALL DONE"
"done $(Get-Date -Format o)" | Out-File "$env:USERPROFILE\Desktop\ollama_corpus_v2.done" -Encoding utf8
