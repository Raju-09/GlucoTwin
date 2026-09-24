# PowerShell health check script for GlucoTwin services
param (
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

Write-Host "Checking GlucoTwin API health at $ApiUrl..." -ForegroundColor Cyan

try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 3
    if ($health.status -eq "ok") {
        Write-Host "✅ GlucoTwin API is healthy! (Model loaded: $($health.model_loaded))" -ForegroundColor Green
    } else {
        Write-Host "⚠️ API responded but status is: $($health.status)" -ForegroundColor Yellow
    }

    $info = Invoke-RestMethod -Uri "$ApiUrl/model-info" -Method Get -TimeoutSec 3
    Write-Host "Model: $($info.model_name) ($($info.version))" -ForegroundColor Cyan
    Write-Host "Held-out Test MAE: $($info.test_mae_mgdl) mg/dL (vs Persistence: $($info.persistence_mae_mgdl) mg/dL, -$($info.relative_improvement_pct)%)" -ForegroundColor Green
} catch {
    Write-Host "Could not connect to API at ${ApiUrl}: $_" -ForegroundColor Red
}
