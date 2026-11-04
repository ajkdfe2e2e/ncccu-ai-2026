$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
function Invoke-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed with exit $LASTEXITCODE : $Arguments" }
}
Invoke-Python -Arguments @('scripts/check_cuda.py')
Invoke-Python -Arguments @('-B','-m','unittest','discover','-s','tests','-v')
Invoke-Python -Arguments @('scripts/verify_resume.py','--device','cuda')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$output = 'outputs/resnet18-finetune-' + $stamp
$baseline = 'outputs/resnet18-baseline-20261104-022511'
Invoke-Python -Arguments @('train.py','--device','cuda','--epochs','4','--image-size','224','--batch-size','8','--workers','0','--lr','0.0001','--scheduler','cosine','--min-lr','0.000001','--init-checkpoint',($baseline + '/best.pt'),'--output',$output)
Invoke-Python -Arguments @('predict.py','--device','cuda','--split','valid','--checkpoint',($output + '/best.pt'),'--output',($output + '/valid.csv'))
Invoke-Python -Arguments @('scripts/compare_runs.py','--baseline',($baseline + '/valid.csv'),'--candidate',($output + '/valid.csv'),'--metadata','cxr_first/valid.csv','--output',($output + '/comparison.json'),'--bootstrap','500')
Invoke-Python -Arguments @('scripts/benchmark_valid.py','--run',$output)
Invoke-Python -Arguments @('scripts/inference_manifest.py',($output + '/best.pt'),'--output',($output + '/inference_manifest.json'))
Write-Output "OPTIMIZATION_COMPLETE=$output"
