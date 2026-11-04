param([int]$InstallerProcessId = 0)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if ($InstallerProcessId -gt 0) {
    $installer = Get-Process -Id $InstallerProcessId -ErrorAction SilentlyContinue
    if ($installer) {
        Write-Output "Waiting for existing installer PID $InstallerProcessId (no duplicate download)."
        Wait-Process -Id $InstallerProcessId
    }
}
function Invoke-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed with exit $LASTEXITCODE : $Arguments" }
}
Invoke-Python -Arguments @('scripts/check_cuda.py')
Invoke-Python -Arguments @('-B','-m','unittest','discover','-s','tests','-v')
Invoke-Python -Arguments @('scripts/smoke_pipeline.py','--device','cuda','--pretrained','weights/resnet18-f37072fd.pth')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$output = 'outputs/resnet18-baseline-' + $stamp
Invoke-Python -Arguments @('train.py','--device','cuda','--epochs','1','--image-size','224','--batch-size','8','--pretrained','weights/resnet18-f37072fd.pth','--output',$output)
Invoke-Python -Arguments @('predict.py','--device','cuda','--split','valid','--checkpoint',($output + '/best.pt'),'--output',($output + '/valid.csv'))
Invoke-Python -Arguments @('scripts/inference_manifest.py',($output + '/best.pt'),'--output',($output + '/inference_manifest.json'))
Invoke-Python -Arguments @('scripts/preflight.py','--output',($output + '/preflight.json'))
Write-Output "INITIAL_VALIDATION_COMPLETE=$output"
