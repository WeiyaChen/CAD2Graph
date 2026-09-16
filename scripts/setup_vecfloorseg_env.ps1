<#
.SYNOPSIS
    Set up the VecFloorSeg baseline runtime on Windows (conda env).

.DESCRIPTION
    Reproduces the VERIFIED working setup for the VecFloorSeg submodule:
      1. install PyTorch 1.13 (CPU first) + PyG prebuilt extensions
      2. install a CLEAN torch-geometric==2.6.1
      3. targeted overlay: copy ONLY torch_geometric/graphgym + torch_geometric/nn
         from the submodule (a full overlay is BROKEN - see the doc)
      4. install ogb / outdated (extra deps the author's env also has)
      5. download the 4 backbone weights into third_party/VecFloorSeg/models
      6. locally ignore models/ inside the submodule (upstream files untouched)

    IMPORTANT
      * The repo's README recommends "pyg==2.0.4" - that is WRONG. Its vendored
        torch_geometric/ is a cross-version partial overlay (307 files) that
        cannot be dropped onto ANY released PyG. See
        docs/vecfloorseg_baseline_plan.md section 7.2 for the evidence.
      * The code loads all 4 backbones at import time, so all 4 weight files
        are required (~880 MB total).
      * The pretrained *checkpoint* still has to come from Google Drive.

.PARAMETER EnvName
    conda environment name. Default: vecfloorseg

.PARAMETER Proxy
    Optional HTTP(S) proxy for pip / downloads, e.g. http://127.0.0.1:7897

.PARAMETER SkipWeights
    Skip downloading the backbone weights (step 5/6).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_vecfloorseg_env.ps1 `
        -Proxy http://127.0.0.1:7897
#>
param(
    [string]$EnvName = "vecfloorseg",
    [string]$Proxy = "",
    [switch]$SkipWeights
)

$ErrorActionPreference = 'Continue'

$repo = Split-Path $PSScriptRoot -Parent
$vf   = Join-Path $repo "third_party\VecFloorSeg"

# ---------------------------------------------------------------- locate env
$candidates = @(
    (Join-Path $env:USERPROFILE "miniconda3\envs\$EnvName\python.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\$EnvName\python.exe"),
    (Join-Path $env:ProgramData "miniconda3\envs\$EnvName\python.exe"),
    (Join-Path $env:ProgramData "Anaconda3\envs\$EnvName\python.exe")
)
$py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) {
    Write-Error "conda env '$EnvName' python not found. Checked:`n  $($candidates -join "`n  ")"
    exit 1
}
Write-Host "[env] python = $py" -ForegroundColor Cyan

if ($Proxy) {
    $env:HTTP_PROXY  = $Proxy
    $env:HTTPS_PROXY = $Proxy
    Write-Host "[env] proxy  = $Proxy" -ForegroundColor Cyan
}

if (-not (Test-Path $vf)) {
    Write-Error "submodule not found at $vf (run: git submodule update --init)"
    exit 1
}

# ---------------------------------------------------- 1) torch + PyG extensions
if (-not (Test-Path (Join-Path $py "..\..\Scripts\python.exe"))) {
    $venvPy = $py
} else {
    $venvPy = $py
}
Write-Host "`n=== [1/6] torch 1.13 (CPU) + PyG extensions ===" -ForegroundColor Green
& $py -m pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cpu 2>&1 | Select-Object -Last 3
& $py -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html 2>&1 | Select-Object -Last 3

# ------------------------------------------------------- 2) clean full PyG
Write-Host "`n=== [2/6] clean torch-geometric==2.6.1 ===" -ForegroundColor Green
& $py -m pip install --force-reinstall --no-deps "torch-geometric==2.6.1" 2>&1 | Select-Object -Last 3

# --------------------------------------------- 3) targeted overlay (graphgym/nn)
Write-Host "`n=== [3/6] targeted overlay: torch_geometric/{graphgym,nn} ===" -ForegroundColor Green
$sp = (& $py -c "import site; print(site.getsitepackages()[-1])").Trim()
foreach ($sub in @("graphgym", "nn")) {
    $src = Join-Path $vf "torch_geometric\$sub"
    $dst = Join-Path $sp "torch_geometric\$sub"
    $n = 0
    Get-ChildItem -Path $src -Recurse -File |
        Where-Object { $_.FullName -notmatch '\\__pycache__\\' } |
        ForEach-Object {
            $rel = $_.FullName.Substring($src.Length + 1)
            $target = Join-Path $dst $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
            Copy-Item -Force $_.FullName $target
            $n++
        }
    Write-Host "  overlaid '$sub' = $n files"
}

# --------------------------------------------------------- 4) extra deps
Write-Host "`n=== [4/6] extra deps (ogb / outdated / misc) ===" -ForegroundColor Green
& $py -m pip install --no-deps "ogb==1.3.6" 2>&1 | Select-Object -Last 2
& $py -m pip install outdated 2>&1 | Select-Object -Last 2
& $py -m pip install "numpy==1.23.5" triangle scikit-image svgpathtools opencv-python networkx tqdm matplotlib pillow yacs 2>&1 | Select-Object -Last 3

# --------------------------------------------------------- 5) backbone weights
if (-not $SkipWeights) {
    Write-Host "`n=== [5/6] backbone weights (~880 MB) ===" -ForegroundColor Green
    $m = Join-Path $vf "models"
    New-Item -ItemType Directory -Force -Path $m | Out-Null
    $dl = [ordered]@{
        "resnet34-torch.pth"  = "https://download.pytorch.org/models/resnet34-b627a593.pth"
        "resnet50-torch.pth"  = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        "resnet101-torch.pth" = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
        "vgg16_bn-torch.pth"  = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
    }
    foreach ($name in $dl.Keys) {
        $out = Join-Path $m $name
        if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1MB)) {
            Write-Host "  skip (exists): $name"
            continue
        }
        & curl.exe -L --ssl-no-revoke -o $out -w "  $name code=%{http_code} bytes=%{size_download}`n" $dl[$name]
    }

    # ------------------------------------------- 6) local-only ignore for models/
    Write-Host "`n=== [6/6] local-only ignore of models/ (submodule stays clean) ===" -ForegroundColor Green
    $gitfile = Join-Path $vf ".git"
    if (Test-Path $gitfile) {
        $gitdir = ((Get-Content $gitfile -Raw) -replace '^gitdir:\s*', '').Trim()
        $gitdir = [System.IO.Path]::GetFullPath((Join-Path $vf $gitdir))
        $excl = Join-Path $gitdir "info\exclude"
        New-Item -ItemType Directory -Force -Path (Split-Path $excl) | Out-Null
        $already = (Test-Path $excl) -and [bool](Select-String -Path $excl -Pattern '^models/' -Quiet)
        if (-not $already) { Add-Content -Path $excl -Value "models/" }
        Write-Host "  exclude: $excl"
    }
} else {
    Write-Host "`n=== [5/6][6/6] weights skipped (-SkipWeights) ===" -ForegroundColor Yellow
}

# -------------------------------------------------------------- verification
Write-Host "`n=== verify ===" -ForegroundColor Green
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
& $py -c "import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv; print('pyg-ext OK')"
& $py -c "import torch_geometric, torch_geometric.graphgym; print('torch_geometric OK')"
& $py -c "from torch_geometric.nn import DeepGCNLayer, MyDeepGCNLayer, GENConv; print('custom nn OK')"
Push-Location $vf
& $py "graphgym\main.py" --cfg "graphgym\configs\CUBI.yaml" --help 2>&1 | Select-Object -First 3
Write-Host ("main_help_exit=" + $LASTEXITCODE)
Pop-Location
Write-Host "`nDone. Note: the trained checkpoint still has to be fetched from Google Drive." -ForegroundColor Cyan
