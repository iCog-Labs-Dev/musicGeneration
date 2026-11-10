# Native Windows CI/evidence, no elevation, Developer Mode, asset acquisition or dispatch.
[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$ScratchParent = $env:RUNNER_TEMP,
    [switch]$Renderer,
    [string]$Manifest,
    [string]$Dll,
    [string]$DllSha256,
    [string]$EvidenceParent
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'This script requires native Windows, not WSL/simulation' }
if ($Renderer) {
    if (-not $Manifest -or -not $Dll -or -not $DllSha256 -or -not $EvidenceParent) {
        throw 'Enabled renderer requires -Manifest, -Dll, -DllSha256, -EvidenceParent; see docs/audio-testing.md'
    }
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $Dll -PathType Leaf) -or
        -not (Test-Path -LiteralPath $EvidenceParent -PathType Container)) {
        throw 'Enabled renderer prerequisite missing: approved local manifest, DLL or external evidence directory'
    }
}
$Repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = (Get-Command $Python -ErrorAction Stop).Source
if (-not $ScratchParent) { $ScratchParent = [System.IO.Path]::GetTempPath() }
$ScratchParent = (Resolve-Path -LiteralPath $ScratchParent).Path
# Use explicit code points so Windows PowerShell 5 and pwsh both preserve the name.
$UnicodeName = 'aimusic wheel caf' + [char]0x00e9 + ' ' + [char]0x97f3 + [char]0x8272
$Root = Join-Path $ScratchParent ($UnicodeName + ' ' + [guid]::NewGuid().ToString('N'))
if ($Root.StartsWith($Repo + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'ScratchParent must be outside the checkout; generated assets/paths must not be tracked'
}
New-Item -ItemType Directory -Path $Root | Out-Null
Start-Transcript -LiteralPath (Join-Path $Root 'commands.log') | Out-Null
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    Write-Host ('COMMAND ' + (@($Executable) + $Arguments | ConvertTo-Json -Compress))
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : $Executable" }
}
. (Join-Path $PSScriptRoot 'windows-audio-environment.ps1')
$SavedEnv = Enter-AudioTestEnvironment
try {
    Invoke-Checked $Python @('-I', (Join-Path $Repo 'tests/windows_token.py'))
    $Source = Join-Path $Root 'source'
    $Checks = Join-Path $Root 'checks'
    $Wheels = Join-Path $Root 'wheels'
    New-Item -ItemType Directory -Path $Source, $Checks, $Wheels | Out-Null
    Copy-Item -LiteralPath (Join-Path $Repo 'aimusic') -Destination $Source -Recurse
    foreach ($Name in @('pyproject.toml', 'readme.md')) {
        Copy-Item -LiteralPath (Join-Path $Repo $Name) -Destination $Source
    }
    Copy-Item -LiteralPath (Join-Path $Repo 'tests') -Destination $Checks -Recurse
    Copy-Item -LiteralPath (Join-Path $Repo 'docs') -Destination $Checks -Recurse
    $Venv = Join-Path $Root 'venv'
    Invoke-Checked $Python @('-m', 'venv', $Venv)
    $InstalledPython = Join-Path $Venv 'Scripts/python.exe'
    Invoke-Checked $InstalledPython @('-m', 'pip', 'install', 'setuptools>=68', 'wheel')
    Invoke-Checked $InstalledPython @('-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-index', '--wheel-dir', $Wheels, $Source)
    $Built = @(Get-ChildItem -LiteralPath $Wheels -Filter 'aimusic-*.whl')
    if ($Built.Count -ne 1) { throw 'Expected exactly one built aimusic wheel' }
    New-Item -ItemType Directory -Path (Join-Path $Checks 'wheels') | Out-Null
    Copy-Item -LiteralPath $Built[0].FullName -Destination (Join-Path $Checks 'wheels')
    $Extra = if ($Renderer) { 'audio-render,dev' } else { 'audio,dev' }
    $Requirements = @('-m', 'pip', 'install', ($Built[0].FullName + '[' + $Extra + ']'))
    if ($Renderer) { $Requirements += 'soundfile==0.14.0' }
    Invoke-Checked $InstalledPython $Requirements
    # This exact checkout is deliberately shared read-only with the CI user.
    $Revision = & git -c "safe.directory=$Repo" -C $Repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Could not record source revision' }
    @{ source_revision = $Revision; wheel_sha256 = (Get-FileHash $Built[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant();
       renderer_enabled = [bool]$Renderer } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Root 'build.json')
    # No aimusic source in this cwd, no PYTHONPATH, and no editable installation.
    Push-Location $Checks
    try {
        if ($Renderer) {
            Invoke-Checked $InstalledPython @('-m', 'tests.windows_fluidsynth_evidence', '--run',
                '--manifest', $Manifest, '--dll', $Dll, '--dll-sha256', $DllSha256,
                '--out', $EvidenceParent, '--checkout', $Repo)
        } else {
            Invoke-Checked $InstalledPython @('-m', 'tests.windows_ci', '--checkout', $Repo)
        }
    } finally { Pop-Location }
    Write-Host "External build/test logs (never commit): $Root"
} finally {
    Exit-AudioTestEnvironment $SavedEnv
    Stop-Transcript | Out-Null
}
