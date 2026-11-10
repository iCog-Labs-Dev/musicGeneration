# Paths are JSON data encoded as UTF-8/base64, never interpolated PowerShell code.
function New-AudioStandardUserCommand([string]$Script, [string]$Python, [string]$Scratch, [string]$Git) {
    $Json = @{ Script = $Script; Python = $Python; Scratch = $Scratch; Git = $Git } | ConvertTo-Json -Compress
    $Payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Json))
    $Code = @'
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$Spec = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__PAYLOAD__')) | ConvertFrom-Json
$env:TEMP = $Spec.Scratch
$env:TMP = $Spec.Scratch
$env:RUNNER_TEMP = $Spec.Scratch
$env:PATH = [IO.Path]::GetDirectoryName($Spec.Git) + ';' + $env:PATH
try {
    # PS5 command discovery treats [] as wildcards; pwsh treats paths literally.
    $Command = if ($PSVersionTable.PSVersion.Major -le 5) {
        Get-Command -Name ([WildcardPattern]::Escape($Spec.Script)) -ErrorAction Stop
    } else { $Spec.Script }
    & $Command -Python $Spec.Python -ScratchParent $Spec.Scratch
    if (-not $?) { exit 1 }
    exit 0
} catch {
    Write-Error $_
    exit 1
}
'@
    [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Code.Replace('__PAYLOAD__', $Payload)))
}
