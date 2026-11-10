# GitHub-hosted bootstrap ONLY. Local users run test-windows-audio.ps1 directly.
# Admin setup provisions an ephemeral Users-only account; ALL build/check work
# runs with alternate credentials and independently verifies its unelevated token.
[CmdletBinding()]
param([string]$Python = 'python', [string]$ScratchParent = $env:RUNNER_TEMP)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Hosted account provisioning needs admin setup; use test-windows-audio.ps1 directly for local standard-user checks'
}
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'Account bootstrap is only for disposable GitHub-hosted Windows runners'
}
$Repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = (Get-Command $Python -ErrorAction Stop).Source
$Git = (Get-Command git -ErrorAction Stop).Source
$ScratchParent = (Resolve-Path -LiteralPath $ScratchParent).Path
$Scratch = Join-Path $ScratchParent ('aimusic-standard-user-' + [guid]::NewGuid().ToString('N'))
if ($Scratch.StartsWith($Repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Scratch directory must be outside checkout'
}
$UserName = 'amci' + [guid]::NewGuid().ToString('N').Substring(0, 12)
$Password = [Security.SecureString]::new()
$Random = [Security.Cryptography.RandomNumberGenerator]::Create()
$Bytes = [byte[]]::new(48)
$User = $null
$OriginalAcls = @{}
try {
    # No plaintext password string, argv, environment, transcript, or credential file.
    foreach ($Character in 'aA1!'.ToCharArray()) { $Password.AppendChar($Character) }
    $Random.GetBytes($Bytes)
    $Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'
    foreach ($Byte in $Bytes) { $Password.AppendChar($Alphabet[$Byte % 64]) }
    $Password.MakeReadOnly()
    $User = New-LocalUser -Name $UserName -Password $Password -AccountExpires (Get-Date).AddHours(1) -UserMayNotChangePassword
    $Users = Get-LocalGroup -SID 'S-1-5-32-545'
    Add-LocalGroupMember -Group $Users.Name -Member $User
    $Credential = [pscredential]::new("$env:COMPUTERNAME\$UserName", $Password)
    New-Item -ItemType Directory -Path $Scratch | Out-Null
    # Grant read/execute only for source, interpreter and shell; no write grant.
    foreach ($Path in @($Repo, (Split-Path -Parent $Python), $PSHOME) | Select-Object -Unique) {
        $Acl = Get-Acl -LiteralPath $Path
        $OriginalAcls[$Path] = $Acl.Sddl
        $Rule = [Security.AccessControl.FileSystemAccessRule]::new($User.SID, 'ReadAndExecute', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $Acl.AddAccessRule($Rule)
        Set-Acl -LiteralPath $Path -AclObject $Acl
    }
    # Grant write access only to this fresh external scratch tree.
    $Acl = Get-Acl -LiteralPath $Scratch
    $Acl.SetAccessRuleProtection($true, $false)
    foreach ($Sid in @([Security.Principal.WindowsIdentity]::GetCurrent().User,
                       [Security.Principal.SecurityIdentifier]::new('S-1-5-18'))) {
        $Acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($Sid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow'))
    }
    $Acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($User.SID, 'Modify', 'ContainerInherit,ObjectInherit', 'None', 'Allow'))
    Set-Acl -LiteralPath $Scratch -AclObject $Acl
    . (Join-Path $PSScriptRoot 'windows-audio-standard-user-command.ps1')
    $Encoded = New-AudioStandardUserCommand -Script (Join-Path $PSScriptRoot 'test-windows-audio.ps1') -Python $Python -Scratch $Scratch -Git $Git
    $Stdout = Join-Path $Scratch 'standard-user.stdout.log'
    $Stderr = Join-Path $Scratch 'standard-user.stderr.log'
    # Base64 has no quoting/metacharacters. Credential API handles the SecureString;
    # UseNewEnvironment avoids inheriting runner auth/session variables into checks.
    $Child = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') -Credential $Credential -LoadUserProfile -UseNewEnvironment -WorkingDirectory $Scratch -ArgumentList @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $Encoded) -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -Wait -PassThru
    # Logs are not commands; disable workflow command processing while printing them.
    $LogToken = [guid]::NewGuid().ToString('N')
    Write-Host "::stop-commands::$LogToken"
    try {
        Get-Content -LiteralPath $Stdout -Encoding utf8
        Get-Content -LiteralPath $Stderr -Encoding utf8
    } finally { Write-Host "::$LogToken::" }
    Write-Host "Standard-user diagnostics retained outside checkout: $Scratch"
    if ($Child.ExitCode -ne 0) { throw "Standard-user wheel checks failed: $($Child.ExitCode)" }
} finally {
    # No broad recursive deletion: retain diagnostics on this disposable VM.
    try {
        if ($null -ne $User) { Remove-LocalUser -SID $User.SID }
    } finally {
        $Password.Dispose()
        $Random.Dispose()
        [Array]::Clear($Bytes, 0, $Bytes.Length)
        foreach ($Path in $OriginalAcls.Keys) {
            $Acl = Get-Acl -LiteralPath $Path
            $Acl.SetSecurityDescriptorSddlForm($OriginalAcls[$Path])
            Set-Acl -LiteralPath $Path -AclObject $Acl
        }
    }
}
