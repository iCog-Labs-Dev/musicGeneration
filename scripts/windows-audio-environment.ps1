# Dot-source from the local/hosted harness. No process/global configuration changes.
function Enter-AudioTestEnvironment {
    $Names = @('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE', 'PYTHONUTF8',
        'AIMUSIC_RUN_FLUIDSYNTH_TESTS', 'AIMUSIC_FLUIDSYNTH_MANIFEST',
        'AIMUSIC_FLUIDSYNTH_ARTIFACTS', 'AIMUSIC_SOUNDFONT', 'AIMUSIC_AUDIO_BACKEND')
    $Saved = @{}
    foreach ($Name in $Names) {
        if (Test-Path -LiteralPath "Env:$Name") {
            $Saved[$Name] = (Get-Item -LiteralPath "Env:$Name").Value
        }
        # SetEnvironmentVariable(name, $null) can create an EMPTY variable in pwsh.
        # Empty SoundFont is invalid config; remove the key, do not relax validation.
        Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
    }
    $env:PYTHONUTF8 = '1'
    $env:AIMUSIC_AUDIO_BACKEND = 'native'
    return @{ Names = $Names; Saved = $Saved }
}

function Exit-AudioTestEnvironment([hashtable]$State) {
    foreach ($Name in $State.Names) {
        Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        if ($State.Saved.ContainsKey($Name)) {
            if ($State.Saved[$Name] -eq '') {
                # PS5's Env provider deletes on empty assignment. Preserve an
                # originally empty key when returning control to the caller.
                if (-not ('AudioTestEnvironmentNative' -as [type])) {
                    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class AudioTestEnvironmentNative {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetEnvironmentVariable(string name, string value);
    public static void RestoreEmpty(string name) {
        if (!SetEnvironmentVariable(name, ""))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
    }
}
"@
                }
                [AudioTestEnvironmentNative]::RestoreEmpty($Name)
            } else {
                Set-Item -LiteralPath "Env:$Name" -Value $State.Saved[$Name]
            }
        }
    }
}
