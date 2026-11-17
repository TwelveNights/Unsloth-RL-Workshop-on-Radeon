# Replaces $HOME\local-ai-workshop\unsloth\workshop with the unsloth-studio-workshop branch
# and puts an "Update Workshop" shortcut on the desktop that runs a saved copy of this script.
# Run on a workshop machine with:
#   irm https://raw.githubusercontent.com/TwelveNights/Unsloth-RL-Workshop-on-Radeon/unsloth-studio-workshop/install.ps1 | iex
& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'  # the progress bar makes Invoke-WebRequest very slow on PowerShell 5.1
    [Net.ServicePointManager]::SecurityProtocol = 'Tls12'

    # A mirror that serves this script rewrites these two lines to point at itself.
    $url = 'https://raw.githubusercontent.com/TwelveNights/Unsloth-RL-Workshop-on-Radeon/unsloth-studio-workshop/install.ps1'
    $zipUrl = 'https://github.com/TwelveNights/Unsloth-RL-Workshop-on-Radeon/archive/refs/heads/unsloth-studio-workshop.zip'
    $d = "$HOME\local-ai-workshop\unsloth"
    $z = "$env:TEMP\unsloth-workshop.zip"
    $x = "$d\Unsloth-RL-Workshop-on-Radeon-unsloth-studio-workshop"

    # Git marks its object files read-only and .git hidden, and Remove-Item and robocopy's
    # purge refuse those, so clear the flags on everything under a folder first.
    function Clear-Flags($p) {
        $flags = [IO.FileAttributes]'ReadOnly, Hidden, System'
        Get-ChildItem $p -Recurse -Force | ForEach-Object { $_.Attributes = $_.Attributes -band -bnot $flags }
    }
    function Remove-Folder($p) {
        if (-not (Test-Path $p)) { return }
        Clear-Flags $p
        Remove-Item -Recurse $p
    }

    # Download first, so a failed download leaves the old folder in place.
    Invoke-WebRequest $zipUrl -OutFile $z
    Remove-Folder $x
    Remove-Folder "$d\workshop.old"  # left behind by earlier versions of this script
    Expand-Archive $z $d
    Remove-Item $z

    # Mirror into the existing folder rather than replacing it: Jupyter's server and kernels
    # use it as their working directory, which blocks renaming or deleting the folder itself
    # but not overwriting the files in it. /MIR also deletes files that are not in the branch.
    if (Test-Path "$d\workshop") { Clear-Flags "$d\workshop" }
    robocopy $x "$d\workshop" /MIR /IS /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying into $d\workshop failed (robocopy exit code $LASTEXITCODE)." }
    Remove-Folder $x

    # The shortcut runs a saved copy of this script. A shortcut that downloads a script and
    # runs it straight from memory is what malware shortcuts do, and Defender flags it.
    $u = "$d\update-workshop.ps1"
    Invoke-WebRequest $url -OutFile $u
    $lnk = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Update Workshop.lnk")
    $lnk.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $lnk.Arguments = "-NoExit -NoProfile -ExecutionPolicy RemoteSigned -File `"$u`""
    $lnk.Description = 'Replace the workshop folder with the latest version from GitHub'
    $lnk.Save()

    Write-Host "Workshop installed to $d\workshop"
}
