# Replaces $HOME\local-ai-workshop\unsloth\workshop with the unsloth-studio-workshop branch
# and puts an "Update Workshop" shortcut on the desktop that runs this script again.
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

    # Download first, so a failed download leaves the old folder in place.
    Invoke-WebRequest $zipUrl -OutFile $z
    foreach ($p in $x, "$d\workshop.old") { if (Test-Path $p) { Remove-Item -Recurse $p } }
    Expand-Archive $z $d
    Remove-Item $z

    # Rename before deleting: a rename fails as a whole if something (a Jupyter kernel, an
    # Explorer window) holds the folder open, where a delete would stop halfway.
    if (Test-Path "$d\workshop") {
        try { Rename-Item "$d\workshop" workshop.old }
        catch { Remove-Item -Recurse $x; throw "Close Jupyter and anything else using $d\workshop, then run the update again." }
    }
    Rename-Item $x workshop
    if (Test-Path "$d\workshop.old") { Remove-Item -Recurse "$d\workshop.old" }

    $lnk = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Update Workshop.lnk")
    $lnk.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $lnk.Arguments = "-NoExit -NoProfile -ExecutionPolicy Bypass -Command `"irm $url | iex`""
    $lnk.Description = 'Replace the workshop folder with the latest version from GitHub'
    $lnk.Save()

    Write-Host "Workshop installed to $d\workshop"
}
