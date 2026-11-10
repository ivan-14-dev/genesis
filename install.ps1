# GENESIS one-command installer for Windows (PowerShell 5.1+):
#
#   irm https://github.com/ivan-14-dev/genesis/releases/latest/download/install.ps1 | iex
#
# It installs the latest release into its own virtual environment (%USERPROFILE%\.genesis\venv), adds a `genesis`
# command to the user PATH, runs `genesis init` (admin key, memory and web credentials generated on this machine,
# printed once and saved in %USERPROFILE%\.genesis\credentials.txt) and opens the web interface.
# Options are environment variables, set before running the line above:
#
#   $env:GENESIS_VERSION = "0.4.0"   install that release instead of the latest
#   $env:GENESIS_HOME    = "D:\genesis"   GENESIS folder
#   $env:GENESIS_PORT    = "8766"    web interface port
#   $env:GENESIS_SOURCE  = "C:\path\genesis_cognitive-0.4.0-py3-none-any.whl"   wheel, folder or URL to install
#   $env:GENESIS_NO_START = "1"      do not start the web interface at the end

$ErrorActionPreference = "Stop"
& {
    $Repo = "ivan-14-dev/genesis"
    $Version = if ($env:GENESIS_VERSION) { $env:GENESIS_VERSION } else { "latest" }
    $HomeDir = if ($env:GENESIS_HOME) { $env:GENESIS_HOME } else { Join-Path $HOME ".genesis" }
    $Port = if ($env:GENESIS_PORT) { $env:GENESIS_PORT } else { "8766" }
    $Source = $env:GENESIS_SOURCE
    if ($Port -notmatch '^\d+$') { throw "GENESIS_PORT must be a number" }

    # The Microsoft Store "python" alias exists but does not run: check that the interpreter really works.
    $Python = $null
    $PyArgs = @()
    foreach ($candidate in @("py -3", "python", "python3")) {
        $parts = $candidate -split " "
        if (-not (Get-Command $parts[0] -ErrorAction SilentlyContinue)) { continue }
        $extra = @($parts | Select-Object -Skip 1)
        $works = $false
        try {
            $ErrorActionPreference = "Continue"  # a probe writing to stderr must not abort the installer
            & $parts[0] @extra -c "import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 10))" 2>$null |
                Out-Null
            $works = ($LASTEXITCODE -eq 0)
        } catch {
            $works = $false
        } finally {
            $ErrorActionPreference = "Stop"
        }
        if ($works) { $Python = $parts[0]; $PyArgs = $extra; break }
    }
    if (-not $Python) { throw "Python 3.10 or newer (with venv) is required: https://www.python.org/downloads/" }

    if (-not $Source) {
        Write-Host "Looking up GENESIS release: $Version"
        if ($Version -eq "latest") {
            $Api = "https://api.github.com/repos/$Repo/releases/latest"
        } else {
            $Tag = if ($Version.StartsWith("v")) { $Version } else { "v$Version" }
            $Api = "https://api.github.com/repos/$Repo/releases/tags/$Tag"
        }
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor
            [Net.SecurityProtocolType]::Tls12
        $Release = Invoke-RestMethod -Uri $Api -Headers @{ "User-Agent" = "genesis-install" }
        $Wheel = $Release.assets | Where-Object { $_.name -like "*.whl" } | Select-Object -First 1
        if (-not $Wheel) { throw "release $($Release.tag_name) has no wheel asset" }
        $Source = $Wheel.browser_download_url
    }

    Write-Host "Installing GENESIS into $HomeDir"
    New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
    $Venv = Join-Path $HomeDir "venv"
    $VenvPython = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        & $Python @PyArgs -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment $Venv" }
    }
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    & $VenvPython -m pip install --quiet --upgrade --force-reinstall $Source
    if ($LASTEXITCODE -ne 0) { throw "pip could not install $Source" }

    # A small shim in its own folder: putting the venv's Scripts folder on PATH would shadow the system python.
    $BinDir = Join-Path $HomeDir "bin"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $Exe = Join-Path $Venv "Scripts\genesis.exe"
    Set-Content -Path (Join-Path $BinDir "genesis.cmd") -Encoding ASCII -Value "@`"$Exe`" %*"
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (($UserPath -split ";") -contains $BinDir)) {
        $NewPath = if ($UserPath) { "$UserPath;$BinDir" } else { $BinDir }
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
        Write-Host "Added $BinDir to your user PATH (open a new terminal to use 'genesis')."
    }
    if (-not (($env:Path -split ";") -contains $BinDir)) { $env:Path = "$env:Path;$BinDir" }

    & $Exe --version
    Write-Host ""
    & $Exe init --home $HomeDir --port $Port
    if ($LASTEXITCODE -ne 0) { throw "genesis init failed" }
    Write-Host ""
    if ($env:GENESIS_NO_START -eq "1") {
        Write-Host "Done. Start it with: genesis start --open"
        return
    }
    Write-Host "Starting the web interface (Ctrl+C to stop; later: genesis start --open)"
    & $Exe start --home $HomeDir --port $Port --open
}
