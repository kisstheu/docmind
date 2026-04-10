[CmdletBinding()]
param(
    [Parameter(ValueFromPipeline = $true)]
    [AllowNull()]
    [string]$InputObject,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

begin {
    $ErrorActionPreference = "Stop"
    $stdinLines = New-Object System.Collections.Generic.List[string]
}

process {
    if ($PSBoundParameters.ContainsKey("InputObject")) {
        $stdinLines.Add([string]$InputObject)
    }
}

end {
    chcp 65001 > $null
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [Console]::OutputEncoding = $utf8NoBom
    [Console]::InputEncoding = $utf8NoBom
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $pythonExe = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "Python venv not found: $pythonExe"
    }

    $stdinText = $null
    if ($stdinLines.Count -gt 0) {
        $stdinText = [string]::Join(
            [Environment]::NewLine,
            $stdinLines
        )
        if (-not $stdinText.EndsWith([Environment]::NewLine)) {
            $stdinText += [Environment]::NewLine
        }
    }

    if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
        if ($null -ne $stdinText) {
            $PythonArgs = @("-")
        } else {
            throw "Usage: .\\bootstrap\\python_utf8.ps1 <python args> or pipe UTF-8 stdin into it."
        }
    }

    $stdinFile = $null
    try {
        if ($null -ne $stdinText) {
            $stdinFile = [System.IO.Path]::GetTempFileName()
            Set-Content -Path $stdinFile -Encoding utf8 -Value $stdinText
            $proc = Start-Process `
                -FilePath $pythonExe `
                -ArgumentList $PythonArgs `
                -RedirectStandardInput $stdinFile `
                -NoNewWindow `
                -PassThru `
                -Wait
            exit $proc.ExitCode
        }

        & $pythonExe @PythonArgs
        if ($null -ne $LASTEXITCODE) {
            exit $LASTEXITCODE
        }
        exit 0
    }
    finally {
        if ($stdinFile -and (Test-Path $stdinFile)) {
            Remove-Item -Path $stdinFile -Force -ErrorAction SilentlyContinue
        }
    }
}
