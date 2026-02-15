param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot/../..").Path
)

$ErrorActionPreference = "Stop"

Push-Location $RepoRoot
try {
    cargo build --release --manifest-path runtime_rust_dll/Cargo.toml

    $dllDir = Join-Path $RepoRoot "runtime_rust_dll\target\release"
    if (-not (Test-Path (Join-Path $dllDir "acestep_runtime.dll"))) {
        throw "acestep_runtime.dll not found at $dllDir"
    }

    $env:ACESTEP_RUNTIME_DLL = Join-Path $dllDir "acestep_runtime.dll"
    $env:PATH = "$dllDir;$env:PATH"

    python examples/ffi/python/regression.py
    dotnet run --project examples/ffi/csharp/ffi_demo.csproj -c Release

    Write-Host "ffi regression: PASS"
}
finally {
    Pop-Location
}
