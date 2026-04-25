# DendryTerra Benchmark Runner Script
# Executes benchmark tests comparing different DendrySampler configurations
#
# Usage: .\run-benchmark.ps1 [-GridSize 128] [-Mode far|all]
#   Mode 'far' (default): PIXEL_RIVER vs PIXEL_RIVER_FAR/FAR2 comparison
#   Mode 'all':           full benchmark suite

param(
    [int]$GridSize = 64,
    [string]$Mode = "far",
    [string]$JavaHome = "C:/JAVA/jdk-23"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DendryTerra Benchmark Runner" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Set Java home
$env:JAVA_HOME = $JavaHome
Write-Host "Using JAVA_HOME: $env:JAVA_HOME" -ForegroundColor Yellow
Write-Host ""

# Check if Java is available
$null = & "$env:JAVA_HOME/bin/java.exe" -version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Java not found at $env:JAVA_HOME" -ForegroundColor Red
    exit 1
}
Write-Host "Java detected successfully" -ForegroundColor Green

Write-Host "Grid Size: ${GridSize}x${GridSize} = $($GridSize * $GridSize) samples" -ForegroundColor Green
Write-Host "Mode:      $Mode" -ForegroundColor Green
Write-Host ""
Write-Host "Building project..." -ForegroundColor Yellow

# Build the project first
& .\gradlew.bat build -x test
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Running benchmarks..." -ForegroundColor Yellow
Write-Host ""

# Run the benchmark using the dedicated benchmark task
& .\gradlew.bat benchmark --args="$GridSize $Mode" --console=plain

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Benchmark completed successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "Benchmark failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    exit $LASTEXITCODE
}
