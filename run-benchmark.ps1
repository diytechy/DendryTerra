# DendryTerra Benchmark Runner Script
# Executes benchmark tests comparing different DendrySampler configurations
#
# Usage: .\run-benchmark.ps1 [-GridSize 500] [-Mode far|all] [-Spacing 4] [-AVX 2|3]
#   AVX 2 (default): AVX2 256-bit SIMD
#   AVX 3:           AVX-512 512-bit SIMD (only on supported CPUs)

param(
    [int]$GridSize = 500,
    [string]$Mode = "far",
    [int]$Spacing = 4,
    [int]$AVX = 3,
    # Gradle daemon JDK — keep at 23 to avoid daemon issues with JDK 25.
    # The benchmark process itself runs on JDK 25 via the Gradle toolchain launcher.
    [string]$JavaHome = "C:/JAVA/jdk-23"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DendryTerra Benchmark Runner" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$env:JAVA_HOME = $JavaHome
Write-Host "Gradle JDK:    $env:JAVA_HOME" -ForegroundColor Yellow
Write-Host "Benchmark JDK: C:/JAVA/jdk-25.0.1  (via toolchain)" -ForegroundColor Yellow
Write-Host ""

# Verify the Gradle daemon JDK is present
$null = & "$env:JAVA_HOME/bin/java.exe" -version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Java not found at $env:JAVA_HOME" -ForegroundColor Red
    exit 1
}

Write-Host "Grid Size: ${GridSize}x${GridSize} = $($GridSize * $GridSize) samples" -ForegroundColor Green
Write-Host "Mode:      $Mode" -ForegroundColor Green
Write-Host "Spacing:   $Spacing world units/sample  (covers $($GridSize * $Spacing)x$($GridSize * $Spacing) world units)" -ForegroundColor Green
Write-Host "AVX level: $AVX" -ForegroundColor Green
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

# Run the benchmark (passed as Gradle project properties)
& .\gradlew.bat benchmark "-PbenchmarkGrid=$GridSize" "-PbenchmarkMode=$Mode" "-PbenchmarkSpacing=$Spacing" "-PbenchmarkAVX=$AVX" --console=plain

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
