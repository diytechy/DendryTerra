@echo off
REM DendryTerra Benchmark Runner (Batch File)
REM Usage: run-benchmark.bat [gridSize] [mode] [spacing] [avx]
REM   avx: 2 (default, AVX2) or 3 (AVX-512, only on supported CPUs)
REM Example: run-benchmark.bat 500 far 4 3

setlocal

REM Set defaults
set GRID_SIZE=500
set MODE=far
set SPACING=4
set AVX=2
if not "%1"=="" set GRID_SIZE=%1
if not "%2"=="" set MODE=%2
if not "%3"=="" set SPACING=%3
if not "%4"=="" set AVX=%4

REM Gradle daemon must run on JDK 23. JDK 25 has a Kotlin plugin version-parsing
REM bug (IllegalArgumentException: 25.0.1) when used as the daemon JVM in Gradle 8.14.2.
REM The benchmark process itself runs on JDK 25 via the Gradle toolchain launcher.
set JAVA_HOME=C:\JAVA\jdk-23

echo ============================================================
echo DendryTerra Benchmark Runner
echo ============================================================
echo.
echo Gradle daemon: %JAVA_HOME%
echo Benchmark JVM: C:\JAVA\jdk-25.0.1  (via Gradle toolchain)
echo Grid Size:   %GRID_SIZE%x%GRID_SIZE%
echo Mode:        %MODE%
echo Spacing:     %SPACING% world units/sample
echo AVX level:   %AVX%
echo.

REM Build the project
echo Building project...
call gradlew.bat build -x test
if errorlevel 1 (
    echo ERROR: Build failed!
    exit /b 1
)

echo.
echo Running benchmarks...
echo.

REM Run the benchmark (passed as Gradle project properties, no quoting issues)
call gradlew.bat benchmark -PbenchmarkGrid=%GRID_SIZE% -PbenchmarkMode=%MODE% -PbenchmarkSpacing=%SPACING% -PbenchmarkAVX=%AVX% --console=plain

if errorlevel 1 (
    echo.
    echo ============================================================
    echo Benchmark FAILED
    echo ============================================================
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo Benchmark completed successfully!
    echo ============================================================
)

endlocal
