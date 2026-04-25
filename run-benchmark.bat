@echo off
REM DendryTerra Benchmark Runner (Batch File)
REM Usage: run-benchmark.bat [gridSize] [mode]
REM   mode: far (default) - PIXEL_RIVER vs FAR cache comparison
REM         all            - full benchmark suite
REM Example: run-benchmark.bat 128 far
REM Example: run-benchmark.bat 64 all

setlocal

REM Set default grid size, mode, and sample spacing
set GRID_SIZE=500
set MODE=far
set SPACING=4
if not "%1"=="" set GRID_SIZE=%1
if not "%2"=="" set MODE=%2
if not "%3"=="" set SPACING=%3

REM Set Java home
set JAVA_HOME=C:\JAVA\jdk-23

echo ============================================================
echo DendryTerra Benchmark Runner
echo ============================================================
echo.
echo Using JAVA_HOME: %JAVA_HOME%
echo Grid Size: %GRID_SIZE%x%GRID_SIZE%
echo Mode:      %MODE%
echo Spacing:   %SPACING% world units/sample
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
call gradlew.bat benchmark -PbenchmarkGrid=%GRID_SIZE% -PbenchmarkMode=%MODE% -PbenchmarkSpacing=%SPACING% --console=plain

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
