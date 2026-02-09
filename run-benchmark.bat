@echo off
REM DendryTerra Benchmark Runner (Batch File)
REM Usage: run-benchmark.bat [gridSize]
REM Example: run-benchmark.bat 128

setlocal

REM Set default grid size
set GRID_SIZE=64
if not "%1"=="" set GRID_SIZE=%1

REM Set Java home
set JAVA_HOME=C:\JAVA\jdk-23

echo ============================================================
echo DendryTerra Benchmark Runner
echo ============================================================
echo.
echo Using JAVA_HOME: %JAVA_HOME%
echo Grid Size: %GRID_SIZE%x%GRID_SIZE%
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

REM Run the benchmark
call gradlew.bat benchmark --args="%GRID_SIZE%" --console=plain

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
