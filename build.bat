@echo off
REM Builds the causa library and runs the C test suite.
REM Requires Visual Studio Build Tools (vcvarsall) on PATH.

setlocal

if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
call "%VCVARS%" x64 || exit /b 1

echo === Building build tool ===
cl /nologo /O2 /Febuild.exe build.c || exit /b 1

echo === Building library ===
build.exe --mode release --target static --clean || exit /b 1

echo === Building tests ===
cl /nologo /O2 /Iinclude /Febuild\causa_test.exe test\test_causa.c build\causa.lib || exit /b 1

echo === Running tests ===
build\causa_test.exe
if errorlevel 1 exit /b 1

endlocal
