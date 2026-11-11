@echo off
REM Builds and tests the Python extension.

setlocal

echo === Building Python extension ===
python setup.py build_ext --inplace --force || exit /b 1

echo.
echo === Tensor binding tests ===
python -u test_causa.py || exit /b 1

echo.
echo === Causal binding tests ===
python -u test_causa_scm.py || exit /b 1

endlocal
