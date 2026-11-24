@echo off
REM Phase 42: build dist\JarvisSetup.exe, one installer for a clean PC (your laptop): no Python, git or setup.
REM It contains Jarvis and your trained models (run quantize.bat first for smaller ones). ~5 minutes.
REM Needs Inno Setup for the single .exe:  winget install JRSoftware.InnoSetup
call .venv\Scripts\activate
echo [1/4] Checking this version first (tests + self-test)...
python -m pytest -q -x tests || ( echo Tests failed: not building an installer from a broken version. & exit /b 1 )
python -m jarvis.selftest || exit /b 1
echo [2/4] Building the app with PyInstaller...
python -c "import PyInstaller" 2>nul || pip install pyinstaller
python -m PyInstaller installer\jarvis.spec --noconfirm --distpath dist --workpath build\pyinstaller || exit /b 1
echo [3/4] Checking the built app starts...
start "" /wait dist\Jarvis\Jarvis.exe --selftest-exe
set RESULT=%ERRORLEVEL%
type "%TEMP%\jarvis_selftest.txt" 2>nul
if not "%RESULT%"=="0" ( echo The built app didn't pass its self-test, so no installer was made. & exit /b 1 )
for /f %%v in ('git describe --tags --always') do set JARVIS_VERSION=%%v
echo [4/4] Making the installer...
set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
  echo Inno Setup isn't installed, so here's the app folder instead: dist\Jarvis  ^(copy it to the other PC^)
  echo For a single JarvisSetup.exe:  winget install JRSoftware.InnoSetup   and run this again.
  exit /b 0
)
%ISCC% /Q installer\jarvis.iss || exit /b 1
echo Done: dist\JarvisSetup.exe  ^(copy it to the laptop and run it; no admin rights needed^)
