; Inno Setup script: dist\Jarvis (from PyInstaller) -> dist\JarvisSetup.exe. Used by build_installer.bat.
; Installs for the current user only (no admin rights), into %LOCALAPPDATA%\Programs\Jarvis-0.
; Your data (%USERPROFILE%\.jarvis0: settings, history, corrections, memory index) is never removed.
#define AppVersion GetEnv("JARVIS_VERSION")
#if AppVersion == ""
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{7C3F2A51-8E44-4B7A-9A3D-JARVIS0000001}
AppName=Jarvis-0
AppVersion={#AppVersion}
AppPublisher=Jarvis-0
DefaultDirName={localappdata}\Programs\Jarvis-0
DefaultGroupName=Jarvis-0
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=JarvisSetup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayName=Jarvis-0
CloseApplications=yes

[Tasks]
Name: "autostart"; Description: "Start Jarvis when I sign in to Windows"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\Jarvis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\Jarvis-0"; Filename: "{app}\Jarvis.exe"
Name: "{userdesktop}\Jarvis-0"; Filename: "{app}\Jarvis.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Jarvis-0"; \
  ValueData: """{app}\Jarvis.exe"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\Jarvis.exe"; Description: "Start Jarvis now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "taskkill"; Parameters: "/IM Jarvis.exe /F"; Flags: runhidden; RunOnceId: "StopJarvis"
