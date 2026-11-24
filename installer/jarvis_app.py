"""Entry point of the installed app (Jarvis.exe, built by build_installer.bat)."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from jarvis.daemon import main
    main()
