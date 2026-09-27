"""Launch installed SIGMA.exe, verify window process, take PowerShell screenshot, and verify GUI."""
import os
import sys
import time
import subprocess

def main():
    exe_path = os.path.abspath(r"installed_app\SIGMA.exe")
    print(f"Launching executable: {exe_path}")
    
    if not os.path.exists(exe_path):
        print(f"ERROR: File not found at {exe_path}")
        sys.exit(1)
        
    proc = subprocess.Popen([exe_path])
    print(f"Process launched with PID: {proc.pid}")
    
    # Wait for GUI main window to open
    time.sleep(4)
    
    poll = proc.poll()
    if poll is not None:
        print(f"ERROR: Process exited with code {poll}")
        sys.exit(1)
    else:
        print("SUCCESS: SIGMA.exe is running in GUI mode!")
        
    # Take screenshot via PowerShell
    ps_cmd = (
        'Add-Type -AssemblyName System.Windows.Forms; '
        'Add-Type -AssemblyName System.Drawing; '
        '$Screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; '
        '$Bitmap = New-Object System.Drawing.Bitmap $Screen.Width, $Screen.Height; '
        '$Graphic = [System.Drawing.Graphics]::FromImage($Bitmap); '
        '$Graphic.CopyFromScreen($Screen.Left, $Screen.Top, 0, 0, $Bitmap.Size); '
        '$Bitmap.Save("C:\\Users\\Sumit Mahaseth\\.gemini\\antigravity-ide\\brain\\a22d9701-b92b-4339-ad87-e987cf7e455f\\sigma_installed_window.png")'
    )
    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
    print("PowerShell screenshot stdout:", res.stdout)
    print("PowerShell screenshot stderr:", res.stderr)
    
    time.sleep(1)
    
    # Clean up process
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        
    print("Verification completed cleanly.")

if __name__ == "__main__":
    main()
