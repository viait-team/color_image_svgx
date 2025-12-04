Here is the technical design document `image-to-svgx-tools-setup.md`. It covers the implementation strategy for integrating **pngquant**, **ImageMagick**, **potrace**, and **tesseract** into your VS Code OSS fork using the "Script-Based Setup" flow.

***

# Image-to-SVGX Toolchain Integration Guide

**Document:** `image-to-svgx-tools-setup.md`
**Purpose:** Define the architecture and implementation steps to install external dependencies (`pngquant`, `ImageMagick`, `potrace`, `tesseract`) within the VS Code OSS forked repository.
**Strategy:** Lazy-loading (On-demand download via Terminal Scripts).

---

## 1. Architecture Overview

To avoid bloating the repository size, we will not commit binary files. Instead, we will implement a **"Setup Wizard"** workflow.

1.  **Hosting:** Pre-compiled binaries for Windows, macOS, and Linux are hosted on GitHub Releases (or an S3 bucket).
2.  **Scripts:** Native OS scripts (`.ps1` for Windows, `.sh` for Unix) handle downloading, extracting, and permission setting.
3.  **Storage:** Tools are installed into the Extension's **Global Storage** (`context.globalStorageUri`), keeping them isolated from system directories.
4.  **Entry Points:** The setup is triggered via the Welcome Page, Command Palette, or auto-detection notifications.

---

## 2. Directory Structure

Inside your built-in extension folder (e.g., `extensions/image-to-svgx/`):

```text
extensions/image-to-svgx/
├── package.json              # Defines Commands & Welcome Page
├── src/
│   └── extension.ts          # Main logic to trigger setup
└── scripts/                  # Native Install Scripts
    ├── install.ps1           # Windows Installer
    └── install.sh            # macOS / Linux Installer
```

---

## 3. Asset Preparation (The Binaries)

Before coding, you must package the tools. Create three archives and upload them to your repository's Release page (e.g., tag `v1.0.0-assets`).

**Archive Contents (`tools-win32.zip`, `tools-mac.zip`, `tools-linux.zip`):**
*   `bin/tesseract` (executable)
*   `bin/potrace` (executable)
*   `bin/pngquant` (executable)
*   `bin/magick` (ImageMagick executable)
*   `data/tessdata/` (Tesseract language models)

---

## 4. Implementation Steps

### Step A: Native Setup Scripts

These scripts run in the user's terminal to provide visual feedback during installation.

#### 1. Windows Script (`scripts/install.ps1`)
```powershell
# scripts/install.ps1
param([string]$InstallPath)

Write-Host ">>> INITIALIZING IMAGE-TO-SVGX TOOLCHAIN <<<" -ForegroundColor Cyan
Write-Host "Target: $InstallPath"

# 1. Configuration
$BaseUrl = "https://github.com/YOUR-ORG/YOUR-REPO/releases/download/v1.0.0-assets"
$ZipUrl = "$BaseUrl/tools-win32.zip"
$ZipFile = Join-Path $InstallPath "tools.zip"

# 2. Download
Write-Host "Downloading binaries (pngquant, magick, potrace, tesseract)..."
try {
    Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipFile -ErrorAction Stop
} catch {
    Write-Error "Download Failed: $_"
    exit 1
}

# 3. Extract
Write-Host "Extracting..."
Expand-Archive -Path $ZipFile -DestinationPath $InstallPath -Force
Remove-Item $ZipFile -Force

Write-Host ">>> SETUP COMPLETE successfully. <<<" -ForegroundColor Green
```

#### 2. macOS & Linux Script (`scripts/install.sh`)
```bash
#!/bin/bash
# scripts/install.sh

INSTALL_PATH="$1"
OS_TYPE=$(uname -s)

echo ">>> INITIALIZING IMAGE-TO-SVGX TOOLCHAIN <<<"
echo "Target: $INSTALL_PATH"

# 1. OS Detection
if [ "$OS_TYPE" == "Darwin" ]; then
    URL="https://github.com/YOUR-ORG/YOUR-REPO/releases/download/v1.0.0-assets/tools-mac.zip"
elif [ "$OS_TYPE" == "Linux" ]; then
    URL="https://github.com/YOUR-ORG/YOUR-REPO/releases/download/v1.0.0-assets/tools-linux.zip"
else
    echo "Unsupported OS: $OS_TYPE"; exit 1
fi

# 2. Setup Dir
mkdir -p "$INSTALL_PATH"
cd "$INSTALL_PATH" || exit

# 3. Download
echo "Downloading tools..."
if command -v curl >/dev/null; then curl -L -o tools.zip "$URL"
elif command -v wget >/dev/null; then wget -O tools.zip "$URL"
else echo "Error: curl/wget not found."; exit 1; fi

# 4. Extract
unzip -o tools.zip
rm tools.zip

# 5. Permissions & Quarantine (Mac)
echo "Setting permissions..."
chmod +x ./bin/*

if [ "$OS_TYPE" == "Darwin" ]; then
    echo "Bypassing MacOS Quarantine..."
    xattr -d com.apple.quarantine ./bin/* 2>/dev/null || true
fi

echo ">>> SETUP COMPLETE successfully. <<<"
```

---

### Step B: Extension Logic (TypeScript)

In `src/extension.ts`, implement the command that launches these scripts in the integrated terminal.

```typescript
import * as vscode from 'vscode';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';

export function activate(context: vscode.ExtensionContext) {

    // 1. Register the Setup Command
    let setupCmd = vscode.commands.registerCommand('image2svgx.setupTools', async () => {
        const platform = os.platform();
        const scriptsDir = path.join(context.extensionPath, 'scripts');
        const installDir = context.globalStorageUri.fsPath;

        // Ensure storage exists
        await vscode.workspace.fs.createDirectory(context.globalStorageUri);

        // Create Terminal
        const term = vscode.window.createTerminal({
            name: "Toolchain Installer",
            iconPath: new vscode.ThemeIcon("cloud-download")
        });
        term.show();

        if (platform === 'win32') {
            const script = path.join(scriptsDir, 'install.ps1');
            // PowerShell Bypass is required for scripts downloaded or local
            term.sendText(`powershell -ExecutionPolicy Bypass -File "${script}" "${installDir}"`);
        } else {
            const script = path.join(scriptsDir, 'install.sh');
            term.sendText(`chmod +x "${script}"`);
            term.sendText(`"${script}" "${installDir}"`);
        }
    });

    context.subscriptions.push(setupCmd);

    // 2. Run Auto-Check on Startup
    checkForTools(context);
}

function checkForTools(context: vscode.ExtensionContext) {
    const installDir = context.globalStorageUri.fsPath;
    // Check for one key tool to verify installation
    const testTool = os.platform() === 'win32' ? 'tesseract.exe' : 'tesseract';
    const toolExists = fs.existsSync(path.join(installDir, 'bin', testTool));

    if (!toolExists) {
        vscode.window.showInformationMessage(
            "Image-to-SVGX tools are missing. Please run the setup.",
            "Run Setup Now"
        ).then(selection => {
            if (selection === "Run Setup Now") {
                vscode.commands.executeCommand('image2svgx.setupTools');
            }
        });
    }
}
```

---

### Step C: Configuration & Entry Points

Update `package.json` to expose the setup via the **Welcome Page** and **Command Palette**.

```json
{
  "contributes": {
    "commands": [
      {
        "command": "image2svgx.setupTools",
        "title": "Image2SVGX: Install External Tools",
        "category": "Image2SVGX"
      }
    ],
    "walkthroughs": [
      {
        "id": "image2svgx.gettingStarted",
        "title": "Get Started with Image to SVG",
        "description": "Prepare your environment for chart conversion.",
        "steps": [
          {
            "id": "install.tools",
            "title": "Install Toolchain",
            "description": "We need to download **pngquant**, **ImageMagick**, **potrace**, and **Tesseract**. Click below to open the terminal and run the installer.",
            "media": { "svg": "media/setup.svg", "altText": "Setup" },
            "button": {
              "title": "Run Installer Script",
              "command": "image2svgx.setupTools"
            }
          }
        ]
      }
    ]
  }
}
```

---

## 5. Usage in Processing Script

When running your actual Python processing script (the one that does the conversion), pass the tool paths dynamically.

**TypeScript Command to Run Conversion:**
```typescript
const storagePath = context.globalStorageUri.fsPath;
const binPath = path.join(storagePath, 'bin');

// Add bin to PATH environment variable for the Python process
const newEnv = {
    ...process.env,
    PATH: `${binPath}${path.delimiter}${process.env.PATH}`
};

// Now python can call 'tesseract' or 'magick' directly
cp.execFile('python', ['convert_script.py', '--input', 'image.png'], { env: newEnv }, ...);
```
