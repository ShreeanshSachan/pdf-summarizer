// main.js (at the root of your project: pdf-summarizer/main.js)
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1000, // Adjust as needed
        height: 800, // Adjust as needed
        minWidth: 600, // Minimum width
        minHeight: 500, // Minimum height
        webPreferences: {
            // Ensure these paths are correct relative to where index.html is loaded
            preload: path.join(__dirname, 'frontend', 'preload.js'), // If you use a preload script
            nodeIntegration: true, // Be cautious with nodeIntegration in production
            contextIsolation: false, // For simpler initial setup, but consider true with preload
            webSecurity: false // Required for file:// access to local resources, be cautious
        }
    });

    // Load the index.html of the app from the 'frontend' directory
    mainWindow.loadFile(path.join(__dirname, 'frontend', 'index.html'));

    // Open the DevTools (useful for debugging during development)
    // mainWindow.webContents.openDevTools();

    mainWindow.on('closed', () => {
        mainWindow = null;
        // Terminate the Python process when the Electron app closes
        if (pythonProcess) {
            console.log('Terminating Python backend process...');
            // Use 'SIGINT' for graceful shutdown, 'SIGKILL' for forceful
            pythonProcess.kill('SIGINT');
            pythonProcess = null;
        }
    });
}

// Function to start the Python backend
function startPythonBackend() {
    let pythonExecutable;
    let backendArgs = [];
    let backendCwd;

    if (app.isPackaged) {
        // Production mode (after electron-builder packaging)
        // The PyInstaller --onedir output is a directory named 'main'
        // electron-builder copies 'backend/dist' into 'resources/app.asar.unpacked/backend/dist'
        pythonExecutable = path.join(process.resourcesPath, 'app.asar.unpacked', 'backend', 'dist', 'main', 'main.exe');
        backendCwd = path.join(process.resourcesPath, 'app.asar.unpacked', 'backend', 'dist', 'main'); // Directory containing main.exe
        // No additional args needed for the PyInstaller executable itself, as they are embedded
        backendArgs = []; // PyInstaller executable is self-contained and runs main.py logic
    } else {
        // Development mode (npm start)
        pythonExecutable = path.join(__dirname, 'backend', 'venv', 'Scripts', 'python.exe');
        backendCwd = path.join(__dirname, 'backend'); // Directory containing main.py
        backendArgs = [
            path.join(__dirname, 'backend', 'main.py'), // Path to main.py
            '--host', '0.0.0.0',
            '--port', '8000',
            '--no-reload' // Explicitly disable reload for stability
        ];
    }

    console.log(`Attempting to start Python backend from: ${pythonExecutable} with args: ${backendArgs.join(' ')}`);

    pythonProcess = spawn(pythonExecutable, backendArgs, {
        cwd: backendCwd,
        env: {
            ...process.env,
            GOOGLE_API_KEY: process.env.GOOGLE_API_KEY || '',
            OPENAI_API_KEY: process.env.OPENAI_API_KEY || '' // Pass OpenAI API key
        }
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        if (code !== 0 && code !== null) { // Handle unexpected exits
            console.error('Python backend crashed!');
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.executeJavaScript(`alert('The backend service has stopped unexpectedly. Please restart the application.');`);
            }
        }
    });

    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.executeJavaScript(`alert('Failed to start backend: ${err.message}. Please ensure Python and dependencies are installed correctly.');`);
        }
    });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.on('ready', () => {
    // Start the Python backend before creating the Electron window
    startPythonBackend();
    createWindow();
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    // On OS X it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (mainWindow === null) {
        createWindow();
    }
});

// Handle app quit event to ensure backend process is killed
app.on('before-quit', () => {
    if (pythonProcess) {
        console.log('Attempting to kill Python backend on app quit...');
        pythonProcess.kill('SIGINT');
        pythonProcess = null;
    }
});
