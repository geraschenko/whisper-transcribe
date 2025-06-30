#!/usr/bin/env python3

import sys
import os
import signal
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from logging.handlers import RotatingFileHandler

from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QActionGroup
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPainter


def setup_logging(config_dir: Path) -> logging.Logger:
    """Setup global logger with automatic rotation"""
    config_dir.mkdir(exist_ok=True)
    log_file = config_dir / "whisper-transcribe.log"

    logger = logging.getLogger("whisper_transcribe")
    logger.setLevel(logging.INFO)

    # RotatingFileHandler: max 1MB per file, keep 3 backup files
    # This gives us ~4MB total (current + 3 backups)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=3,  # Keep 3 old files (.1, .2, .3)
    )

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logging(Path.home() / ".config" / "whisper-transcribe")


def load_preferred_device_id(config_file_path: Path) -> int:
    """Load preferred device ID from config file, return -1 if not found"""
    if not config_file_path.exists():
        return -1

    try:
        with config_file_path.open("r") as f:
            config = json.load(f)
            return config.get("preferred_device_id", -1)
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return -1


def save_preferred_device_id(device_id: int, config_file_path: Path) -> bool:
    """Save preferred device ID to config file, return success status"""
    try:
        config_data = {"preferred_device_id": device_id}
        with config_file_path.open("w") as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        logger.warning(f"Could not save config: {e}")
        return False


def get_active_device_id(
    preferred_device_id: int, audio_devices: Dict[int, str]
) -> int:
    """Get the device ID to actually use (preferred if available, else -1)"""
    if preferred_device_id >= 0:
        # Check if preferred device is available
        if preferred_device_id in audio_devices:
            return preferred_device_id
    return -1  # Fall back to default device


def build_transcribe_command(script_dir: Path, device_id: int) -> str:
    """Build the transcription command with optional device selection"""
    transcribe_cmd = "./build/transcribe"
    if device_id >= 0:
        transcribe_cmd += f" --capture {device_id}"
    return f"cd '{script_dir}' && {transcribe_cmd} | while IFS= read -r line; do printf '%s ' \"$line\" | xdotool type --clearmodifiers --file -; done"


def prepare_device_menu_items(
    audio_devices: Dict[int, str], preferred_device_id: int
) -> List[Dict[str, Any]]:
    """Prepare device menu item data"""
    menu_items = []

    # Add device options.
    for device_id, device_name in audio_devices.items():
        display_name = device_name
        if device_id == preferred_device_id:
            display_name += " ⭐"

        menu_items.append(
            {
                "type": "device",
                "device_id": device_id,
                "display_name": display_name,
                "is_checked": device_id == preferred_device_id,
            }
        )

    # Add separator if we have devices.
    if audio_devices:
        menu_items.append({"type": "separator"})

    # Add default option.
    default_display = "Use Default Device"
    if preferred_device_id == -1:
        default_display += " ⭐"

    menu_items.append(
        {
            "type": "device",
            "device_id": -1,
            "display_name": default_display,
            "is_checked": preferred_device_id == -1,
        }
    )

    # Add refresh option.
    menu_items.append({"type": "separator"})
    menu_items.append({"type": "refresh", "display_name": "🔄 Refresh Devices"})

    return menu_items


def fetch_audio_devices(transcribe_binary_path: Path) -> Dict[int, str]:
    """Fetch audio devices from transcribe binary, return device mapping (empty on error)."""
    try:
        result = subprocess.run(
            [str(transcribe_binary_path), "--list-devices"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error detecting audio devices: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error detecting audio devices: {e}")
        return {}

    devices = {}
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            parts = line.split(":", 1)
            if len(parts) == 2:
                device_id = int(parts[0].strip())
                device_name = parts[1].strip()
                devices[device_id] = device_name
    return devices


def log_device_status(audio_devices: Dict[int, str], preferred_device_id: int) -> None:
    """Log device status information"""
    logger.info(f"Found {len(audio_devices)} audio devices:")
    active_device_id = get_active_device_id(preferred_device_id, audio_devices)

    for device_id, device_name in audio_devices.items():
        markers = []
        if device_id == preferred_device_id:
            markers.append("preferred")
        if device_id == active_device_id:
            markers.append("active")
        marker_text = f" ({', '.join(markers)})" if markers else ""
        logger.info(f"  {device_id}: {device_name}{marker_text}")

    # Show fallback info if preferred device not available
    if preferred_device_id >= 0 and active_device_id == -1:
        logger.info(
            f"  Note: Preferred device {preferred_device_id} not available, using default"
        )


class TranscriptionApp(QObject):
    # Signal for thread-safe communication with Qt
    toggle_requested = pyqtSignal()

    def __init__(self, script_dir: Path, app: QApplication):
        super().__init__()

        # Store injected dependencies
        self.app = app
        self.script_dir = script_dir
        
        # Construct and validate transcribe binary path
        self.transcribe_binary = script_dir / "build" / "transcribe"
        if not self.transcribe_binary.exists():
            logger.error(f"Transcribe binary not found: {self.transcribe_binary}")
            sys.exit(1)
        
        # Ensure script directory exists
        self.script_dir.mkdir(exist_ok=True)
        
        # Derive other paths from script directory
        self.pid_file = self.script_dir / "app.pid"
        self.config_file = self.script_dir / "config.json"

        # State
        self.transcribing = False
        self.transcribe_process = None
        self.tray_icon = None
        self.audio_devices = {}
        self.preferred_device_id = -1

        # Load configuration and detect audio devices
        self.preferred_device_id = load_preferred_device_id(self.config_file)
        self.detect_audio_devices()

        # Setup signal handling
        self.setup_signal_handlers()
        self.setup_tray()

        # Connect internal signal
        self.toggle_requested.connect(self.toggle_transcription)

        # Write PID file
        self.write_pid_file()

        logger.info("Whisper transcription app started")
        logger.info(f"PID: {os.getpid()}")
        logger.info("Use whisper-transcribe-toggle to control")

    def create_icon(self, active=False):
        """Create system tray icon."""
        # Try to use system theme icons first
        icon_name = (
            "microphone-sensitivity-high" if active else "microphone-sensitivity-muted"
        )
        base_icon = QIcon.fromTheme(icon_name)

        if base_icon.isNull():
            # Fallback to colored squares if theme icons not available
            pixmap = QPixmap(16, 16)
            color = QColor(255, 0, 0) if active else QColor(128, 128, 128)
            pixmap.fill(color)
            return QIcon(pixmap)

        # Colorize the theme icon
        pixmap = base_icon.pixmap(16, 16)

        if active:
            # Apply green tint for active state
            colored_pixmap = QPixmap(pixmap.size())
            colored_pixmap.fill(QColor(255, 0, 0, 180))  # Semi-transparent red
            colored_pixmap.setMask(pixmap.createMaskFromColor(QColor(0, 0, 0, 0)))

            # Composite the colored overlay onto the original
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceAtop)
            painter.drawPixmap(0, 0, colored_pixmap)
            painter.end()

        return QIcon(pixmap)

    def detect_audio_devices(self):
        """Detect available audio capture devices"""
        self.audio_devices = fetch_audio_devices(self.transcribe_binary)
        log_device_status(self.audio_devices, self.preferred_device_id)

    def set_audio_device(self, device_id):
        """Set the preferred audio device (user action - saves to config)"""
        # Refresh device list to get current state
        self.detect_audio_devices()

        # Update preference and save config
        self.preferred_device_id = device_id
        save_preferred_device_id(self.preferred_device_id, self.config_file)
        logger.info(f"User selected audio device: {device_id}")

        # Update menu to reflect changes
        menu = self.create_context_menu()
        self.tray_icon.setContextMenu(menu)

    def setup_tray(self):
        """Setup system tray icon and menu"""
        self.tray_icon = QSystemTrayIcon(self)
        self.update_tray_icon()

        # Set full context menu (required for icon visibility)
        menu = self.create_context_menu()
        self.tray_icon.setContextMenu(menu)

        self.tray_icon.show()

    def create_context_menu(self):
        """Create the system tray context menu and return it"""
        menu = QMenu()

        # Toggle transcription action
        toggle_action = QAction("Toggle Transcription", self)
        toggle_action.triggered.connect(self.toggle_transcription)
        menu.addAction(toggle_action)

        menu.addSeparator()

        # Audio input device submenu
        device_menu = QMenu("Audio Input Device", menu)
        self.populate_device_menu(device_menu)
        menu.addMenu(device_menu)

        menu.addSeparator()

        # Quit action
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_app)
        menu.addAction(quit_action)

        return menu

    def populate_device_menu(self, device_menu):
        """Populate the device selection submenu"""
        device_menu.clear()

        # Create action group for radio button behavior
        device_group = QActionGroup(self)
        device_group.setExclusive(True)

        # Get menu item data
        menu_items = prepare_device_menu_items(
            self.audio_devices, self.preferred_device_id
        )

        # Create Qt menu items from data
        for item in menu_items:
            if item["type"] == "separator":
                device_menu.addSeparator()
            elif item["type"] == "device":
                action = QAction(item["display_name"], self)
                action.setCheckable(True)
                action.setChecked(item["is_checked"])
                action.triggered.connect(
                    lambda _checked, dev_id=item["device_id"]: self.set_audio_device(
                        dev_id
                    )
                )
                device_group.addAction(action)
                device_menu.addAction(action)
            elif item["type"] == "refresh":
                refresh_action = QAction(item["display_name"], self)
                refresh_action.triggered.connect(self.refresh_and_update_menu)
                device_menu.addAction(refresh_action)

    def refresh_and_update_menu(self):
        """Refresh devices and update the menu"""
        self.detect_audio_devices()
        # Update the context menu with refreshed device list
        menu = self.create_context_menu()
        self.tray_icon.setContextMenu(menu)

    def setup_signal_handlers(self):
        """Setup Unix signal handlers"""

        def signal_handler(_signum, _frame):
            # Emit Qt signal for thread-safe handling
            self.toggle_requested.emit()

        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGINT, self.signal_quit)
        signal.signal(signal.SIGTERM, self.signal_quit)

    def signal_quit(self, signum, _frame):
        logger.info(f"Received signal {signum}, quitting...")
        self.quit_app()

    def write_pid_file(self):
        try:
            with self.pid_file.open("w") as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logger.warning(f"Could not write PID file: {e}")

    def cleanup_pid_file(self):
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not remove PID file: {e}")

    def update_tray_icon(self):
        """Update tray icon based on current state"""
        self.tray_icon.setIcon(self.create_icon(self.transcribing))
        status = "ACTIVE" if self.transcribing else "INACTIVE"
        self.tray_icon.setToolTip(f"Whisper Transcribe - {status}")


    def toggle_transcription(self):
        """Toggle transcription on/off."""
        if self.transcribing:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self):
        """Start transcription process"""
        if self.transcribing:
            return

        logger.info("Starting transcription...")

        try:
            # Build command with optional device selection
            active_device_id = get_active_device_id(
                self.preferred_device_id, self.audio_devices
            )
            if active_device_id >= 0:
                logger.info(f"Using audio device: {active_device_id}")
            else:
                logger.info("Using default audio device")

            # Start the transcription pipeline in a subprocess
            # We use shell=True to handle the pipeline properly
            cmd = build_transcribe_command(self.script_dir, active_device_id)

            self.transcribe_process = subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setsid,  # Create new process group
            )

            self.transcribing = True
            self.update_tray_icon()

            logger.info(f"Transcription started, PID: {self.transcribe_process.pid}")

        except Exception as e:
            logger.error(f"Error starting transcription: {e}")

    def stop_transcription(self):
        """Stop transcription process"""
        if not self.transcribing or not self.transcribe_process:
            return

        logger.info("Stopping transcription...")

        try:
            # Kill the entire process group
            os.killpg(os.getpgid(self.transcribe_process.pid), signal.SIGTERM)

            # Wait briefly for graceful shutdown
            try:
                self.transcribe_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(self.transcribe_process.pid), signal.SIGKILL)
                self.transcribe_process.wait()

            self.transcribe_process = None
            self.transcribing = False
            self.update_tray_icon()

            logger.info("Transcription stopped")

        except Exception as e:
            logger.error(f"Error stopping transcription: {e}")
            # Force reset state
            self.transcribe_process = None
            self.transcribing = False
            self.update_tray_icon()

    def quit_app(self):
        """Quit the application"""
        logger.info("Quitting application...")

        # Stop any running transcription
        if self.transcribing:
            self.stop_transcription()

        # Cleanup
        self.cleanup_pid_file()
        self.tray_icon.hide()

        # Quit Qt application
        self.app.quit()

    def run(self):
        """Run the application"""
        # Timer to allow Unix signal handling in Qt event loop
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(100)  # Check for signals every 100ms
        
        try:
            sys.exit(self.app.exec_())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
            self.quit_app()


def main() -> None:
    # Create QApplication
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        logger.error("System tray not available")
        sys.exit(1)
    
    # Run the transcription app
    script_dir = Path(__file__).parent
    transcription_app = TranscriptionApp(script_dir, app)
    transcription_app.run()


if __name__ == "__main__":
    main()
