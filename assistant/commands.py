"""Built-in local command handlers for device and system operations."""

import datetime
import subprocess
from utils.logger import logger


class CommandHandler:
    """Executes local system actions and utility tasks."""

    def execute(self, intent_name: str, parameters: dict) -> str | None:
        """Dispatches an intent to the corresponding action handler."""
        method_name = f"_handle_{intent_name}"
        handler = getattr(self, method_name, None)

        if handler:
            return handler(parameters)
        return None

    def _handle_get_time(self, parameters: dict) -> str:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The time is {now}."

    def _handle_get_date(self, parameters: dict) -> str:
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {today}."

    def _handle_set_volume(self, parameters: dict) -> str:
        level_str = parameters.get("level", "50")
        try:
            level_int = max(0, min(100, int(level_str)))
            # Adjust Windows system volume if pycaw is available
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(level_int / 100.0, None)
                logger.info(f"Hardware master volume set to {level_int}%.")
            except Exception as e:
                logger.debug(f"Hardware volume change skipped: {e}")

            return f"Volume set to {level_int} percent."
        except ValueError:
            return f"Volume level '{level_str}' is not a valid number."

    def _handle_stop_assistant(self, parameters: dict) -> str:
        logger.info("Stop command received.")
        return "Goodbye! Have a great day."

    def _handle_open_application(self, parameters: dict) -> str:
        app_name = parameters.get("app_name", "").strip().lower()
        if not app_name:
            return "Please specify which application you would like me to open."

        # Common Windows shortcuts
        app_map = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "browser": "start msedge",
        }

        cmd = app_map.get(app_name, app_name)
        try:
            subprocess.Popen(cmd, shell=True)
            return f"Opening {app_name}."
        except Exception as e:
            logger.error(f"Failed to open application {app_name}: {e}")
            return f"Sorry, I couldn't open {app_name}."
