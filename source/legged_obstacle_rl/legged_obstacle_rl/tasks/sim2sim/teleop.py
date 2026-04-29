import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from evdev import InputDevice, categorize, ecodes, list_devices


@dataclass
class State:
    lin_x: float = 0.0
    lin_y: float = 0.0
    ang_z: float = 0.0
    base_height: float = 0.38
    stop: bool = False


# Shared state instance
state = State()


def teleop_backend(state_obj):
    """Background worker that listens for keyboard events."""
    # Find keyboard
    devices = [InputDevice(path) for path in list_devices()]
    dev = next((d for d in devices if "keyboard" in d.name.lower()), None)

    if not dev:
        print("Error: No keyboard found. Check permissions/group.")
        return

    key_map = {
        "KEY_W": ("lin_x", 0.1),
        "KEY_S": ("lin_x", -0.1),
        "KEY_A": ("lin_y", 0.1),
        "KEY_D": ("lin_y", -0.1),
        "KEY_Q": ("ang_z", 0.1),
        "KEY_E": ("ang_z", -0.1),
        "KEY_R": ("base_height", 0.01),
        "KEY_F": ("base_height", -0.01),
    }

    # Internal loop blocks until an event occurs
    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            if key_event.keystate in (1, 2):  # 1=Press, 2=Hold
                key_name = key_event.keycode

                if key_name == "KEY_ESC" or key_name == "KEY_8":
                    state_obj.stop = True
                    break

                if key_name in key_map:
                    attr, val = key_map[key_name]
                    new_val = getattr(state_obj, attr) + val
                    setattr(state_obj, attr, new_val)

                # Clamp values
                state_obj.lin_x = np.clip(state_obj.lin_x, -1.5, 1.5)
                state_obj.lin_y = np.clip(state_obj.lin_y, -1.5, 1.5)
                state_obj.ang_z = np.clip(state_obj.ang_z, -1.5, 1.5)
                state_obj.base_height = np.clip(state_obj.base_height, 0.1, 0.5)


def start_teleop_thread():
    teleop_thread = threading.Thread(target=teleop_backend, args=(state,), daemon=True)
    teleop_thread.start()


def main():
    start_teleop_thread()
    print("Teleop thread started. Controls: WASD, QE, RF. ESC to quit.")

    try:
        while not state.stop:
            print(f"\rRobot Cmd -> X: {state.lin_x:.2f} | Y: {state.lin_y:.2f} | H: {state.base_height:.2f}  ", end="")
            time.sleep(0.02)

    except KeyboardInterrupt:
        state.stop = True

    print("\nRobot control stopped.")


if __name__ == "__main__":
    main()
