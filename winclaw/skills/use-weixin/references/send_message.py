import os
import subprocess
import sys
import time

from pywinauto import Application, Desktop
from pywinauto.keyboard import send_keys

CONTACT_NAME = "上岸"
MESSAGE_TEXT = "明天早上8点来开会"


def find_wechat_exe() -> str | None:
    env_path = os.environ.get("WECHAT_EXE", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        r"C:\\Program Files\\Tencent\\WeChat\\WeChat.exe",
        r"C:\\Program Files (x86)\\Tencent\\WeChat\\WeChat.exe",
        os.path.join(
            os.environ.get("APPDATA", ""),
            r"Tencent\\WeChat\\WeChat.exe",
        ),
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            r"Tencent\\WeChat\\WeChat.exe",
        ),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def set_clipboard_text(text: str) -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$value = [Console]::In.ReadToEnd(); Set-Clipboard -Value $value",
        ],
        input=text,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "无法写入剪贴板")


def get_wechat_window():
    for pattern in (".*微信.*", ".*WeChat.*"):
        windows = Desktop(backend="uia").windows(title_re=pattern, visible_only=True)
        for win in windows:
            if win.element_info.control_type == "Window":
                return win
    return None


def connect_or_start_wechat():
    window = get_wechat_window()
    if window is not None:
        return window

    wechat_exe = find_wechat_exe()
    if not wechat_exe:
        raise FileNotFoundError(
            "未找到微信可执行文件，请先启动微信，或设置环境变量WECHAT_EXE指向WeChat.exe"
        )

    Application(backend="uia").start(wechat_exe)
    for _ in range(30):
        window = get_wechat_window()
        if window is not None:
            return window
        time.sleep(1)
    raise TimeoutError("微信启动后未找到主窗口，请确认已登录桌面端微信")


def focus_window(window) -> None:
    window.restore()
    window.set_focus()
    time.sleep(0.8)


def open_chat(contact_name: str) -> None:
    send_keys("^f")
    time.sleep(0.5)
    set_clipboard_text(contact_name)
    send_keys("^v")
    time.sleep(0.8)
    send_keys("{ENTER}")
    time.sleep(1.0)


def send_message(message: str) -> None:
    set_clipboard_text(message)
    send_keys("^v")
    time.sleep(0.3)
    send_keys("{ENTER}")
    time.sleep(0.5)


def main() -> int:
    try:
        window = connect_or_start_wechat()
        focus_window(window)
        open_chat(CONTACT_NAME)
        send_message(MESSAGE_TEXT)
        print(f'已向 "{CONTACT_NAME}" 发送: {MESSAGE_TEXT}')
        return 0
    except Exception as exc:
        print(f"执行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
