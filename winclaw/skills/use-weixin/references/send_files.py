"""
使用 pywinauto 控制微信桌面端，向目标群组发送文件。

依赖安装：
    pip install pywinauto pyperclip

使用方法：
    1. 确保微信已登录并在前台运行
    2. 修改下方配置区的参数
    3. 运行脚本：python wechat_send_file.py
"""

import os
import sys
import time

import pyperclip
from pywinauto import Application, Desktop
from pywinauto.findwindows import ElementNotFoundError
from pywinauto.keyboard import send_keys

# ============================================================
#  配置区 —— 请在此修改
# ============================================================

# 目标群组 / 联系人名称（微信搜索框中的精确名称）
TARGET_NAME = "目标群组名称"

# 要发送的文件路径列表（支持多文件）
FILES_TO_SEND = [
    r"C:\Users\YourName\Documents\example.pdf",
    r"C:\Users\YourName\Desktop\report.xlsx",
]

# 每次操作之间的等待时间（秒），网络慢时可适当调大
STEP_DELAY = 1.0

# 微信主窗口标题关键字
WECHAT_TITLE = "微信"

# ============================================================


def find_wechat_window():
    """查找微信主窗口"""
    try:
        app = Application(backend="uia").connect(title_re=".*微信.*", timeout=5)
        win = app.window(title_re=".*微信.*")
        win.set_focus()
        print(f"[✓] 已连接微信窗口：{win.window_text()}")
        return app, win
    except ElementNotFoundError:
        print("[✗] 未找到微信窗口，请先启动并登录微信。")
        sys.exit(1)


def search_and_open_target(win, target_name: str):
    """通过搜索框定位目标群组/联系人并打开聊天窗口"""
    print(f"[→] 搜索目标：{target_name}")

    # 点击搜索框（微信顶部搜索图标，class 通常为 SearchBar 或类似）
    try:
        search_btn = win.child_window(title="搜索", control_type="Edit")
        search_btn.click_input()
    except Exception:
        # 备用方案：使用快捷键 Ctrl+F
        win.set_focus()
        send_keys("^f")

    time.sleep(STEP_DELAY)

    # 清空并输入目标名称（通过剪贴板粘贴，避免中文输入法问题）
    pyperclip.copy(target_name)
    send_keys("^a")
    send_keys("^v")
    time.sleep(STEP_DELAY)

    # 在搜索结果列表中点击第一个匹配项
    try:
        result_item = win.child_window(title=target_name, control_type="ListItem")
        result_item.click_input()
        print(f"[✓] 已打开聊天窗口：{target_name}")
    except Exception:
        # 直接按回车进入第一个结果
        print("[!] 未精确匹配，尝试按回车进入第一个结果…")
        send_keys("{ENTER}")

    time.sleep(STEP_DELAY)


def send_files_via_menu(win, file_paths: list):
    """通过聊天窗口的「文件」按钮发送文件"""
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[!] 文件不存在，跳过：{file_path}")
            continue

        print(f"[→] 发送文件：{file_path}")

        # ---- 方式一：点击工具栏「文件」按钮 ----
        try:
            file_btn = win.child_window(title="文件", control_type="Button")
            file_btn.click_input()
            time.sleep(STEP_DELAY)
        except Exception:
            # ---- 方式二：使用快捷键 Ctrl+Shift+A（部分版本支持）----
            print("[!] 未找到「文件」按钮，尝试使用快捷键…")
            win.set_focus()
            send_keys("^+a")
            time.sleep(STEP_DELAY)

        # 系统文件选择对话框出现后，输入文件路径
        _handle_file_dialog(file_path)
        time.sleep(STEP_DELAY * 2)  # 等待文件加载预览

        # 确认发送（按回车或点击「发送」按钮）
        _confirm_send(win)
        time.sleep(STEP_DELAY)
        print(f"[✓] 已发送：{os.path.basename(file_path)}")


def _handle_file_dialog(file_path: str):
    """处理系统文件选择对话框"""
    try:
        # 等待「打开」对话框出现
        dialog = Desktop(backend="uia").window(title_re=".*打开.*|.*Open.*", timeout=5)
        dialog.set_focus()

        # 直接在文件名输入框填入完整路径
        filename_edit = dialog.child_window(auto_id="1148", control_type="Edit")
        filename_edit.set_text("")
        pyperclip.copy(file_path)
        filename_edit.click_input()
        send_keys("^a")
        send_keys("^v")
        time.sleep(0.5)
        send_keys("{ENTER}")
    except Exception as e:
        print(f"[!] 文件对话框处理异常：{e}，尝试备用方案…")
        # 备用：直接键入路径后回车
        pyperclip.copy(file_path)
        send_keys("^v")
        time.sleep(0.5)
        send_keys("{ENTER}")


def _confirm_send(win):
    """点击发送按钮或按回车确认"""
    try:
        send_btn = win.child_window(title="发送(S)", control_type="Button")
        send_btn.click_input()
    except Exception:
        try:
            send_btn = win.child_window(title="发送", control_type="Button")
            send_btn.click_input()
        except Exception:
            send_keys("{ENTER}")


def send_files_via_drag_path(win, file_paths: list):
    """
    备用方案：将文件路径复制到聊天输入框后通过剪贴板粘贴文件。
    注意：此方法仅适用于单个文件，且微信版本需支持粘贴文件。
    """
    import subprocess

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[!] 文件不存在，跳过：{file_path}")
            continue

        # 将文件复制到剪贴板（使用 PowerShell）
        ps_cmd = (
            f'$filePath = "{file_path}"; '
            f"$col = New-Object System.Collections.Specialized.StringCollection; "
            f"$col.Add($filePath); "
            f"[System.Windows.Forms.Clipboard]::SetFileDropList($col)"
        )
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        time.sleep(0.5)

        # 聚焦输入框并粘贴
        try:
            input_box = win.child_window(title="输入", control_type="Edit")
            input_box.click_input()
        except Exception:
            win.set_focus()

        send_keys("^v")
        time.sleep(STEP_DELAY)
        send_keys("{ENTER}")
        time.sleep(STEP_DELAY)
        print(f"[✓] 已发送（剪贴板方式）：{os.path.basename(file_path)}")


# ============================================================
#  主流程
# ============================================================


def main():
    print("=" * 50)
    print("  微信文件发送脚本  (基于 pywinauto)")
    print("=" * 50)

    # 1. 连接微信
    app, win = find_wechat_window()

    # 2. 打开目标聊天窗口
    search_and_open_target(win, TARGET_NAME)

    # 3. 发送文件
    send_files_via_menu(win, FILES_TO_SEND)

    print("\n[完成] 所有文件发送完毕。")


if __name__ == "__main__":
    main()
