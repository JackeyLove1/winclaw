"""Cron scheduling tool backed by APScheduler with lease-based leadership."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from examples.base import Tool, ToolResult

AUTO_EXPIRY_DAYS = 7
JITTER_MINUTES = {0, 30}
JITTER_OFFSET_MAX = 4


@dataclass
class _LeaseState:
    owner_id: str
    pid: int
    expires_at: float
    last_heartbeat_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "pid": self.pid,
            "expires_at": self.expires_at,
            "last_heartbeat_at": self.last_heartbeat_at,
        }


class CronTool(Tool):
    """Schedule recurring or one-shot prompts with cron expressions."""

    def __init__(
        self,
        workdir: Path,
        lease_ttl_seconds: int = 15,
        lease_heartbeat_seconds: int = 5,
    ) -> None:
        self._workdir = Path(workdir)
        self._tasks_file = self._workdir / ".claude" / "scheduled_tasks.json"
        self._lease_file = self._workdir / ".claude" / "cron.lease.json"
        self._owner_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._lease_ttl_seconds = max(5, int(lease_ttl_seconds))
        self._lease_heartbeat_seconds = max(1, int(lease_heartbeat_seconds))
        self._queue: Queue[str] = Queue()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._scheduler = AsyncIOScheduler()
        self._scheduler_started = False
        self._running = False
        self._lease_task: asyncio.Task[None] | None = None
        self._lease_active = False

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule future prompts with cron. "
            "Actions: create, list, delete."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "delete"],
                    "description": "Operation to perform",
                },
                "cron": {
                    "type": "string",
                    "description": "5-field cron expression: 'min hour dom month dow'",
                },
                "prompt": {
                    "type": "string",
                    "description": "Prompt injected when task fires",
                },
                "recurring": {
                    "type": "boolean",
                    "description": "true=repeat; false=one-shot at next cron match",
                },
                "durable": {
                    "type": "boolean",
                    "description": "true=persist under .claude/scheduled_tasks.json",
                },
                "id": {
                    "type": "string",
                    "description": "Task ID for delete",
                },
            },
            "required": ["action"],
        }

    async def start(self) -> None:
        """Start scheduler and lease heartbeat."""
        if self._running:
            return
        self._running = True
        self._load_durable_tasks()
        self._reschedule_all_locked()
        self._ensure_scheduler_started()
        await self._lease_step()
        self._lease_task = asyncio.create_task(self._lease_loop())

    async def stop(self) -> None:
        """Stop scheduler and release lease."""
        self._running = False
        if self._lease_task is not None:
            self._lease_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lease_task
            self._lease_task = None
        self._release_lease()
        self._lease_active = False
        if self._scheduler_started:
            self._scheduler.shutdown(wait=False)
            self._scheduler_started = False
            self._scheduler = AsyncIOScheduler()

    def drain_notifications(self) -> list[str]:
        """Drain all fired notifications."""
        notifications: list[str] = []
        while True:
            try:
                notifications.append(self._queue.get_nowait())
            except Empty:
                break
        return notifications

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        try:
            if action == "create":
                out = self._create_task(
                    cron=str(kwargs.get("cron") or "").strip(),
                    prompt=str(kwargs.get("prompt") or "").strip(),
                    recurring=bool(kwargs.get("recurring", True)),
                    durable=bool(kwargs.get("durable", False)),
                )
                return ToolResult(success=True, content=out)
            if action == "list":
                return ToolResult(success=True, content=self._list_tasks())
            if action == "delete":
                out = self._delete_task(str(kwargs.get("id") or "").strip())
                return ToolResult(success=True, content=out)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _lease_loop(self) -> None:
        while self._running:
            try:
                await self._lease_step()
            except Exception:
                self._lease_active = False
            await asyncio.sleep(self._lease_heartbeat_seconds)

    async def _lease_step(self) -> None:
        active = self._acquire_or_renew_lease()
        if active and not self._lease_active:
            print("[Cron lease] acquired")
        if not active and self._lease_active:
            print("[Cron lease] lost")
        self._lease_active = active

    def _acquire_or_renew_lease(self) -> bool:
        now = time.time()
        lease = self._read_lease()
        can_takeover = lease is None or lease.expires_at <= now or not self._is_pid_alive(lease.pid)
        if lease and lease.owner_id == self._owner_id:
            can_takeover = True
        if not can_takeover:
            return False

        next_lease = _LeaseState(
            owner_id=self._owner_id,
            pid=os.getpid(),
            expires_at=now + self._lease_ttl_seconds,
            last_heartbeat_at=now,
        )
        self._write_lease(next_lease)
        confirm = self._read_lease()
        return bool(confirm and confirm.owner_id == self._owner_id)

    def _release_lease(self) -> None:
        lease = self._read_lease()
        if lease and lease.owner_id == self._owner_id:
            with contextlib.suppress(OSError):
                self._lease_file.unlink()

    def _holds_lease_now(self) -> bool:
        lease = self._read_lease()
        if lease is None:
            return False
        if lease.owner_id != self._owner_id:
            return False
        return lease.expires_at > time.time()

    def _read_lease(self) -> _LeaseState | None:
        if not self._lease_file.exists():
            return None
        try:
            data = json.loads(self._lease_file.read_text(encoding="utf-8"))
            return _LeaseState(
                owner_id=str(data["owner_id"]),
                pid=int(data["pid"]),
                expires_at=float(data["expires_at"]),
                last_heartbeat_at=float(data["last_heartbeat_at"]),
            )
        except Exception:
            return None

    def _write_lease(self, lease: _LeaseState) -> None:
        self._lease_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._lease_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(lease.to_json(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self._lease_file)

    def _is_pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _load_durable_tasks(self) -> None:
        with self._lock:
            self._tasks.clear()
            if not self._tasks_file.exists():
                return
            data = json.loads(self._tasks_file.read_text(encoding="utf-8"))
            for task in data:
                if task.get("durable"):
                    self._tasks[str(task["id"])] = dict(task)
            self._cleanup_expired_locked()

    def _save_durable_tasks_locked(self) -> None:
        durable = [t for t in self._tasks.values() if t.get("durable")]
        self._tasks_file.parent.mkdir(parents=True, exist_ok=True)
        self._tasks_file.write_text(json.dumps(durable, indent=2) + "\n", encoding="utf-8")

    def _list_tasks(self) -> str:
        with self._lock:
            self._cleanup_expired_locked()
            tasks = list(self._tasks.values())
        if not tasks:
            return "No scheduled tasks."
        tasks.sort(key=lambda t: t["createdAt"])
        lines: list[str] = []
        for task in tasks:
            mode = "recurring" if task["recurring"] else "one-shot"
            store = "durable" if task["durable"] else "session"
            age_h = (time.time() - float(task["createdAt"])) / 3600
            lines.append(
                f"{task['id']}  {task['cron']}  [{mode}/{store}] ({age_h:.1f}h old): "
                f"{task['prompt'][:80]}"
            )
        return "\n".join(lines)

    def _create_task(self, cron: str, prompt: str, recurring: bool, durable: bool) -> str:
        if not cron:
            raise ValueError("cron is required")
        if not prompt:
            raise ValueError("prompt is required")
        CronTrigger.from_crontab(cron)
        task_id = uuid.uuid4().hex[:8]
        task = {
            "id": task_id,
            "cron": cron,
            "prompt": prompt,
            "recurring": recurring,
            "durable": durable,
            "createdAt": time.time(),
            "last_fired": None,
            "jitter_offset": self._compute_jitter(cron) if recurring else 0,
        }
        with self._lock:
            self._tasks[task_id] = task
            self._schedule_task_locked(task)
            if durable:
                self._save_durable_tasks_locked()
        mode = "recurring" if recurring else "one-shot"
        store = "durable" if durable else "session-only"
        return f"Created task {task_id} ({mode}, {store}): cron={cron}"

    def _delete_task(self, task_id: str) -> str:
        if not task_id:
            raise ValueError("id is required for delete")
        with self._lock:
            task = self._tasks.pop(task_id, None)
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(job_id=self._job_id(task_id))
            if task and task.get("durable"):
                self._save_durable_tasks_locked()
        if task:
            return f"Deleted task {task_id}"
        return f"Task {task_id} not found"

    def _schedule_task_locked(self, task: dict[str, Any]) -> None:
        self._ensure_scheduler_started()
        job_id = self._job_id(str(task["id"]))
        with contextlib.suppress(Exception):
            self._scheduler.remove_job(job_id=job_id)

        cron_expr = self._jittered_expr(task["cron"], int(task.get("jitter_offset", 0)))
        trigger = CronTrigger.from_crontab(cron_expr)
        if task["recurring"]:
            self._scheduler.add_job(
                self._on_job_fire,
                trigger=trigger,
                args=[str(task["id"])],
                id=job_id,
                replace_existing=True,
            )
            return

        now = datetime.now().astimezone()
        next_fire = trigger.get_next_fire_time(None, now)
        if next_fire is None:
            self._tasks.pop(str(task["id"]), None)
            return
        self._scheduler.add_job(
            self._on_job_fire,
            trigger=DateTrigger(run_date=next_fire),
            args=[str(task["id"])],
            id=job_id,
            replace_existing=True,
        )

    def _reschedule_all_locked(self) -> None:
        for task in list(self._tasks.values()):
            self._schedule_task_locked(task)

    def _on_job_fire(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or not self._lease_active or not self._holds_lease_now():
                return
            self._cleanup_expired_locked()
            if task_id not in self._tasks:
                return
            task = self._tasks[task_id]
            note = f"[Scheduled task {task['id']}]: {task['prompt']}"
            self._queue.put(note)
            task["last_fired"] = time.time()
            print(f"[Cron] Fired: {task['id']}")
            if not task["recurring"]:
                self._tasks.pop(task_id, None)
            if task.get("durable"):
                self._save_durable_tasks_locked()

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired_ids: list[str] = []
        for task_id, task in self._tasks.items():
            if not task.get("recurring"):
                continue
            age_days = (now - float(task["createdAt"])) / 86400
            if age_days > AUTO_EXPIRY_DAYS:
                expired_ids.append(task_id)
        for task_id in expired_ids:
            self._tasks.pop(task_id, None)
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(job_id=self._job_id(task_id))
            print(f"[Cron] Auto-expired: {task_id}")
        if expired_ids:
            self._save_durable_tasks_locked()

    def _compute_jitter(self, cron_expr: str) -> int:
        fields = cron_expr.strip().split()
        if not fields:
            return 0
        minute = fields[0]
        try:
            minute_val = int(minute)
        except ValueError:
            return 0
        if minute_val in JITTER_MINUTES:
            return (hash(cron_expr) % JITTER_OFFSET_MAX) + 1
        return 0

    def _jittered_expr(self, cron_expr: str, jitter_offset: int) -> str:
        if jitter_offset <= 0:
            return cron_expr
        fields = cron_expr.strip().split()
        if len(fields) != 5:
            return cron_expr
        try:
            minute = int(fields[0])
        except ValueError:
            return cron_expr
        if minute not in JITTER_MINUTES:
            return cron_expr
        fields[0] = str(minute + jitter_offset)
        return " ".join(fields)

    def _ensure_scheduler_started(self) -> None:
        if self._scheduler_started:
            return
        self._scheduler.start()
        self._scheduler_started = True

    def _job_id(self, task_id: str) -> str:
        return f"cron-task-{task_id}"
