"""Chat ID 预热池：预先为每个可用账号创建若干 chat_id 放在队列里，
请求到来时直接从队列 pop 一个省去 /chats/new 握手（实测 500ms~6s 不等）。

典型收益：每次请求节省 500~3000ms 握手时延；最坏情况抖动时节省 5~6s。

工作流：
- 服务启动 → 每账号预建 target_per_account 个 chat_id
- 请求用掉一个 chat_id → 后台立即补位一个
- 每账号池大小上限：target_per_account (默认 3)
- chat_id 有 TTL (默认 30 分钟)，超时背景任务丢弃+重建
- 请求取不到预热 chat_id 时：fallback 到同步 create_chat（当前行为）
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

log = logging.getLogger("qwen2api.chat_pool")


class _Entry:
    __slots__ = ("chat_id", "created_at")

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.created_at = time.time()


class ChatIdPool:
    """按账号邮箱 key 的 chat_id 队列。线程/协程安全。"""

    def __init__(
        self,
        client,
        *,
        target_per_account: int = 5,
        ttl_seconds: float = 10 * 60,
        max_concurrency: int = 16,
        default_model: str = "qwen3.6-plus",
    ):
        self._client = client
        self._target = target_per_account
        self._ttl = ttl_seconds
        self._max_concurrency = max(1, int(max_concurrency))
        self._prewarm_semaphore = asyncio.Semaphore(self._max_concurrency)
        self._default_model = default_model
        self._queues: dict[str, deque[_Entry]] = {}
        self._lock = asyncio.Lock()
        self._refill_task: Optional[asyncio.Task] = None
        self._refilling_emails: set[str] = set()
        self._shutdown = False

    async def _delete_entry(self, account_or_email, chat_id: str, *, source: str) -> None:
        if not chat_id:
            return
        if isinstance(account_or_email, str):
            account = getattr(self._client, "account_pool", None).get_by_email(account_or_email) if getattr(self._client, "account_pool", None) else None
        else:
            account = account_or_email
        token = getattr(account, "token", None)
        if not token:
            log.debug("[ChatIdPool] skip delete chat_id=%s source=%s: missing token", chat_id, source)
            return
        await self._client.delete_chat_reliable(token, chat_id, source=source)

    def _delete_entry_background(self, account_or_email, chat_id: str, *, source: str) -> None:
        if not chat_id:
            return
        background_delete = getattr(self._client, "delete_chat_background", None)
        if background_delete is not None:
            if isinstance(account_or_email, str):
                account = getattr(self._client, "account_pool", None).get_by_email(account_or_email) if getattr(self._client, "account_pool", None) else None
            else:
                account = account_or_email
            token = getattr(account, "token", None)
            if token:
                background_delete(token, chat_id, source=source)
                return
        try:
            asyncio.get_running_loop().create_task(self._delete_entry(account_or_email, chat_id, source=source))
        except RuntimeError:
            log.debug("[ChatIdPool] skip background delete chat_id=%s source=%s: no running loop", chat_id, source)

    @property
    def target(self) -> int:
        return self._target

    @property
    def ttl(self) -> float:
        return self._ttl

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def update_config(
        self,
        *,
        target: int | None = None,
        ttl_seconds: float | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        """运行时热更新参数。target 调小会在下一轮 refill 时把多余的 chat_id 丢掉；
        调大会在下一轮补位时扩容。TTL 变化影响下一次 acquire 的过期判断。"""
        if target is not None:
            self._target = max(0, int(target))
        if ttl_seconds is not None:
            self._ttl = max(30.0, float(ttl_seconds))
        if max_concurrency is not None:
            next_value = max(1, int(max_concurrency))
            if next_value != self._max_concurrency:
                self._max_concurrency = next_value
                self._prewarm_semaphore = asyncio.Semaphore(self._max_concurrency)
        log.info(
            "[ChatIdPool] config updated target=%s ttl=%ss max_concurrency=%s",
            self._target,
            self._ttl,
            self._max_concurrency,
        )

    async def apply_config(
        self,
        *,
        target: int | None = None,
        ttl_seconds: float | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        previous_target = self._target
        self.update_config(target=target, ttl_seconds=ttl_seconds, max_concurrency=max_concurrency)
        if target is None or self._target >= previous_target:
            return
        await self.prune_to_target()

    async def start(self) -> None:
        """服务启动时调用，完成首轮预热 + 启动后台补位 loop。"""
        # 初次预热 & 启动补位 loop
        self._refill_task = asyncio.create_task(self._refill_loop())
        log.info(f"[ChatIdPool] started (target={self._target}, ttl={self._ttl}s)")

    async def stop(self) -> None:
        self._shutdown = True
        if self._refill_task:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.flush_all(source="chat_pool_stop")

    async def acquire(self, email: str, model: str | None = None) -> Optional[str]:
        """优先从预热池取 chat_id；池空或过期则返回 None（调用方走同步 create_chat）。"""
        if not email:
            return None
        expired: list[str] = []
        selected: str | None = None
        async with self._lock:
            q = self._queues.get(email)
            if not q:
                return None
            now = time.time()
            while q:
                entry = q.popleft()
                if now - entry.created_at < self._ttl:
                    log.debug(f"[ChatIdPool] HIT email={email} chat_id={entry.chat_id}")
                    selected = entry.chat_id
                    break
                # 过期就丢弃继续找下一个
                log.debug(f"[ChatIdPool] expired chat_id={entry.chat_id} email={email}")
                expired.append(entry.chat_id)
        for chat_id in expired:
            self._delete_entry_background(email, chat_id, source="chat_pool_expired")
        if selected:
            await self._schedule_refill(email, model or self._default_model, reason="consume")
        return selected

    async def _prewarm_one(self, account, model: str) -> None:
        """为某账号预建一个 chat_id 加入队列。"""
        try:
            token = account.token
            email = account.email
            if not token:
                log.warning(f"[ChatIdPool] prewarm skipped email={email}: missing token")
                return
            async with self._prewarm_semaphore:
                chat_id = await self._client.executor.create_chat(token, model, use_prewarmed=False)
            should_delete = False
            async with self._lock:
                q = self._queues.setdefault(email, deque())
                if len(q) >= self._target:
                    should_delete = True
                else:
                    q.append(_Entry(chat_id))
                    log.info(f"[ChatIdPool] prewarmed email={email} chat_id={chat_id} pool_size={len(q)}")
            if should_delete:
                self._delete_entry_background(account, chat_id, source="chat_pool_overfill")
                log.debug("[ChatIdPool] discarded overfill email=%s chat_id=%s", email, chat_id)
                return
        except Exception as e:
            # Make sure empty-string exceptions still show class name
            err = str(e) or type(e).__name__
            log.warning(f"[ChatIdPool] prewarm failed email={getattr(account, 'email', '?')}: {err}")

    async def _schedule_refill(self, email: str, model: str, *, reason: str) -> None:
        if self._shutdown or self._target <= 0 or not email:
            return
        async with self._lock:
            if email in self._refilling_emails:
                return
            q_size = len(self._queues.get(email, []))
            if q_size >= self._target:
                return
            self._refilling_emails.add(email)

        async def runner() -> None:
            try:
                await self._refill_account_once(email, model, reason=reason)
            finally:
                async with self._lock:
                    self._refilling_emails.discard(email)

        try:
            task = asyncio.create_task(runner())
            task.set_name(f"chat-id-refill-{email}")
        except RuntimeError:
            async with self._lock:
                self._refilling_emails.discard(email)

    async def _refill_account_once(self, email: str, model: str, *, reason: str) -> None:
        pool = getattr(self._client, "account_pool", None)
        if pool is None:
            return
        account = pool.get_by_email(email) if hasattr(pool, "get_by_email") else None
        if account is None:
            account = next((a for a in getattr(pool, "accounts", []) if getattr(a, "email", None) == email), None)
        if account is None:
            return
        if not getattr(account, "token", "") or getattr(account, "status_code", "valid") != "valid":
            return
        async with self._lock:
            q_size = len(self._queues.get(email, []))
            if q_size >= self._target:
                return
        log.debug("[ChatIdPool] schedule refill email=%s reason=%s pool_size=%s target=%s", email, reason, q_size, self._target)
        await self._prewarm_one(account, model)

    async def _refill_loop(self) -> None:
        """定期轮询：每账号池低于 target 则补位。30 秒一轮。"""
        interval = 30.0
        # 初始化立即跑一轮
        await asyncio.sleep(1.0)
        while not self._shutdown:
            try:
                await self._refill_once()
            except Exception as e:
                log.warning(f"[ChatIdPool] refill error: {e}")
            await asyncio.sleep(interval)

    async def _refill_once(self) -> None:
        """遍历账号池里所有 valid 账号，每个不足 target 就补位。"""
        pool = getattr(self._client, "account_pool", None)
        if pool is None:
            return
        await self.prune_expired()
        all_accounts = getattr(pool, "accounts", []) or []

        # 只对有 token + 状态 valid 的账号预热
        valid = [a for a in all_accounts if getattr(a, "token", "") and getattr(a, "status_code", "valid") == "valid"]

        refill_tasks: list[asyncio.Task] = []
        batch_size = max(1, self._max_concurrency * 4)

        async def drain_batch() -> None:
            if not refill_tasks:
                return
            results = await asyncio.gather(*refill_tasks, return_exceptions=True)
            refill_tasks.clear()
            for result in results:
                if isinstance(result, Exception):
                    log.warning("[ChatIdPool] refill task failed: %s", result)

        for acc in valid:
            async with self._lock:
                q_size = len(self._queues.get(acc.email, []))
            deficit = self._target - q_size
            # 每轮每账号最多补 1 个，避免突发 API 压力
            if deficit > 0:
                refill_tasks.append(
                    asyncio.create_task(
                        self._refill_account_once(acc.email, self._default_model, reason="periodic")
                    )
                )
                if len(refill_tasks) >= batch_size:
                    await drain_batch()
            elif deficit < 0:
                await self.prune_account_to_target(acc.email)
        await drain_batch()

    async def invalidate(self, email: str, chat_id: str) -> None:
        """标记某个 chat_id 为坏的——从池里移除，防止下次又被取到。

        用于上游返回空响应 / 5xx / 超时后的清理。"""
        if not email or not chat_id:
            return
        removed = False
        async with self._lock:
            q = self._queues.get(email)
            if not q:
                return
            remaining = deque(e for e in q if e.chat_id != chat_id)
            self._queues[email] = remaining
            if len(remaining) != len(q):
                removed = True
                log.info(f"[ChatIdPool] invalidated email={email} chat_id={chat_id}")
        if removed:
            await self._delete_entry(email, chat_id, source="chat_pool_invalidate")

    async def contains(self, email: str, chat_id: str) -> bool:
        if not email or not chat_id:
            return False
        async with self._lock:
            return any(e.chat_id == chat_id for e in self._queues.get(email, []))

    async def chat_ids(self, email: str | None = None) -> set[str]:
        async with self._lock:
            if email:
                return {e.chat_id for e in self._queues.get(email, [])}
            ids: set[str] = set()
            for q in self._queues.values():
                ids.update(e.chat_id for e in q)
            return ids

    async def flush_account(self, email: str) -> int:
        """把某账号池里的所有 chat_id 清空。用于该账号命中空响应/5xx 后的保守处理，
        防止同批次预热的其他 chat_id 也是坏的。返回清理数量。"""
        if not email:
            return 0
        entries: list[_Entry] = []
        async with self._lock:
            q = self._queues.get(email)
            if not q:
                return 0
            entries = list(q)
            n = len(q)
            self._queues[email] = deque()
            if n:
                log.info(f"[ChatIdPool] flushed {n} entries for email={email}")
        for entry in entries:
            await self._delete_entry(email, entry.chat_id, source="chat_pool_flush")
        return len(entries)

    async def flush_all(self, *, source: str = "chat_pool_flush_all") -> int:
        async with self._lock:
            items = [(email, entry.chat_id) for email, q in self._queues.items() for entry in q]
            self._queues = {}
        for email, chat_id in items:
            await self._delete_entry(email, chat_id, source=source)
        if items:
            log.info("[ChatIdPool] flushed all entries count=%s source=%s", len(items), source)
        return len(items)

    async def prune_expired(self) -> int:
        """Remove expired prewarmed chats even if nobody acquires from the pool."""
        now = time.time()
        expired: list[tuple[str, str]] = []
        async with self._lock:
            for email, q in list(self._queues.items()):
                kept = deque()
                for entry in q:
                    if now - entry.created_at >= self._ttl:
                        expired.append((email, entry.chat_id))
                    else:
                        kept.append(entry)
                self._queues[email] = kept
        for email, chat_id in expired:
            await self._delete_entry(email, chat_id, source="chat_pool_expired")
        if expired:
            log.info("[ChatIdPool] pruned expired entries count=%s ttl=%ss", len(expired), self._ttl)
        return len(expired)

    async def prune_account_to_target(self, email: str) -> int:
        if not email:
            return 0
        removed: list[str] = []
        async with self._lock:
            q = self._queues.get(email)
            if not q:
                return 0
            while len(q) > self._target:
                removed.append(q.pop().chat_id)
        for chat_id in removed:
            await self._delete_entry(email, chat_id, source="chat_pool_prune")
        return len(removed)

    async def prune_to_target(self) -> int:
        emails = list(self._queues.keys())
        total = 0
        for email in emails:
            total += await self.prune_account_to_target(email)
        return total

    async def size(self, email: str) -> int:
        async with self._lock:
            return len(self._queues.get(email, []))

    async def total_size(self) -> int:
        async with self._lock:
            return sum(len(q) for q in self._queues.values())
