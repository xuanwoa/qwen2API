import asyncio
import logging
import json
from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.gc")

async def garbage_collect_chats(client: QwenClient):
    """
    后台守护进程：每隔 15 分钟遍历所有存活的账号，
    调用千问列表接口，删除由 API 产生且已成为孤儿的对话 (title 包含 api_)。
    """
    while True:
        await asyncio.sleep(900)  # 15分钟
        log.info("[GC] 开始自动焚烧孤儿会话...")
        pool = client.account_pool
        for acc in pool.accounts:
            if not acc.is_available():
                continue
            try:
                # 获取会话列表
                res = await client.engine.api_call("GET", "/api/v2/chats?limit=50", acc.token)
                if isinstance(res, dict) and res.get("status") == 200:
                    data = json.loads(res.get("body", "{}"))
                    if isinstance(data, dict):
                        chats = data.get("data", [])
                        if isinstance(chats, list):
                            for c in chats:
                                if isinstance(c, dict) and c.get("title", "").startswith("api_"):
                                    # 异步焚烧
                                    asyncio.create_task(client.delete_chat(acc.token, c["id"]))
            except Exception as e:
                log.warning(f"[GC] 账号 {acc.email} 焚烧失败: {e}")
