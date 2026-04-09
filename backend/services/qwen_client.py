import asyncio
import json
import logging
import time
import uuid
from typing import Optional
from backend.core.browser_engine import BrowserEngine
from backend.core.account_pool import AccountPool, Account
from backend.core.config import settings
from backend.services.auth_resolver import AuthResolver

log = logging.getLogger("qwen2api.client")

AUTH_FAIL_KEYWORDS = ("token", "unauthorized", "expired", "forbidden", "401", "403", "invalid", "login", "activation", "pending activation", "not activated")
PENDING_ACTIVATION_KEYWORDS = ("pending activation", "please check your email", "not activated")
BANNED_KEYWORDS = ("banned", "suspended", "blocked", "disabled", "risk control", "violat", "forbidden by policy")

def _is_auth_error(error_msg: str) -> bool:
    msg = error_msg.lower()
    return any(keyword in msg for keyword in AUTH_FAIL_KEYWORDS)

def _is_pending_activation_error(error_msg: str) -> bool:
    msg = error_msg.lower()
    return any(keyword in msg for keyword in PENDING_ACTIVATION_KEYWORDS)

def _is_banned_error(error_msg: str) -> bool:
    msg = error_msg.lower()
    return any(keyword in msg for keyword in BANNED_KEYWORDS)

class QwenClient:
    def __init__(self, engine: BrowserEngine, account_pool: AccountPool):
        self.engine = engine
        self.account_pool = account_pool
        self.auth_resolver = AuthResolver(account_pool)

    async def create_chat(self, token: str, model: str) -> str:
        ts = int(time.time())
        body = {"title": f"api_{ts}", "models": [model], "chat_mode": "normal",
                "chat_type": "t2t", "timestamp": ts}

        r = await self.engine.api_call("POST", "/api/v2/chats/new", token, body)
        if r["status"] == 429:
            raise Exception("429 Too Many Requests (Engine Queue Full)")

        body_text = r.get("body", "")
        if r["status"] != 200:
            body_lower = body_text.lower()
            if (r["status"] in (401, 403)
                    or "unauthorized" in body_lower or "forbidden" in body_lower
                    or "token" in body_lower or "login" in body_lower
                    or "401" in body_text or "403" in body_text):
                raise Exception(f"unauthorized: create_chat HTTP {r['status']}: {body_text[:100]}")
            raise Exception(f"create_chat HTTP {r['status']}: {body_text[:100]}")

        try:
            data = json.loads(body_text)
            if not data.get("success") or "id" not in data.get("data", {}):
                raise Exception("Qwen API returned error or missing id")
            return data["data"]["id"]
        except Exception as e:
            body_lower = body_text.lower()
            if any(kw in body_lower for kw in ("html", "login", "unauthorized", "activation",
                                                "pending", "forbidden", "token", "expired", "invalid")):
                raise Exception(f"unauthorized: account issue: {body_text[:200]}")
            raise Exception(f"create_chat parse error: {e}, body={body_text[:200]}")

    async def delete_chat(self, token: str, chat_id: str):
        await self.engine.api_call("DELETE", f"/api/v2/chats/{chat_id}", token)

    async def verify_token(self, token: str) -> bool:
        """Verify token validity via direct HTTP (no browser page needed)."""
        if not token:
            return False

        try:
            import httpx
            from backend.services.auth_resolver import BASE_URL

            # 伪造浏览器指纹，避免被 Aliyun WAF 拦截
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://chat.qwen.ai/",
                "Origin": "https://chat.qwen.ai",
                "Connection": "keep-alive"
            }

            async with httpx.AsyncClient(timeout=15) as hc:
                resp = await hc.get(
                    f"{BASE_URL}/api/v1/auths/",
                    headers=headers,
                )
            if resp.status_code != 200:
                return False

            # 增加对空响应/非 JSON 响应的容错，防止 GFW 拦截或代理返回假 200 OK 导致崩溃
            try:
                data = resp.json()
                return data.get("role") == "user"
            except Exception as e:
                log.warning(f"[verify_token] JSON parse error (可能是被拦截或代理异常): {e}, status={resp.status_code}, text={resp.text[:100]}")
                # 如果遇到阿里云 WAF 拦截，通常是因为 httpx 直接请求被墙，或者 token 本身就是正常的。
                # 由于这是为了快速验证，如果被 WAF 拦截 (HTML)，我们姑且假定它是活着的，交给后面的浏览器引擎去真实处理
                if "aliyun_waf" in resp.text.lower() or "<!doctype" in resp.text.lower():
                    log.info(f"[verify_token] 遇到 WAF 拦截页面，放行交给底层无头浏览器引擎处理。")
                    return True
                return False
        except Exception as e:
            log.warning(f"[verify_token] HTTP error: {e}")
            return False

    async def list_models(self, token: str) -> list:
        try:
            import httpx
            from backend.services.auth_resolver import BASE_URL

            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://chat.qwen.ai/",
                "Origin": "https://chat.qwen.ai",
                "Connection": "keep-alive"
            }

            async with httpx.AsyncClient(timeout=10) as hc:
                resp = await hc.get(
                    f"{BASE_URL}/api/models",
                    headers=headers,
                )
            if resp.status_code != 200:
                return []
            try:
                return resp.json().get("data", [])
            except Exception as e:
                log.warning(f"[list_models] JSON parse error: {e}, status={resp.status_code}, text={resp.text[:100]}")
                return []
        except Exception:
            return []

    def _build_payload(self, chat_id: str, model: str, content: str, has_custom_tools: bool = False) -> dict:
        ts = int(time.time())
        feature_config = {
            "thinking_enabled": True, "output_schema": "phase", "research_mode": "normal",
            "auto_thinking": True, "thinking_mode": "Auto", "thinking_format": "summary",
            "auto_search": not has_custom_tools,
            "code_interpreter": not has_custom_tools,
            "function_calling": not has_custom_tools,  # disable Qwen native tool calls
            "plugins_enabled": not has_custom_tools,
        }
        return {
            "stream": True, "version": "2.1", "incremental_output": True,
            "chat_id": chat_id, "chat_mode": "normal", "model": model, "parent_id": None,
            "messages": [{
                "fid": str(uuid.uuid4()), "parentId": None, "childrenIds": [str(uuid.uuid4())],
                "role": "user", "content": content, "user_action": "chat", "files": [],
                "timestamp": ts, "models": [model], "chat_type": "t2t",
                "feature_config": feature_config,
                "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t", "parent_id": None,
            }],
            "timestamp": ts,
        }

    def parse_sse_chunk(self, chunk: str) -> list[dict]:
        events = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
                events.append(obj)
            except Exception:
                continue
        
        parsed = []
        for evt in events:
            if evt.get("choices"):
                delta = evt["choices"][0].get("delta", {})
                parsed.append({
                    "type": "delta",
                    "phase": delta.get("phase", "answer"),
                    "content": delta.get("content", ""),
                    "status": delta.get("status", ""),
                    "extra": delta.get("extra", {})
                })
        return parsed

    async def chat_stream_events_with_retry(self, model: str, content: str, has_custom_tools: bool = False, exclude_accounts: Optional[set[str]] = None):
        """无感容灾重试逻辑：上游挂了自动换号"""
        exclude = set(exclude_accounts or set())
        for attempt in range(settings.MAX_RETRIES):
            acc = await self.account_pool.acquire_wait(timeout=60, exclude=exclude)
            if not acc:
                pool_status = self.account_pool.status()
                raise Exception(
                    "No available accounts in pool "
                    f"(total={pool_status['total']}, valid={pool_status['valid']}, "
                    f"invalid={pool_status['invalid']}, activation_pending={pool_status.get('activation_pending', 0)}, "
                    f"rate_limited={pool_status['rate_limited']}, in_use={pool_status['in_use']}, waiting={pool_status['waiting']})"
                )
                
            try:
                chat_id = await self.create_chat(acc.token, model)
                payload = self._build_payload(chat_id, model, content, has_custom_tools)
                
                # First yield the chat_id and account to the consumer
                yield {"type": "meta", "chat_id": chat_id, "acc": acc}

                buffer = ""
                async for chunk_result in self.engine.fetch_chat(acc.token, chat_id, payload):
                    if chunk_result.get("status") == 429:
                        raise Exception("Engine Queue Full")
                    if chunk_result.get("status") != 200 and chunk_result.get("status") != "streamed":
                        raise Exception(f"HTTP {chunk_result['status']}: {chunk_result.get('body', '')[:100]}")
                    
                    if "chunk" in chunk_result:
                        buffer += chunk_result["chunk"]
                        while "\n\n" in buffer:
                            msg, buffer = buffer.split("\n\n", 1)
                            events = self.parse_sse_chunk(msg)
                            for evt in events:
                                yield {"type": "event", "event": evt}
                    elif "body" in chunk_result and chunk_result["body"] and chunk_result["body"] != "streamed":
                        buffer += chunk_result["body"]
                
                if buffer:
                    events = self.parse_sse_chunk(buffer)
                    for evt in events:
                        yield {"type": "event", "event": evt}
                return
                
            except Exception as e:
                err_msg = str(e).lower()
                should_save = False
                if "429" in err_msg or "rate limit" in err_msg or "too many" in err_msg:
                    self.account_pool.mark_rate_limited(acc, error_message=str(e))
                    exclude.add(acc.email)
                elif _is_pending_activation_error(err_msg):
                    self.account_pool.mark_invalid(acc, reason="pending_activation", error_message=str(e))
                    exclude.add(acc.email)
                    acc.activation_pending = True
                    should_save = True
                    asyncio.create_task(self.auth_resolver.auto_heal_account(acc))
                elif _is_banned_error(err_msg):
                    self.account_pool.mark_invalid(acc, reason="banned", error_message=str(e))
                    exclude.add(acc.email)
                elif _is_auth_error(err_msg):
                    self.account_pool.mark_invalid(acc, reason="auth_error", error_message=str(e))
                    exclude.add(acc.email)
                    asyncio.create_task(self.auth_resolver.auto_heal_account(acc))
                else:
                    acc.status_code = "invalid"
                    acc.last_error = str(e)
                    exclude.add(acc.email)

                if should_save:
                    await self.account_pool.save()

                self.account_pool.release(acc)
                log.warning(f"[Retry {attempt+1}/{settings.MAX_RETRIES}] Account {acc.email} failed: {e}. Retrying...")
                
        raise Exception(f"All {settings.MAX_RETRIES} attempts failed. Please check upstream accounts.")
