import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

log = logging.getLogger("qwen2api.browser")

JS_FETCH = """
async (args) => {
    const opts = {
        method: args.method,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + args.token
        }
    };
    if (args.body_json) opts.body = args.body_json;
    const res = await fetch(args.url, opts);
    const text = await res.text();
    return { status: res.status, body: text };
}
"""

JS_STREAM_CHUNKED = """
async (args) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1800000);
    try {
        const res = await fetch(args.url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + args.token
            },
            body: args.payload_json,
            signal: controller.signal
        });
        if (!res.ok) {
            const t = await res.text();
            clearTimeout(timer);
            return { status: res.status, body: t.substring(0, 2000) };
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        const FLUSH_CHARS = 400;   // 每攒够 400 字符发一次，减少 IPC 调用次数
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                if (buf) await send_chunk(args.chat_id, buf);
                break;
            }
            buf += decoder.decode(value, { stream: true });
            // 遇到完整 SSE 消息边界（\n\n）或缓冲够了就立刻发送
            if (buf.includes('\n\n') && buf.length >= FLUSH_CHARS) {
                await send_chunk(args.chat_id, buf);
                buf = '';
            }
        }
        clearTimeout(timer);
        return { status: 200, body: '__DONE__' };
    } catch(e) {
        clearTimeout(timer);
        return { status: 0, body: 'JS error: ' + e.message };
    }
}
"""

JS_STREAM_FULL = """
async (args) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1800000);  // 1800s timeout
    try {
        const res = await fetch(args.url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + args.token
            },
            body: args.payload_json,
            signal: controller.signal
        });
        if (!res.ok) {
            const t = await res.text();
            clearTimeout(timer);
            return { status: res.status, body: t.substring(0, 2000) };
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let body = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            body += decoder.decode(value, { stream: true });
        }
        clearTimeout(timer);
        return { status: res.status, body: body };
    } catch(e) {
        clearTimeout(timer);
        return { status: 0, body: 'JS error: ' + e.message };
    }
}
"""

_CAMOUFOX_OPTS = {
    "headless": True,
    "humanize": False,
    "i_know_what_im_doing": True,
    "firefox_user_prefs": {
        "layers.acceleration.disabled": True,
        "gfx.webrender.enabled": False,
        "gfx.webrender.all": False,
        "gfx.webrender.software": False,
        "gfx.canvas.azure.backends": "skia",
        "media.hardware-video-decoding.enabled": False,
    },
}

@asynccontextmanager
async def _new_browser():
    from camoufox.async_api import AsyncCamoufox
    async with AsyncCamoufox(**_CAMOUFOX_OPTS) as browser:
        yield browser

class BrowserEngine:
    def __init__(self, pool_size: int = 3, base_url: str = "https://chat.qwen.ai"):
        self.pool_size = pool_size
        self.base_url = base_url
        self._browser = None
        self._browser_cm = None
        self._pages: asyncio.Queue = asyncio.Queue()
        self._started = False
        self._ready = asyncio.Event()

    async def start(self):
        if self._started:
            return
        try:
            await self._start_camoufox()
        except Exception as e:
            log.error(f"[Browser] camoufox failed: {e}")
        finally:
            self._ready.set()

    async def _start_camoufox(self):
        await self._ensure_browser_installed()
        from camoufox.async_api import AsyncCamoufox
        log.info("Starting browser engine (camoufox)...")
        self._browser_cm = AsyncCamoufox(**_CAMOUFOX_OPTS)
        self._browser = await self._browser_cm.__aenter__()
        await self._init_pages()
        self._started = True
        log.info("Browser engine started")

    async def _init_pages(self):
        log.info(f"[Browser] 正在初始化 {self.pool_size} 个并发渲染引擎页面...")
        self.stream_queues = {}  # chat_id -> asyncio.Queue()

        async def handle_chunk(chat_id, chunk):
            if chat_id in self.stream_queues:
                self.stream_queues[chat_id].put_nowait(chunk)

        for i in range(self.pool_size):
            page = await self._browser.new_page()
            try:
                await page.expose_function("send_chunk", handle_chunk)
            except Exception as e:
                log.error(f"[Browser] expose_function failed: {e}")

            try:
                await page.goto(self.base_url, wait_until="domcontentloaded", timeout=60000)
            except Exception:
                pass
            await asyncio.sleep(0.5)
            self._pages.put_nowait(page)
            log.info(f"  [Browser] Page {i+1}/{self.pool_size} ready (等待接入千问核心数据)")

    @staticmethod
    async def _ensure_browser_installed():
        import sys, subprocess
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "-m", "camoufox", "path"],
                    capture_output=True, text=True, timeout=10
                )
            )
            cache_dir = result.stdout.strip()
            if cache_dir:
                exe_name = "camoufox.exe" if os.name == "nt" else "camoufox"
                exe_path = os.path.join(cache_dir, exe_name)
                if os.path.exists(exe_path):
                    return
        except Exception:
            pass
        log.info("[Browser] 未检测到 camoufox，正在自动下载...")
        try:
            loop = asyncio.get_event_loop()
            def _do_install():
                from camoufox.pkgman import CamoufoxFetcher
                CamoufoxFetcher().install()
            await loop.run_in_executor(None, _do_install)
        except Exception as e:
            log.error(f"[Browser] 下载失败: {e}")

    async def stop(self):
        self._started = False
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._browser_cm:
            try:
                await self._browser_cm.__aexit__(None, None, None)
            except Exception:
                pass

    async def api_call(self, method: str, path: str, token: str, body: dict = None) -> dict:
        await asyncio.wait_for(self._ready.wait(), timeout=300)
        if not self._started:
            return {"status": 0, "body": "Browser engine failed to start"}
        # Wait queue: timeout protects the system from hanging
        try:
            page = await asyncio.wait_for(self._pages.get(), timeout=60)
        except asyncio.TimeoutError:
            log.warning("[Browser] Queue timeout (60s) — No available pages.")
            return {"status": 429, "body": "Too Many Requests (Queue full)"}
            
        needs_refresh = False
        try:
            result = await page.evaluate(JS_FETCH, {
                "method": method, "url": path, "token": token,
                "body_json": json.dumps(body, ensure_ascii=False) if body else None,
            })
            if result.get("status") == 0 and result.get("body", "").startswith("JS error:"):
                needs_refresh = True
            return result
        except Exception as e:
            log.error(f"api_call error: {e}")
            needs_refresh = True
            return {"status": 0, "body": str(e)}
        finally:
            if needs_refresh:
                asyncio.create_task(self._refresh_page_and_return(page))
            else:
                self._pages.put_nowait(page)

    async def fetch_chat(self, token: str, chat_id: str, payload: dict, buffered: bool = False):
        await asyncio.wait_for(self._ready.wait(), timeout=300)
        if not self._started:
            yield {"status": 0, "body": "Browser engine failed to start"}
            return

        try:
            page = await asyncio.wait_for(self._pages.get(), timeout=60)
        except asyncio.TimeoutError:
            log.warning("[Browser] Fetch chat queue timeout (60s) — No available pages.")
            yield {"status": 429, "body": "Too Many Requests (Queue full)"}
            return

        needs_refresh = False
        url = f'/api/v2/chat/completions?chat_id={chat_id}'

        # ── 缓冲模式（工具调用）：一次 JS evaluate，一次 IPC，最快 ──────────
        if buffered:
            try:
                res = await asyncio.wait_for(
                    page.evaluate(JS_STREAM_FULL, {"url": url, "token": token, "payload_json": json.dumps(payload, ensure_ascii=False)}),
                    timeout=1800,
                )
                if res.get("status") != 200:
                    log.warning(f"[Browser] buffered JS Error: {res.get('body','')[:100]}")
                    needs_refresh = True
                yield res
            except asyncio.TimeoutError:
                needs_refresh = True
                yield {"status": 0, "body": "Timeout"}
            except Exception as e:
                needs_refresh = True
                yield {"status": 0, "body": str(e)}
            finally:
                if needs_refresh:
                    asyncio.create_task(self._refresh_page_and_return(page))
                else:
                    self._pages.put_nowait(page)
            return

        # ── 流式模式（普通对话）：send_chunk IPC 实时转发 ────────────────────
        queue: asyncio.Queue = asyncio.Queue()
        self.stream_queues[chat_id] = queue
        try:
            # 启动 JS 流式读取（不 await），同时从队列实时 yield 每个 chunk
            js_task = asyncio.create_task(
                page.evaluate(JS_STREAM_CHUNKED, {
                    "url": url, "token": token,
                    "payload_json": json.dumps(payload, ensure_ascii=False),
                    "chat_id": chat_id
                })
            )
            # 从队列实时转发 chunk，直到 JS 任务结束且队列清空
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=120)
                    yield {"status": "streamed", "chunk": chunk}
                except asyncio.TimeoutError:
                    if js_task.done():
                        break
                    log.warning(f"[Browser] 流式读取超时等待 chunk (chat_id={chat_id})")
                    needs_refresh = True
                    break
                if js_task.done() and queue.empty():
                    break

            # 获取 JS 任务最终返回值（状态码 + 错误信息）
            if not js_task.done():
                try:
                    final = await asyncio.wait_for(js_task, timeout=10)
                except Exception:
                    final = {"status": 0, "body": "task cancelled"}
            else:
                final = js_task.result() if not js_task.exception() else {"status": 0, "body": str(js_task.exception())}

            if isinstance(final, dict) and final.get("status") not in (200, "streamed"):
                log.warning(f"[Browser] JS 返回非200: {final.get('body','')[:100]}")
                if final.get("status") != 200 and final.get("body") != "__DONE__":
                    needs_refresh = True
                    yield final
        except asyncio.TimeoutError:
            needs_refresh = True
            yield {"status": 0, "body": "Timeout"}
        except Exception as e:
            needs_refresh = True
            yield {"status": 0, "body": str(e)}
        finally:
            self.stream_queues.pop(chat_id, None)
            if needs_refresh:
                asyncio.create_task(self._refresh_page_and_return(page))
            else:
                self._pages.put_nowait(page)

    async def _refresh_page(self, page):
        try:
            await asyncio.wait_for(
                page.goto(self.base_url, wait_until="domcontentloaded"),
                timeout=20000,
            )
        except Exception:
            pass

    async def _refresh_page_and_return(self, page):
        await self._refresh_page(page)
        self._pages.put_nowait(page)
