import asyncio
import logging
from backend.core.account_pool import AccountPool, Account
from backend.core.browser_engine import _new_browser
from backend.core.config import settings
from backend.core.account_pool import Account
import logging
import asyncio
import random
import string
import time
import json
import re
from typing import Optional
from camoufox.async_api import AsyncCamoufox

log = logging.getLogger(__name__)

BASE_URL = "https://chat.qwenlm.ai"

def _new_browser():
    return AsyncCamoufox(
        headless=True,
        enable_cache=True,
        block_images=True,
        i_know_what_im_doing=True,
        os=["windows"],
        windowsize=(1920, 1080)
    )

async def get_fresh_token(email: str, password: str) -> str:
    """如果提供了此功能，用 playwright 重新登录获取 Token，这里提供一个 mock 或抛错以防未实现"""
    raise NotImplementedError("Auto-login not fully implemented yet in the separated architecture")

def _gen_password(length=14):
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pwd = "".join(random.choices(chars, k=length))
        if (any(c.isupper() for c in pwd) and any(c.islower() for c in pwd)
                and any(c.isdigit() for c in pwd) and any(c in "!@#$%^&*" for c in pwd)):
            return pwd

def _gen_username():
    first = random.choice(["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
                            "Drew", "Avery", "Quinn", "Blake", "Sage", "Reese", "Dakota", "Emery"])
    last = random.choice(["Smith", "Brown", "Wilson", "Lee", "Chen", "Wang", "Kim", "Park",
                           "Davis", "Miller", "Garcia", "Martinez", "Anderson", "Taylor", "Thomas"])
    return f"{first} {last}"

MAIL_BASE = "https://mail.chatgpt.org.uk"

class _EmailSession:
    def __init__(self):
        from curl_cffi import requests as cffi_requests
        self._session = cffi_requests.Session(impersonate="chrome")
        self._session.headers.update({
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self._current_token = ""
        self._token_expires_at = 0
        self._initialized = False

    def _init_session(self) -> bool:
        try:
            resp = self._session.get(f"{MAIL_BASE}/", timeout=15)
            if resp.status_code != 200:
                return False
            match = re.search(r'window\.__BROWSER_AUTH\s*=\s*(\{[^}]+\})', resp.text)
            if match:
                auth_data = json.loads(match.group(1))
                self._current_token = auth_data.get("token", "")
                self._token_expires_at = auth_data.get("expires_at", 0)
                self._initialized = True
                return True
            return False
        except Exception as e:
            log.warning(f"[MailSession] init error: {e}")
            return False

    def _ensure_token(self) -> bool:
        if not self._initialized or not self._current_token or time.time() > self._token_expires_at - 120:
            return self._init_session()
        return True

    def get_email(self) -> str:
        if not self._ensure_token():
            raise Exception("mail.chatgpt.org.uk: session init failed")
        resp = self._session.get(
            f"{MAIL_BASE}/api/generate-email",
            headers={"accept": "*/*", "referer": f"{MAIL_BASE}/",
                     "x-inbox-token": self._current_token},
            timeout=15,
        )
        if resp.status_code == 401:
            self._initialized = False
            self._init_session()
            resp = self._session.get(
                f"{MAIL_BASE}/api/generate-email",
                headers={"accept": "*/*", "referer": f"{MAIL_BASE}/",
                         "x-inbox-token": self._current_token},
                timeout=15,
            )
        data = resp.json()
        if not data.get("success"):
            raise Exception(f"mail.chatgpt.org.uk: generate-email failed: {data}")
        email = str(data.get("data", {}).get("email", "")).strip()
        new_tok = data.get("auth", {}).get("token", "")
        if new_tok:
            self._current_token = new_tok
            self._token_expires_at = data.get("auth", {}).get("expires_at", 0)
        return email

    def poll_verify_link(self, email: str, timeout_sec: int = 300) -> str:
        keywords = ("qwen", "verify", "activate", "confirm", "aliyun", "alibaba", "qwenlm")
        log.info(f"[MailSession] Polling inbox for {email} (timeout {timeout_sec}s)...")
        deadline = time.time() + timeout_sec
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                resp = self._session.get(
                    f"{MAIL_BASE}/api/emails",
                    params={"email": email},
                    headers={"accept": "*/*", "referer": f"{MAIL_BASE}/",
                             "x-inbox-token": self._current_token},
                    timeout=15,
                )
                if resp.status_code == 401:
                    self._initialized = False
                    self._init_session()
                    time.sleep(3)
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("auth", {}).get("token"):
                        self._current_token = data["auth"]["token"]
                        self._token_expires_at = data["auth"].get("expires_at", 0)
                    emails_list = data.get("data", {}).get("emails", [])
                    log.info(f"[MailSession] 第{attempt}次轮询，收件箱邮件数: {len(emails_list)}")
                    for msg in emails_list:
                        subject = str(msg.get("subject", ""))
                        parts = []
                        for field in ("html_content", "content", "body", "html", "text", "raw"):
                            v = msg.get(field)
                            if v: parts.append(str(v))
                        for field in ("payload", "data", "message"):
                            v = msg.get(field)
                            if isinstance(v, dict): parts.extend(str(x) for x in v.values() if x)
                            elif isinstance(v, str) and v: parts.append(v)
                        combined = " ".join(parts)
                        combined = (combined.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                                    .replace("\\u003c", "<").replace("\\u003e", ">")
                                    .replace("\\u0026", "&").replace("\\/", "/"))
                        all_links = re.findall(r'https?://[^\s"\'<>\\,\)]+', combined)
                        for link in all_links:
                            link = link.rstrip(".,;)")
                            if any(kw in link.lower() for kw in keywords):
                                log.info(f"[MailSession] 找到验证链接: {link[:120]}...")
                                return link
                        if any(kw in subject.lower() for kw in keywords) and all_links:
                            return all_links[0]
                else:
                    log.warning(f"[MailSession] 邮件API HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                log.warning(f"[MailSession] 轮询异常: {e}")
            time.sleep(5)
        log.error("[MailSession] 超时：未收到验证邮件")
        return ""

class _AsyncMailClient:
    def __init__(self):
        self._sess: Optional[_EmailSession] = None
        self._email = ""

    async def __aenter__(self):
        self._sess = await asyncio.to_thread(_EmailSession)
        return self

    async def __aexit__(self, *args):
        pass

    async def generate_email(self) -> str:
        self._email = await asyncio.to_thread(self._sess.get_email)
        return self._email

    async def get_verify_link(self, timeout_sec: int = 300) -> str:
        return await asyncio.to_thread(self._sess.poll_verify_link, self._email, timeout_sec)

async def register_qwen_account() -> Optional[Account]:
    log.info("[Register] ── 开始注册流程 ──")
    async with _AsyncMailClient() as mail_client:
        log.info("[Register] [1/7] 生成临时邮箱...")
        email = await mail_client.generate_email()
        password = _gen_password()
        username = _gen_username()
        log.info(f"[Register] [1/7] 邮箱: {email}  用户名: {username}")

        async with _new_browser() as browser:
            page = await browser.new_page()
            log.info(f"[Register] [2/7] 打开注册页面: {BASE_URL}/auth?action=signup")
            try:
                await page.goto(f"{BASE_URL}/auth?action=signup", wait_until="domcontentloaded", timeout=60000)
            except Exception:
                pass

            log.info("[Register] [3/7] 填写注册表单...")
            name_input = None
            for sel in ['input[placeholder*="Full Name"]', 'input[placeholder*="Name"]']:
                try:
                    name_input = await page.wait_for_selector(sel, timeout=15000)
                    if name_input: break
                except Exception:
                    pass
            if not name_input:
                inputs = await page.query_selector_all('input')
                name_input = inputs[0] if len(inputs) >= 4 else None
            if not name_input:
                log.error("[Register] [3/7] 找不到姓名输入框，注册中止")
                return None

            await name_input.click(); await name_input.fill(username)
            email_input = await page.query_selector('input[placeholder*="Email"]')
            if not email_input:
                inputs = await page.query_selector_all('input')
                email_input = inputs[1] if len(inputs) >= 2 else None
            if email_input: await email_input.click(); await email_input.fill(email)

            pwd_input = await page.query_selector('input[placeholder*="Password"]:not([placeholder*="Again"])')
            if not pwd_input:
                inputs = await page.query_selector_all('input')
                pwd_input = inputs[2] if len(inputs) >= 3 else None
            if pwd_input: await pwd_input.click(); await pwd_input.fill(password)

            confirm_input = await page.query_selector('input[placeholder*="Again"]')
            if not confirm_input:
                inputs = await page.query_selector_all('input')
                confirm_input = inputs[3] if len(inputs) >= 4 else None
            if confirm_input: await confirm_input.click(); await confirm_input.fill(password)

            checkbox = await page.query_selector('input[type="checkbox"]')
            if checkbox and not await checkbox.is_checked(): await checkbox.click()
            else:
                agree = await page.query_selector('text=I agree')
                if agree: await agree.click()

            log.info("[Register] [4/7] 提交注册表单...")
            await asyncio.sleep(1)
            submit = await page.query_selector('button:has-text("Create Account")') or await page.query_selector('button[type="submit"]')
            if submit: await submit.click()
            await asyncio.sleep(6)

            url_after = page.url
            token = None
            if BASE_URL in url_after and "auth" not in url_after:
                await asyncio.sleep(3)
                token = await page.evaluate("localStorage.getItem('token')")

            if not token:
                log.info("[Register] [5/7] 尝试用账号密码直接登录...")
                try:
                    await page.goto(f"{BASE_URL}/auth", wait_until="domcontentloaded", timeout=30000)
                    await asyncio.sleep(3)
                    li_email = await page.query_selector('input[placeholder*="Email"]')
                    if li_email: await li_email.fill(email)
                    li_pwd = await page.query_selector('input[type="password"]')
                    if li_pwd: await li_pwd.fill(password)
                    li_btn = await page.query_selector('button:has-text("Log in")') or await page.query_selector('button[type="submit"]')
                    if li_btn: await li_btn.click()
                    await asyncio.sleep(8)
                    token = await page.evaluate("localStorage.getItem('token')")
                except Exception as e:
                    pass

            if not token:
                log.info("[Register] [6/7] 等待验证邮件（最多5分钟）...")
                verify_link = await mail_client.get_verify_link(timeout_sec=300)

                if not verify_link:
                    log.error("[Register] [6/7] 未收到验证邮件，注册失败")
                    return None

                try:
                    await page.goto(verify_link, wait_until="domcontentloaded", timeout=30000)
                except Exception: pass
                await asyncio.sleep(6)
                token = await page.evaluate("localStorage.getItem('token')")

                if not token:
                    try:
                        await page.goto(f"{BASE_URL}/auth", wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(3)
                        li_email = await page.query_selector('input[placeholder*="Email"]')
                        if li_email: await li_email.fill(email)
                        li_pwd = await page.query_selector('input[type="password"]')
                        if li_pwd: await li_pwd.fill(password)
                        li_btn = await page.query_selector('button:has-text("Log in")') or await page.query_selector('button[type="submit"]')
                        if li_btn: await li_btn.click()
                        await asyncio.sleep(8)
                        token = await page.evaluate("localStorage.getItem('token')")
                    except Exception: pass

            if not token:
                log.error("[Register] 所有方法均无法获取token，注册失败")
                return None

            log.info("[Register] [7/7] 提取 cookies...")
            all_cookies = await page.context.cookies()
            cookie_str = "; ".join(f"{c.get('name','')}={c.get('value','')}" for c in all_cookies if "qwen" in c.get("domain", ""))
            log.info(f"[Register] ✓ 注册完成: {email}")
            return Account(email=email, password=password, token=token, cookies=cookie_str, username=username)

async def activate_account(acc: Account) -> bool:
    """尝试用临时邮箱去收激活邮件并点击链接，由于和 register 基本一致，简略实现"""
    return False

class AuthResolver:
    """自动登录并提取 Token，在检测到 401 时自动自愈凭证"""
    def __init__(self, pool: AccountPool):
        self.pool = pool

    async def refresh_token(self, acc: Account) -> bool:
        if not acc.email or not acc.password:
            log.warning(f"[Auth] 账号 {acc.email} 缺少密码，无法自愈。")
            return False
            
        log.info(f"[Auth] 正在启动独立浏览器为 {acc.email} 自动刷新 Token...")
        try:
            async with _new_browser() as browser:
                page = await browser.new_page()
                await page.goto("https://chat.qwen.ai/auth", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(3)
                
                # 填写邮箱密码
                li_email = await page.query_selector('input[placeholder*="Email"]')
                if li_email: await li_email.fill(acc.email)
                li_pwd = await page.query_selector('input[type="password"]')
                if li_pwd: await li_pwd.fill(acc.password)
                
                # 提交
                li_btn = (await page.query_selector('button:has-text("Log in")') or
                          await page.query_selector('button[type="submit"]'))
                if li_btn: await li_btn.click()
                
                await asyncio.sleep(8)
                
                # 提取 LocalStorage Token
                new_token = await page.evaluate("localStorage.getItem('token')")
                if new_token and new_token != acc.token:
                    acc.token = new_token
                    acc.valid = True
                    await self.pool.save()
                    log.info(f"[Auth] 自愈成功，{acc.email} 获得全新 Token。")
                    return True
                elif new_token == acc.token:
                    acc.valid = True
                    log.info(f"[Auth] {acc.email} 重新校验成功。")
                    return True
                else:
                    log.error(f"[Auth] {acc.email} 登录失败或遭遇滑块验证拦截。")
                    return False
        except Exception as e:
            log.error(f"[Auth] 自愈流程异常: {e}")
            return False
