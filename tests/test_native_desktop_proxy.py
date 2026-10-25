"""Desktop auth through the Railway edge; no listening server or model calls.

Include the pinned Hermes checkout on PYTHONPATH for the real PKCE/provider tests.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import secrets
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit
import uuid

import httpx
from starlette.websockets import WebSocket


ROOT = Path(__file__).resolve().parents[1]


class ProxyFixture:
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hermes-desktop-proxy-")
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ, {
            "HERMES_HOME": temporary.name,
            "ADMIN_PASSWORD": "synthetic-password",
            "RAILWAY_PUBLIC_DOMAIN": "hermes.example.invalid",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        spec = importlib.util.spec_from_file_location(
            f"desktop_proxy_{uuid.uuid4().hex}", ROOT / "server.py")
        self.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.server)
        self.server.preparation["state"] = "ready"
        self.requests = []
        self.gate = True
        self.private = lambda request: httpx.Response(401, json={"detail": "Unauthorized"})

        async def backend(request):
            self.requests.append(request)
            if request.url.path == "/api/status":
                return httpx.Response(200, json={"auth_required": self.gate,
                    "auth_providers": ["basic"], "auth_flows": ["cookie", "native_pkce"]})
            return self.private(request)

        self.server._http_client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        self.addAsyncCleanup(self.server._http_client.aclose)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.app),
                                      base_url="https://hermes.example.invalid")
        self.addAsyncCleanup(self.client.aclose)
        self.server.hermes_session._cookies = {"__Host-hermes_session_at": "internal-admin"}
        self.server.hermes_session.refresh = AsyncMock(return_value=True)
        self.server._http_client.cookies.set("__Host-hermes_session_at", "jar-admin",
                                            domain="127.0.0.1", path="/")

    def authenticate_admin(self):
        self.client.cookies.set("hermes_auth", self.server._make_auth_token())


class AdminLoginHardeningTests(ProxyFixture, unittest.IsolatedAsyncioTestCase):
    async def test_setup_login_throttles_parallel_attempts_even_with_spoofed_ips(self):
        responses = await asyncio.gather(*[
            self.client.post("/setup/login", data={"username": "wrong", "password": "wrong"},
                             headers={"x-forwarded-for": f"192.0.2.{index}"})
            for index in range(20)])
        self.assertEqual(sum(r.status_code == 302 for r in responses), 10)
        self.assertEqual(sum(r.status_code == 429 for r in responses), 10)
        self.assertEqual(len(self.server._login_attempts), 10)
        for response in responses:
            self.assertNotIn("set-cookie", response.headers)
            self.assertEqual(response.headers["cache-control"], "no-store")
            if response.status_code == 429:
                self.assertGreater(int(response.headers["retry-after"]), 0)

    async def test_login_budget_recovers_after_window_and_counts_valid_attempts(self):
        data = {"username": "admin", "password": "synthetic-password"}
        with patch.object(self.server.time, "monotonic", return_value=1000):
            for _ in range(10):
                self.assertEqual((await self.client.post("/setup/login", data=data)).status_code, 302)
            self.assertEqual((await self.client.post("/setup/login", data=data)).status_code, 429)
        with patch.object(self.server.time, "monotonic", return_value=1060):
            self.assertEqual((await self.client.post("/setup/login", data=data)).status_code, 302)

    async def test_admin_cookie_is_secure_behind_railway_tls_and_auth_is_not_cached(self):
        for base_url in ("http://hermes.example.invalid", "https://hermes.example.invalid"):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.app),
                                         base_url=base_url) as client:
                response = await client.post("/setup/login", data={
                    "username": "admin", "password": "synthetic-password", "returnTo": "//evil.invalid"})
                self.assertEqual(response.headers["location"], "/")
                self.assertEqual(response.headers["cache-control"], "no-store")
                cookie = response.headers["set-cookie"].lower()
                for flag in ("secure", "httponly", "samesite=lax"):
                    self.assertIn(flag, cookie)
        self.client.cookies.clear()
        self.assertEqual((await self.client.get("/setup/login")).headers["cache-control"], "no-store")

    async def test_unicode_and_empty_invalid_credentials_fail_without_server_error(self):
        for data in ({}, {"username": "\u2603", "password": "\U0001f512"}):
            response = await self.client.post("/setup/login", data=data)
            self.assertEqual(response.status_code, 302)
            self.assertIn("error=1", response.headers["location"])
            self.assertNotIn("set-cookie", response.headers)


class DesktopProxyTests(ProxyFixture, unittest.IsolatedAsyncioTestCase):
    async def test_anonymous_discovery_does_not_borrow_admin_cookies(self):
        response = await self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("native_pkce", response.json()["auth_flows"])
        self.assertTrue(all("cookie" not in r.headers for r in self.requests))

    async def test_missing_and_invalid_credentials_never_trigger_internal_login(self):
        for headers in ({}, {"authorization": "Bearer invalid"},
                        {"cookie": "hermes_auth=invalid"}):
            response = await self.client.get("/api/private", headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertNotIn("admin", self.requests[-1].headers.get("cookie", ""))
        self.server.hermes_session.refresh.assert_not_awaited()

    async def test_native_bearer_and_cookie_credentials_are_forwarded_unchanged(self):
        self.private = lambda request: httpx.Response(200, json={
            "bearer": request.headers.get("authorization"),
            "cookie": request.headers.get("cookie")})
        response = await self.client.get("/api/private", headers={
            "authorization": "Bearer caller", "cookie": "__Host-hermes_session_at=caller",
            "x-forwarded-prefix": "/untrusted"})
        self.assertEqual(response.json(), {"bearer": "Bearer caller",
                                          "cookie": "__Host-hermes_session_at=caller"})
        self.assertNotIn("x-forwarded-prefix", self.requests[-1].headers)

    async def test_setup_and_root_still_require_the_admin_cookie(self):
        self.client.headers["authorization"] = "Bearer caller"
        for path in ("/setup", "/setup/api/config", "/setup/api/backup/jobs"):
            self.assertEqual((await self.client.get(path)).status_code, 401)
        self.assertTrue((await self.client.get("/", headers={"accept": "text/html"})).headers["location"].startswith("/setup/login"))
        self.assertEqual(self.requests, [])

    async def test_gate_off_missing_domain_and_preparation_fail_closed(self):
        for gate in (False, None, "true"):
            self.gate = gate
            self.assertEqual((await self.client.get("/api/private")).status_code, 503)
            self.assertEqual((await self.client.get("/login")).status_code, 503)
        self.assertTrue(all(r.url.path == "/api/status" for r in self.requests))
        self.requests.clear()
        self.gate = True
        for state in ("running", "failed"):
            self.server.preparation["state"] = state
            self.assertEqual((await self.client.get("/auth/native/authorize")).status_code, 503)
        self.server.preparation["state"] = "ready"
        with patch.dict(os.environ, {"RAILWAY_PUBLIC_DOMAIN": ""}):
            self.assertEqual((await self.client.get("/api/status")).status_code, 503)
        self.assertEqual(self.requests, [])

    async def test_unavailable_or_malformed_discovery_does_not_forward_private_requests(self):
        for outcome in (httpx.Response(200, text="not JSON"), httpx.Response(200, json=[]),
                        httpx.Response(503), httpx.ConnectError("offline")):
            async def failure(request):
                self.assertEqual(request.url.path, "/api/status")
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
            async with httpx.AsyncClient(transport=httpx.MockTransport(failure)) as client:
                with patch.object(self.server, "get_http_client", return_value=client):
                    self.assertEqual((await self.client.get("/api/private")).status_code, 503)

    async def test_existing_admin_requests_keep_the_internal_session(self):
        self.authenticate_admin()
        self.private = lambda request: httpx.Response(200, json={"cookie": request.headers.get("cookie")})
        response = await self.client.get("/api/private")
        self.assertIn("__Host-hermes_session_at=internal-admin", response.json()["cookie"])
        self.assertEqual(len(self.requests), 1)

    async def test_native_settings_writes_recheck_preparation_after_maintenance(self):
        self.private = lambda request: httpx.Response(200)
        async with self.server.lifecycle_lock:
            action = asyncio.create_task(self.client.put("/api/config", json={}))
            # The first discovery must finish before this request waits on the lock.
            while not self.requests:
                await asyncio.sleep(0)
            self.server.preparation["state"] = "running"
        response = await action
        self.assertEqual(response.status_code, 503)
        self.assertTrue(all(r.url.path == "/api/status" for r in self.requests))

    async def test_auth_responses_keep_individual_cookies_and_native_login_page(self):
        self.authenticate_admin()
        self.client.cookies.set("__Host-hermes_session_pkce", "broker")
        self.private = lambda request: httpx.Response(200, text="native login", headers=[
            ("set-cookie", "first=one; Path=/; Secure; HttpOnly"),
            ("set-cookie", "second=two; Path=/; Secure; HttpOnly")])
        response = await self.client.get("/login")
        self.assertEqual(response.text, "native login")
        self.assertEqual(len(response.headers.get_list("set-cookie")), 2)
        self.assertNotIn("internal-admin", self.requests[-1].headers.get("cookie", ""))
        self.client.cookies.clear()
        self.assertEqual((await self.client.get("/login")).text, "native login")


@unittest.skipUnless(importlib.util.find_spec("hermes_cli"), "requires pinned Hermes source on PYTHONPATH")
class PinnedHermesAuthTests(ProxyFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        from fastapi import FastAPI, Request
        from hermes_cli.dashboard_auth import clear_providers, register_provider
        from hermes_cli.dashboard_auth.middleware import gated_auth_middleware
        from hermes_cli.dashboard_auth.routes import router, _reset_password_rate_limit
        from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
        from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
        clear_providers()
        _reset_password_rate_limit()
        _reset_for_tests()
        self.addCleanup(clear_providers)
        self.provider = BasicAuthProvider(username="admin", password_hash=hash_password("synthetic-password"),
                                          secret=b"synthetic-test-secret-at-least-32-bytes")
        register_provider(self.provider)
        upstream = FastAPI()
        upstream.state.auth_required = True
        upstream.include_router(router)
        upstream.middleware("http")(gated_auth_middleware)

        @upstream.get("/api/status")
        async def status():
            return {"auth_required": True, "auth_providers": ["basic"],
                    "auth_flows": ["cookie", "native_pkce"]}

        # FastAPI resolves postponed Request annotations in the module globals.
        globals()["Request"] = Request

        @upstream.get("/api/private")
        async def private(request: Request):
            return {"user_id": request.state.session.user_id}

        # Model the trusted Railway -> loopback proxy scheme normalization.
        async def https_backend(scope, receive, send):
            scope = dict(scope, scheme="https")
            await upstream(scope, receive, send)

        await self.server._http_client.aclose()
        self.server._http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=https_backend))
        self.addAsyncCleanup(self.server._http_client.aclose)

    async def native_login(self):
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        response = await self.client.get("/auth/native/authorize", params={
            "provider": "basic", "code_challenge": challenge, "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:55555/callback", "state": "synthetic-state"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "/login")
        self.assertEqual((await self.client.get("/login")).status_code, 200)
        response = await self.client.post("/auth/password-login", json={
            "provider": "basic", "username": "admin", "password": "synthetic-password"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.headers.get_list("set-cookie")), 3)
        callback = parse_qs(urlsplit(response.json()["next"]).query)
        self.assertEqual(callback["state"], ["synthetic-state"])
        body = {"code": callback["code"][0], "code_verifier": verifier}
        response = await self.client.post("/auth/native/token", json=body)
        self.assertEqual(response.status_code, 200)
        return response.json(), body

    async def test_real_native_pkce_bearer_refresh_and_code_replay(self):
        # The user's default browser can already be logged in to Setup.
        self.authenticate_admin()
        tokens, code_body = await self.native_login()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.app),
                                     base_url="https://hermes.example.invalid") as desktop:
            headers = {"authorization": "Bearer " + tokens["access_token"]}
            response = await desktop.get("/api/private", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["user_id"], "admin")
            ticket = await desktop.post("/api/auth/ws-ticket", headers=headers)
            self.assertEqual(ticket.status_code, 200)
            self.ticket = ticket.json()["ticket"]
            refreshed = await desktop.post("/auth/native/refresh", json={
                "provider": "basic", "refresh_token": tokens["refresh_token"]})
            self.assertEqual(refreshed.status_code, 200)
            replay = await desktop.post("/auth/native/token", json=code_body)
            self.assertEqual(replay.status_code, 400)
            self.assertEqual((await desktop.get("/api/private")).status_code, 401)
        self.server.hermes_session.refresh.assert_not_awaited()
        await self.assert_websocket_ticket()

    async def assert_websocket_ticket(self):
        from hermes_cli.dashboard_auth.ws_tickets import consume_ticket, TicketInvalid

        async def connect(url, **kwargs):
            ticket = parse_qs(urlsplit(url).query).get("ticket", [""])[0]
            try:
                consume_ticket(ticket)
            except TicketInvalid:
                raise self.server.websockets.exceptions.InvalidHandshake("Invalid ticket")

            class Upstream:
                close_code = 1000
                close = AsyncMock()
                send = AsyncMock()

                async def __aiter__(self):
                    yield "connected"
            return Upstream()

        for query, expected in (("ticket=" + self.ticket, "websocket.accept"),
                                ("ticket=" + self.ticket, "websocket.close"),
                                ("", "websocket.close")):
            messages = iter([{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000}])
            send = AsyncMock()
            socket = WebSocket({"type": "websocket", "scheme": "wss", "path": "/api/ws",
                "query_string": query.encode(), "headers": [], "server": ("hermes.example.invalid", 443)},
                receive=AsyncMock(side_effect=lambda: next(messages)), send=send)
            with patch.object(self.server.websockets, "connect", side_effect=connect):
                await self.server.ws_proxy(socket)
            self.assertEqual(send.await_args_list[0].args[0]["type"], expected)

    async def test_windows_password_window_recognizes_its_own_session_cookie(self):
        # Desktop 0.22.5 selects the embedded cookie flow for password-only
        # gateways. It opens /login with no PKCE cookie and waits for a Hermes
        # session cookie, not the template's unrelated hermes_auth cookie.
        for prior_admin_session in (False, True):
            self.client.cookies.clear()
            if prior_admin_session:
                self.authenticate_admin()
            page = await self.client.get("/login")
            self.assertEqual(page.status_code, 200)
            self.assertIn("/auth/password-login", page.text)
            response = await self.client.post("/auth/password-login", json={
                "provider": "basic", "username": "admin", "password": "synthetic-password"})
            self.assertEqual(response.status_code, 200)
            names = {cookie.name for cookie in self.client.cookies.jar}
            self.assertIn("__Host-hermes_session_at", names)
            self.assertIn("__Host-hermes_session_rt", names)
            # Use only the cookies Desktop recognizes, even if its window had
            # a leftover template cookie from the unsuccessful first attempt.
            cookies = httpx.Cookies()
            for cookie in self.client.cookies.jar:
                if cookie.name.startswith("__Host-hermes_session_"):
                    cookies.jar.set_cookie(cookie)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.app),
                    base_url="https://hermes.example.invalid", cookies=cookies) as desktop:
                identity = await desktop.get("/api/auth/me")
                self.assertEqual(identity.status_code, 200)
                self.assertEqual(identity.json()["user_id"], "admin")
                ticket = await desktop.post("/api/auth/ws-ticket")
                self.assertEqual(ticket.status_code, 200)
                self.ticket = ticket.json()["ticket"]
                await self.assert_websocket_ticket()
        self.server.hermes_session.refresh.assert_not_awaited()

    async def test_wrong_password_invalid_bearer_and_legacy_cookie_login(self):
        response = await self.client.post("/auth/password-login", json={
            "provider": "basic", "username": "admin", "password": "wrong"})
        self.assertEqual(response.status_code, 401)
        response = await self.client.post("/auth/password-login", json={
            "provider": "basic", "username": "admin", "password": "synthetic-password"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.headers.get_list("set-cookie")), 3)
        self.assertEqual((await self.client.get("/api/private")).status_code, 200)
        response = await self.client.get("/api/private", headers={"authorization": "Bearer invalid"})
        self.assertEqual(response.status_code, 401)
        self.server.hermes_session.refresh.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
