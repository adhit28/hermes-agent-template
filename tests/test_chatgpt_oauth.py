"""ChatGPT setup must preserve existing data through failed and stale requests."""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from starlette.requests import Request

from test_v2026_9_24_config_safety import ConfigSafetyFixture


def request(body=None):
    return SimpleNamespace(json=AsyncMock(return_value=body or {}))


def payload(response):
    return json.loads(response.body)


def token(expiry=3600):
    claims = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + expiry}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{claims}.synthetic"


class ChatGPTTests(ConfigSafetyFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.real_guard = self.server.guard
        self.server.guard = lambda _: None
        self.config = self.home / "config.yaml"
        self.env = self.home / ".env"
        self.auth = self.home / "auth.json"

    def seed(self, provider="minimax", model="MiniMax-M2.7"):
        self.config.write_text(f"# preserve me\nmodel:\n  provider: {provider}\n  default: {model}\nmcp_servers:\n  keep:\n    command: preserved\n")
        self.env.write_text("# existing credentials\nMINIMAX_API_KEY=synthetic-minimax-key\nLLM_MODEL=MiniMax-M2.7\n")
        self.env.chmod(0o600)
        self.config.chmod(0o640)

    def snapshots(self):
        return tuple(self.server._file_snapshot(path) for path in (self.env, self.config, self.auth))

    def codex_save(self, **extra):
        return request({"vars": {"LLM_MODEL": "synthetic-codex-model"},
                        "_active_oauth_provider": "openai-codex", "_restart": False, **extra})

    def mock_codex(self):
        self.server._codex_connected = Mock(return_value=True)
        self.server._codex_models = AsyncMock(return_value=["synthetic-codex-model"])

    def assert_native_auth_lock_held(self):
        code = """
import sys
from pathlib import Path
from hermes_cli.auth import _auth_store_lock
try:
    with _auth_store_lock(timeout_seconds=.05, target_path=Path(sys.argv[1])):
        sys.exit(7)
except TimeoutError:
    sys.exit(0)
"""
        result = subprocess.run([sys.executable, "-E", "-s", "-c", code, str(self.auth)], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr.decode()[:500])

    def test_xai_save_and_rollback_hold_native_lock_against_other_processes(self):
        self.seed()
        self.auth.write_text('{"providers":{"openai-codex":{"tokens":{"access_token":"keep-access","refresh_token":"keep-refresh"}}}}')
        before = self.snapshots()
        def fail_config(*args, **kwargs):
            self.assert_native_auth_lock_held()
            raise ValueError("simulated config failure")
        self.server._apply_xai_oauth_config = Mock(side_effect=fail_config)
        with self.assertRaisesRegex(ValueError, "simulated config failure"):
            self.server._save_xai_oauth_bundle({"refresh_token": "new-xai"}, "grok-fixture")
        self.assertEqual(self.snapshots(), before)

    async def test_xai_disconnect_locks_shared_store_preserves_codex_and_keeps_loop_responsive(self):
        self.seed()
        codex = {"tokens": {"access_token": "keep-access", "refresh_token": "keep-refresh"}}
        self.auth.write_text(json.dumps({"providers": {"openai-codex": codex, "xai-oauth": {"tokens": {"refresh_token": "remove-xai"}}}}))
        write = self.server._atomic_write_text
        def checked_write(path, content):
            if path == self.auth:
                self.assert_native_auth_lock_held()
            write(path, content)
        self.server._atomic_write_text = checked_write
        deletion = asyncio.create_task(self.server.api_oauth_xai_delete(request()))
        await asyncio.sleep(.02)
        self.assertFalse(deletion.done())
        self.assertEqual((await self.server.route_health(request())).status_code, 200)
        self.assertEqual((await deletion).status_code, 200)
        self.assertEqual(json.loads(self.auth.read_text())["providers"], {"openai-codex": codex})
        self.assertEqual(self.server.read_env(self.env)["MINIMAX_API_KEY"], "synthetic-minimax-key")

    async def apply_model(self, provider, model, **kwargs):
        current = self.server._read_config_for_write(self.config)
        current["model"].update(provider=provider, default=model)
        self.server._write_config_atomic(self.config, current)

    def test_stored_status_handles_legacy_pool_and_expired_refreshable_grants_without_writes(self):
        self.seed("openai-codex", "synthetic-codex-model")
        for kind in ("legacy", "expired", "pool"):
            with self.subTest(kind=kind):
                value = token(-60 if kind == "expired" else 3600)
                state = {"version": 2, "providers": {"openai-codex": {"tokens": {
                    "access_token": value, "refresh_token": "synthetic-refresh"}}}}
                if kind == "pool":
                    state = {"version": 2, "credential_pool": {"openai-codex": [{
                        "id": "fixture", "label": "fixture", "auth_type": "oauth", "source": "device_code",
                        "access_token": value, "refresh_token": "synthetic-refresh"}]}}
                self.auth.write_text(json.dumps(state))
                before = self.auth.read_bytes()
                with patch("httpx.Client.post", side_effect=AssertionError("status must not refresh")):
                    self.assertTrue(self.server.is_config_complete())
                self.assertEqual(self.auth.read_bytes(), before)

    def test_dead_grant_is_not_revived_by_legacy_alias_and_quota_is_not_logout(self):
        self.seed("openai-codex", "synthetic-codex-model")
        grant = {"access_token": token(), "refresh_token": "synthetic-refresh"}
        row = {**grant, "last_status": "dead"}
        state = {"providers": {"openai-codex": {"tokens": grant}}, "credential_pool": {"openai-codex": [row]}}
        self.auth.write_text(json.dumps(state))
        self.assertFalse(self.server._codex_connected())
        row["last_status"] = "exhausted"
        self.auth.write_text(json.dumps(state))
        before = self.auth.read_bytes()
        self.assertTrue(self.server._codex_connected())
        self.assertEqual(self.auth.read_bytes(), before)

    def test_empty_codex_model_cannot_use_an_unrelated_env_model(self):
        self.seed("openai-codex", "''")
        self.server._codex_connected = Mock(return_value=True)
        self.assertFalse(self.server.is_config_complete())

    def test_existing_native_codex_gateway_key_still_works_without_being_shown_as_oauth(self):
        self.seed("openai-codex", "native-gateway-model")
        state = {"credential_pool": {"openai-codex": [{"auth_type": "api_key", "access_token": "synthetic-gateway-key", "base_url": "https://gateway.example.invalid"}]}}
        self.auth.write_text(json.dumps(state))
        before = self.auth.read_bytes()
        self.assertTrue(self.server.is_config_complete())
        self.assertFalse(self.server._codex_connected())
        self.assertEqual(self.auth.read_bytes(), before)

    def test_seeded_auto_model_alone_does_not_mark_api_setup_complete(self):
        self.seed("auto", "upstream-default-model")
        self.env.write_text("MINIMAX_API_KEY=synthetic-key\n")
        self.assertFalse(self.server.is_config_complete())

    def test_missing_or_damaged_codex_auth_cannot_borrow_an_unrelated_api_key(self):
        self.seed("openai-codex", "synthetic-codex-model")
        for content in (None, "{broken", '{"version":2,"providers":{}}'):
            with self.subTest(content=content):
                if content is None:
                    self.auth.unlink(missing_ok=True)
                else:
                    self.auth.write_text(content)
                self.assertFalse(self.server.is_config_complete())

    def test_restart_preserves_native_model_and_other_config(self):
        for provider, routing in (
            ("openai-codex", {"base_url": "https://chatgpt.com/backend-api/codex", "api_mode": "codex_responses"}),
            ("nvidia", {"base_url": "https://proxy.example.invalid/v1", "api_key": "synthetic-proxy-key", "api_mode": "chat_completions"}),
        ):
            with self.subTest(provider=provider):
                self.seed(provider, "native-selected-model")
                config = self.server._read_config_for_write(self.config)
                config["model"].update(routing)
                self.server._write_config_atomic(self.config, config)
                self.server.write_config_yaml(self.server.read_env(self.env), preserve_model=True)
                saved = self.server._read_config_for_write(self.config)
                self.assertEqual(saved["model"], config["model"])
                self.assertEqual(saved["mcp_servers"]["keep"]["command"], "preserved")
                self.assertEqual(self.server.read_env(self.env)["MINIMAX_API_KEY"], "synthetic-minimax-key")

    def test_restart_without_a_saved_model_still_clears_stale_routing(self):
        self.seed("auto", "''")
        config = self.server._read_config_for_write(self.config)
        config["model"].update(base_url="https://stale.example.invalid", api_key="synthetic-stale-key", api_mode="chat_completions")
        self.server._write_config_atomic(self.config, config)
        self.server.write_config_yaml(self.server.read_env(self.env), preserve_model=True)
        self.assertEqual(self.server._read_config_for_write(self.config)["model"],
                         {"provider": "auto", "default": "MiniMax-M2.7"})

    async def test_config_get_reports_native_model_and_revision_without_raw_keys(self):
        self.seed("openai-codex", "native-selected-model")
        self.mock_codex()
        result = payload(await self.server.api_config_get(request()))
        self.assertEqual(result["model_config"]["model"], "native-selected-model")
        self.assertEqual(result["config_revision"], self.server.config_revision())
        self.assertTrue(result["config_complete"])
        self.assertNotIn("synthetic-minimax-key", json.dumps(result))

    async def test_stale_form_cannot_overwrite_keys_or_model(self):
        self.seed()
        revision = self.server.config_revision()
        self.env.write_text("MINIMAX_API_KEY=newer-saved-key\nLLM_MODEL=newer-model\n")
        before = self.snapshots()
        result = await self.server.api_config_put(request({"vars": {"MINIMAX_API_KEY": "stale-key"}, "_config_revision": revision}))
        self.assertEqual(result.status_code, 409)
        self.assertEqual(self.snapshots(), before)

    async def test_get_rejects_a_concurrently_changed_snapshot(self):
        self.seed()
        self.server.config_revision = Mock(side_effect=["before", "after"])
        self.assertEqual((await self.server.api_config_get(request())).status_code, 409)

    async def test_old_browser_form_without_revision_cannot_replace_saved_keys(self):
        self.seed()
        before = self.snapshots()
        req = request({"vars": {"MINIMAX_API_KEY": "stale-key"}})
        req.headers = {"referer": "http://localhost/setup"}
        self.assertEqual((await self.server.api_config_put(req)).status_code, 409)
        self.assertEqual(self.snapshots(), before)

    async def test_legacy_model_only_edit_cannot_pair_an_api_model_with_codex(self):
        self.seed("openai-codex", "synthetic-codex-model")
        before = self.snapshots()
        req = request({"vars": {"LLM_MODEL": "unrelated-model"}, "_restart": False})
        self.assertEqual((await self.server.api_config_put(req)).status_code, 409)
        self.assertEqual(self.snapshots(), before)

    async def test_channel_only_save_preserves_native_model_and_masked_key(self):
        self.seed("openai-codex", "native-selected-model")
        config = self.server._read_config_for_write(self.config)
        config["model"].update(base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses")
        self.server._write_config_atomic(self.config, config)
        self.mock_codex()
        result = await self.server.api_config_put(request({"vars": {
            "MINIMAX_API_KEY": "synthet***", "EMAIL_ADDRESS": "fixture@example.invalid"}, "_restart": False}))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.server.configured_model(), ("native-selected-model", "openai-codex"))
        self.assertEqual(self.server._read_config_for_write(self.config)["model"], config["model"])
        self.assertEqual(self.server.read_env(self.env)["LLM_MODEL"], "native-selected-model")
        self.assertEqual(self.server.read_env(self.env)["MINIMAX_API_KEY"], "synthetic-minimax-key")

    async def test_codex_save_uses_native_model_assignment_and_keeps_other_keys(self):
        self.seed()
        self.mock_codex()
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=self.apply_model)
        self.auth.write_text('{"unrelated":"keep-exactly"}')
        auth_before = self.auth.read_bytes()
        result = await self.server.api_config_put(self.codex_save(_config_revision=self.server.config_revision()))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.server.configured_model(), ("synthetic-codex-model", "openai-codex"))
        self.assertEqual(self.server.read_env(self.env)["MINIMAX_API_KEY"], "synthetic-minimax-key")
        self.assertEqual(self.auth.read_bytes(), auth_before)
        self.server.set_active_model_via_hermes.assert_awaited_once_with("openai-codex", "synthetic-codex-model", base_url="", api_key="")

    async def test_failed_pin_restores_exact_previous_bytes_and_modes(self):
        self.seed()
        self.mock_codex()
        self.server.set_active_model_via_hermes = AsyncMock(return_value="unavailable")
        before = self.snapshots()
        result = await self.server.api_config_put(self.codex_save())
        self.assertEqual(result.status_code, 503)
        self.assertEqual(self.snapshots(), before)

    async def test_staged_oauth_save_keeps_the_previous_model_provider_pair_until_commit(self):
        self.seed()
        self.mock_codex()
        async def pin(provider, model, **kwargs):
            self.assertEqual(self.server.configured_model(), ("MiniMax-M2.7", "minimax"))
            await self.apply_model(provider, model)
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=pin)
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 200)

    async def test_staged_same_provider_save_preserves_complete_native_model_until_commit(self):
        self.seed("openai-codex", "previous-codex-model")
        config = self.server._read_config_for_write(self.config)
        config["model"].update(base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses")
        self.server._write_config_atomic(self.config, config)
        self.mock_codex()
        async def pin(provider, model, **kwargs):
            self.assertEqual(self.server._read_config_for_write(self.config)["model"], config["model"])
            await self.apply_model(provider, model)
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=pin)
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 200)

    async def test_rejected_codex_to_codex_change_cannot_look_like_a_successful_pin(self):
        self.seed("openai-codex", "previous-codex-model")
        self.mock_codex()
        self.server.set_active_model_via_hermes = AsyncMock(return_value="native pin rejected")
        before = self.snapshots()
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 503)
        self.assertEqual(self.snapshots(), before)

    async def test_disconnected_or_unlisted_model_does_not_touch_files(self):
        self.seed()
        self.mock_codex()
        before = self.snapshots()
        self.server._codex_models.return_value = []
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 409)
        self.server._codex_connected.return_value = False
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 409)
        self.assertEqual(self.snapshots(), before)

    async def test_leaving_codex_requires_successful_explicit_pin(self):
        self.seed("openai-codex", "synthetic-codex-model")
        before = self.snapshots()
        self.server.set_active_model_via_hermes = AsyncMock(return_value="not applied")
        result = await self.server.api_config_put(request({"vars": {"LLM_MODEL": "MiniMax-M2.7"}, "_active_provider_key": "MINIMAX_API_KEY"}))
        self.assertEqual(result.status_code, 503)
        self.assertEqual(self.snapshots(), before)

    async def test_failed_pin_cannot_rollback_a_concurrent_native_save(self):
        self.seed()
        self.mock_codex()
        async def concurrent_save(*args, **kwargs):
            self.config.write_text("model: {default: concurrently-saved, provider: nvidia}\ncustom: preserve-new-data\n")
            return "request failed"
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=concurrent_save)
        result = await self.server.api_config_put(self.codex_save())
        self.assertEqual(result.status_code, 503)
        self.assertIn("preserve-new-data", self.config.read_text())
        self.assertEqual(self.server.read_env(self.env)["MINIMAX_API_KEY"], "synthetic-minimax-key")

    async def test_timeout_after_native_commit_is_verified_by_readback(self):
        self.seed()
        self.mock_codex()
        async def committed(*args, **kwargs):
            await self.apply_model(*args, **kwargs)
            return "network timeout after publication"
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=committed)
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 200)
        self.assertEqual(self.server.configured_model()[1], "openai-codex")

    async def test_exception_and_request_cancellation_rollback_staged_files(self):
        self.seed()
        self.mock_codex()
        before = self.snapshots()
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=RuntimeError("interrupted"))
        self.assertEqual((await self.server.api_config_put(self.codex_save())).status_code, 500)
        self.assertEqual(self.snapshots(), before)
        self.server.set_active_model_via_hermes = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.server.api_config_put(self.codex_save())
        self.assertEqual(self.snapshots(), before)

    async def test_bridge_supports_gated_cookies_refresh_and_explicit_default_profile(self):
        self.server._get_hermes_session_token = AsyncMock(return_value="")
        self.server.hermes_session.snapshot = Mock(side_effect=[(0, {}), (1, {"HermesSession": "synthetic-session"})])
        self.server.hermes_session.refresh = AsyncMock(return_value=True)
        call = AsyncMock(side_effect=[httpx.Response(401), httpx.Response(200, json={"ok": True})])
        self.server.get_http_client = Mock(return_value=SimpleNamespace(request=call))
        self.assertEqual(await self.server._codex_api("POST", "/api/providers/oauth/openai-codex/start"), {"ok": True})
        self.assertEqual(call.await_args.kwargs["headers"], {"cookie": "HermesSession=synthetic-session"})
        self.assertEqual(call.await_args.kwargs["params"], {"profile": "default"})
        self.server.hermes_session.refresh.assert_awaited_once_with(0)

    async def test_bridge_ungated_token_and_error_redaction(self):
        self.server._get_hermes_session_token = AsyncMock(return_value="synthetic-session-token")
        call = AsyncMock(return_value=httpx.Response(500, json={"detail": "DO-NOT-EXPOSE-REFRESH-TOKEN"}))
        self.server.get_http_client = Mock(return_value=SimpleNamespace(request=call))
        with self.assertRaises(self.server.HermesAPIError) as error:
            await self.server._codex_api("POST", "/api/providers/oauth/openai-codex/start")
        self.assertNotIn("DO-NOT-EXPOSE", str(error.exception))
        self.assertEqual(call.await_args.kwargs["headers"][self.server._SESSION_TOKEN_HEADER], "synthetic-session-token")

    async def test_start_is_idempotent_and_status_resumes_pending_login(self):
        self.seed()
        self.server._codex_connected = Mock(return_value=False)
        self.server._codex_api = AsyncMock(side_effect=[{
            "session_id": "fixture-session-123", "user_code": "ABC-123", "verification_url": "https://auth.openai.com/codex/device",
            "expires_in": 900, "poll_interval": 5, "access_token": "NEVER-RETURN"}, {"status": "pending"}])
        first = await self.server.api_oauth_codex_start(request())
        second = await self.server.api_oauth_codex_start(request())
        self.assertEqual(first.status_code, 200)
        self.assertEqual(payload(first)["session_id"], payload(second)["session_id"])
        self.assertNotIn("NEVER-RETURN", first.body.decode())
        self.assertEqual(self.server._codex_api.await_args_list[1].args[0], "GET")

    async def test_cancel_failure_keeps_session_and_later_cancel_can_succeed(self):
        self.server._codex_session = {"session_id": "fixture-session-123"}
        self.server._codex_api = AsyncMock(side_effect=self.server.HermesAPIError("unavailable"))
        self.assertEqual((await self.server.api_oauth_codex_cancel(request())).status_code, 503)
        self.assertIsNotNone(self.server._codex_session)
        self.server._codex_api = AsyncMock(return_value={"ok": True})
        self.assertEqual((await self.server.api_oauth_codex_cancel(request())).status_code, 200)
        self.assertIsNone(self.server._codex_session)

    async def test_ended_session_does_not_report_waiting_forever(self):
        self.seed()
        self.server._codex_connected = Mock(return_value=False)
        self.server._codex_session = {"session_id": "fixture-session-123"}
        self.server._codex_api = AsyncMock(side_effect=self.server.HermesAPIError("expired", 404))
        response = await self.server.api_oauth_codex_status(request())
        self.assertEqual(payload(response)["status"], "expired")
        self.assertIsNone(self.server._codex_session)

    async def test_approval_uses_real_status_and_drops_native_error_details(self):
        self.seed()
        self.server._codex_connected = Mock(return_value=True)
        self.server._codex_session = {"session_id": "fixture-session-123"}
        self.server._codex_api = AsyncMock(return_value={"status": "approved", "error_message": "SECRET"})
        response = await self.server.api_oauth_codex_status(request())
        self.assertEqual(payload(response)["status"], "connected")
        self.assertNotIn("SECRET", response.body.decode())

    async def test_invalid_verification_host_is_rejected(self):
        self.seed()
        self.server._codex_api = AsyncMock(return_value={"session_id": "fixture-session-123", "user_code": "ABC-123", "verification_url": "https://attacker.invalid/"})
        self.assertEqual((await self.server.api_oauth_codex_start(request())).status_code, 502)
        self.assertIsNone(self.server._codex_session)

    async def test_invalid_expiry_cannot_expose_upstream_token_text(self):
        self.seed()
        self.server._codex_api = AsyncMock(return_value={"session_id": "fixture-session-123", "user_code": "ABC-123", "verification_url": "https://auth.openai.com/codex/device", "expires_in": "SECRET-REFRESH-TOKEN"})
        response = await self.server.api_oauth_codex_start(request())
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("SECRET", response.body.decode())

    async def test_model_projection_excludes_other_providers_secrets_and_unavailable_ids(self):
        self.server._codex_api = AsyncMock(return_value={"providers": [
            {"slug": "other", "models": ["unrelated"]},
            {"slug": "openai-codex", "models": ["available", "unavailable", "available", {"token": "secret"}], "unavailable_models": ["unavailable"], "token": "secret"}]})
        self.assertEqual(await self.server._codex_models(), ["available"])

    async def test_disconnect_stops_holders_before_native_clear_and_keeps_other_data(self):
        self.seed("openai-codex", "synthetic-codex-model")
        self.auth.write_text(json.dumps({"version": 2, "providers": {"openai-codex": {"tokens": {"access_token": token(), "refresh_token": "synthetic"}}, "xai-oauth": {"tokens": {"refresh_token": "keep-xai"}}}, "credential_pool": {"openai-codex": []}}))
        named = self.home / "profiles" / "fixture"
        named.mkdir(parents=True)
        (named / "auth.json").write_text('{"unrelated":"preserve-profile"}')
        before = (self.env.read_bytes(), self.config.read_bytes(), (named / "auth.json").read_bytes())
        calls = []
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server.dash.stop = AsyncMock(side_effect=lambda: calls.append("stop-dashboard") or True)
        self.server._stop_gateway_for_maintenance = AsyncMock(side_effect=lambda **_: calls.append("stop-gateway"))
        self.server.dash.start = AsyncMock(return_value=True)
        self.server.gw.start = AsyncMock()
        from hermes_cli.auth import clear_provider_auth
        def clear(provider):
            self.assertEqual(calls, ["stop-dashboard", "stop-gateway"])
            return clear_provider_auth(provider)
        with patch("hermes_cli.auth.clear_provider_auth", side_effect=clear):
            response = await self.server.api_oauth_codex_delete(request())
        self.assertEqual(response.status_code, 200)
        state = json.loads(self.auth.read_text())
        self.assertNotIn("openai-codex", state["providers"])
        self.assertNotIn("openai-codex", state["credential_pool"])
        self.assertEqual(state["providers"]["xai-oauth"]["tokens"]["refresh_token"], "keep-xai")
        self.assertEqual(before, (self.env.read_bytes(), self.config.read_bytes(), (named / "auth.json").read_bytes()))
        self.server.gw.start.assert_not_awaited()

    async def test_disconnect_refuses_uncertain_stop_and_does_not_delete_auth(self):
        self.seed()
        self.auth.write_text('{"version":2,"providers":{"openai-codex":{}}}')
        before = self.snapshots()
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server.dash.stop = AsyncMock(return_value=True)
        self.server.dash.start = AsyncMock(return_value=True)
        self.server._stop_gateway_for_maintenance = AsyncMock(side_effect=RuntimeError("still draining"))
        with patch("hermes_cli.auth.clear_provider_auth") as clear:
            self.assertEqual((await self.server.api_oauth_codex_delete(request())).status_code, 503)
            clear.assert_not_called()
        self.assertEqual(before, self.snapshots())

    async def test_reset_does_not_change_config_if_cancellation_is_unconfirmed(self):
        self.seed()
        before = self.snapshots()
        self.server._cancel_codex_session = AsyncMock(side_effect=self.server.HermesAPIError("unconfirmed"))
        self.assertEqual((await self.server.api_config_reset(request())).status_code, 500)
        self.assertEqual(before, self.snapshots())

    async def test_all_new_routes_require_admin_auth_and_mutations_wait_for_preparation(self):
        self.server.guard = self.real_guard
        handlers = (self.server.api_oauth_codex_start, self.server.api_oauth_codex_cancel,
                    self.server.api_oauth_codex_delete, self.server.api_oauth_codex_models, self.server.api_oauth_codex_status)
        req = Request({"type": "http", "method": "POST", "path": "/setup/api/oauth/codex/start", "headers": [], "query_string": b""})
        for handler in handlers:
            self.assertEqual((await handler(req)).status_code, 401)
        self.server._is_authenticated = Mock(return_value=True)
        self.server.preparation["state"] = "running"
        for handler in handlers:
            self.assertEqual((await handler(req)).status_code, 503)

    async def test_native_settings_writes_cannot_interleave_a_setup_transaction(self):
        self.server._is_authenticated = Mock(return_value=True)
        def proxy(_):
            self.assertTrue(self.server.lifecycle_lock.locked())
            return self.server.JSONResponse({"ok": True})
        self.server._proxy_to_dashboard = AsyncMock(side_effect=proxy)
        for path, method in (("/api/config", "PUT"), ("/api/env", "DELETE"), ("/api/model/set", "POST")):
            req = Request({"type": "http", "method": method, "path": path, "headers": [], "query_string": b""})
            async with self.server.lifecycle_lock:
                task = asyncio.create_task(self.server.route_proxy(req))
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            self.assertEqual((await task).status_code, 200)


if __name__ == "__main__":
    unittest.main()
