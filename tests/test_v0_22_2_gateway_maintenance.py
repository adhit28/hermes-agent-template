"""Maintenance must account for native detached gateways and pending actions."""
from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import zipfile

import httpx
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

from test_v2026_9_24_config_safety import ConfigSafetyFixture


class GatewayMaintenanceTests(ConfigSafetyFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.real_guard = self.server.guard
        self.server.guard = lambda _: None

    def test_verified_identity_requires_strict_and_home_scoped_agreement(self):
        with patch("gateway.status.get_running_pid_identity_strict", return_value=(77, 123)), \
                patch("gateway.status.get_running_pid", return_value=77) as scoped:
            self.assertEqual(self.server._verified_root_gateway_pid(), 77)
            scoped.assert_called_once_with(self.home / "gateway.pid", cleanup_stale=False)
        with patch("gateway.status.get_running_pid_identity_strict", return_value=(77, 123)), \
                patch("gateway.status.get_running_pid", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "home could not be verified"):
                self.server._verified_root_gateway_pid()
        with patch("gateway.status.get_running_pid_identity_strict", side_effect=RuntimeError("ambiguous held lock")):
            with self.assertRaisesRegex(RuntimeError, "ambiguous held lock"):
                self.server._verified_root_gateway_pid()

    async def test_native_action_status_authenticates_and_distinguishes_gateway_from_restarter(self):
        self.server.dash.proc = SimpleNamespace(returncode=None)
        self.server._get_hermes_session_token = AsyncMock(return_value="synthetic-token")

        async def get(url, **kwargs):
            name = url.split("/actions/", 1)[1].split("/", 1)[0]
            self.assertEqual(kwargs["headers"][self.server._SESSION_TOKEN_HEADER], "synthetic-token")
            return httpx.Response(200, json={"name": name, "running": name == "gateway-restart", "pid": 77})

        self.server.get_http_client = Mock(return_value=SimpleNamespace(get=AsyncMock(side_effect=get)))
        self.assertEqual(await self.server._native_gateway_action_busy(99), "gateway-restart")
        self.assertIsNone(await self.server._native_gateway_action_busy(77))

    async def test_action_admission_rejects_ambiguous_identity_or_bad_status(self):
        self.server._verified_root_gateway_pid = Mock(side_effect=RuntimeError("ambiguous held lock"))
        _, response = await self.server._gateway_maintenance_check()
        self.assertEqual(response.status_code, 503)
        self.server._verified_root_gateway_pid = Mock(return_value=None)
        self.server.dash.proc = SimpleNamespace(returncode=None)
        self.server._get_hermes_session_token = AsyncMock(return_value="")
        client = SimpleNamespace(get=AsyncMock(return_value=httpx.Response(200, json={"name": "gateway-start", "running": "false"})))
        self.server.get_http_client = Mock(return_value=client)
        _, response = await self.server._gateway_maintenance_check()
        self.assertEqual(response.status_code, 503)
        self.server.dash.proc = SimpleNamespace(returncode=1)
        _, response = await self.server._gateway_maintenance_check()
        self.assertEqual(response.status_code, 503)

    async def test_native_action_status_refreshes_expired_dashboard_session(self):
        self.server.dash.proc = SimpleNamespace(returncode=None)
        self.server._get_hermes_session_token = AsyncMock(return_value="")
        self.server.hermes_session.snapshot = Mock(side_effect=[(0, {}), (1, {"HermesSession": "synthetic"}),
                                                              (1, {}), (1, {}), (1, {})])
        self.server.hermes_session.refresh = AsyncMock(return_value=True)
        calls = []

        async def get(url, **kwargs):
            name = url.split("/actions/", 1)[1].split("/", 1)[0]
            calls.append(kwargs["headers"])
            if len(calls) == 1:
                return httpx.Response(401)
            return httpx.Response(200, json={"name": name, "running": False, "pid": None})

        self.server.get_http_client = Mock(return_value=SimpleNamespace(get=AsyncMock(side_effect=get)))
        self.assertIsNone(await self.server._native_gateway_action_busy(None))
        self.server.hermes_session.refresh.assert_awaited_once_with(0)
        self.assertEqual(calls[1]["cookie"], "HermesSession=synthetic")

    async def test_native_stop_exit_zero_still_waits_for_verified_absence(self):
        self.server.gw.stop = AsyncMock()
        self.server._verified_root_gateway_pid = Mock(side_effect=[77, 77, None])
        self.server._run_hermes_cli = AsyncMock(return_value=(0, "stopped"))
        with patch.object(self.server.asyncio, "sleep", new=AsyncMock()) as sleep:
            await self.server._stop_gateway_for_maintenance()
        self.server.gw.stop.assert_awaited_once()
        args, kwargs = self.server._run_hermes_cli.await_args
        self.assertEqual(args, ("gateway", "stop"))
        self.assertLessEqual(kwargs["timeout"], 70)
        self.assertEqual(self.server._verified_root_gateway_pid.call_count, 3)
        sleep.assert_awaited_once()

    async def test_native_stop_timeout_or_identity_error_cannot_claim_safe_absence(self):
        self.server.gw.stop = AsyncMock()
        self.server._run_hermes_cli = AsyncMock(return_value=(0, "stopped"))
        self.server._verified_root_gateway_pid = Mock(return_value=77)
        with patch.object(self.server, "GATEWAY_STOP_TIMEOUT", 0.01):
            with self.assertRaisesRegex(RuntimeError, "still draining"):
                await self.server._stop_gateway_for_maintenance()
        self.server._verified_root_gateway_pid = Mock(side_effect=RuntimeError("lock identity changed"))
        self.server._run_hermes_cli.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "lock identity changed"):
            await self.server._stop_gateway_for_maintenance()
        self.server._run_hermes_cli.assert_not_awaited()

    async def test_apply_reloads_detached_gateway_without_writing_config_or_estop(self):
        config = self.home / "config.yaml"
        config.write_text("model: {default: synthetic}\n")
        self.server.ENV_FILE.write_text("MINIMAX_API_KEY=synthetic\n")
        self.server.ESTOP_FILE.write_text("paused")
        before = {path: path.read_bytes() for path in (config, self.server.ENV_FILE, self.server.ESTOP_FILE)}
        events = []
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server.dash.stop = AsyncMock(side_effect=lambda: events.append("dashboard-stop"))
        self.server._stop_gateway_for_maintenance = AsyncMock(side_effect=lambda **_: events.append("gateway-stop"))
        self.server.dash.start = AsyncMock(side_effect=lambda: events.append("dashboard-start"))
        self.server.gw.start = AsyncMock(side_effect=lambda **_: events.append("gateway-start"))
        with patch.object(self.server, "is_config_complete", return_value=True):
            response = await self.server.api_dependencies_apply(object())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events, ["dashboard-stop", "gateway-stop", "dashboard-start", "gateway-start"])
        self.server.gw.start.assert_awaited_once_with(write_config=False)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    async def test_busy_native_restart_refuses_apply_before_stopping_children(self):
        self.server._verified_root_gateway_pid = Mock(return_value=77)
        self.server._native_gateway_action_busy = AsyncMock(return_value="gateway-restart")
        self.server.dash.stop = AsyncMock()
        self.server.gw.stop = AsyncMock()
        response = await self.server.api_dependencies_apply(object())
        self.assertEqual(response.status_code, 409)
        self.server.dash.stop.assert_not_awaited()
        self.server.gw.stop.assert_not_awaited()

    async def test_native_gateway_proxy_rechecks_guard_after_waiting_for_maintenance(self):
        request = Request({"type": "http", "method": "POST", "path": "/api/gateway/restart", "headers": []})
        self.server.guard = self.real_guard
        self.server._proxy_to_dashboard = AsyncMock(return_value=JSONResponse({"ok": True}))
        with patch.object(self.server, "_is_authenticated", return_value=True):
            async with self.server.lifecycle_lock:
                action = asyncio.create_task(self.server.route_proxy(request))
                await asyncio.sleep(0)
                self.server._proxy_to_dashboard.assert_not_awaited()
                self.server.preparation["state"] = "running"
            response = await action
        self.assertEqual(response.status_code, 503)
        self.server._proxy_to_dashboard.assert_not_awaited()

    async def test_native_onboarding_and_webhook_restart_producers_share_admission(self):
        self.server.guard = self.real_guard
        self.server._proxy_to_dashboard = AsyncMock(return_value=JSONResponse({"ok": True}))
        paths = (
            "/api/webhooks/enable",
            "/api/messaging/telegram/onboarding/synthetic-pairing/apply",
            "/api/messaging/whatsapp/onboarding/synthetic-pairing/apply",
        )
        with patch.object(self.server, "_is_authenticated", return_value=True):
            for path in paths:
                with self.subTest(path=path):
                    self.server.preparation["state"] = "ready"
                    request = Request({"type": "http", "method": "POST", "path": path, "headers": []})
                    async with self.server.lifecycle_lock:
                        action = asyncio.create_task(self.server.route_proxy(request))
                        await asyncio.sleep(0)
                        self.server._proxy_to_dashboard.assert_not_awaited()
                        self.server.preparation["state"] = "running"
                    response = await action
                    self.assertEqual(response.status_code, 503)
                    self.server._proxy_to_dashboard.assert_not_awaited()

    async def test_unrelated_native_api_is_not_serialized_as_gateway_lifecycle(self):
        # model/set now participates in setup's settings transaction; a
        # read-only credential validation still bypasses that lifecycle lock.
        cookie = f"hermes_auth={self.server._make_auth_token()}".encode()
        request = Request({"type": "http", "method": "POST", "path": "/api/providers/validate", "headers": [(b"cookie", cookie)]})
        self.server._proxy_to_dashboard = AsyncMock(return_value=JSONResponse({"ok": True}))
        async with self.server.lifecycle_lock:
            response = await self.server.route_proxy(request)
        self.assertEqual(response.status_code, 200)
        self.server._proxy_to_dashboard.assert_awaited_once()

    def restore_request(self):
        incoming = io.BytesIO()
        with zipfile.ZipFile(incoming, "w") as archive:
            archive.writestr("config.yaml", "_config_version: 46\n")
        upload = UploadFile(filename="synthetic.zip", file=io.BytesIO(incoming.getvalue()))
        return SimpleNamespace(form=AsyncMock(return_value={"file": upload}))

    async def test_restore_waits_for_detached_gateway_before_import(self):
        events = []

        async def cli(*args, **kwargs):
            if args[0] == "backup":
                events.append("backup")
                with zipfile.ZipFile(args[2], "w") as archive:
                    archive.writestr("config.yaml", "_config_version: 50\n")
            elif args[:2] == ("gateway", "stop"):
                events.append("native-stop")
            elif args[0] == "import":
                self.assertEqual(events[-1], "gateway-absent")
                events.append("import")
            return 0, "ok"

        def pid():
            if events.count("native-stop"):
                events.append("gateway-absent")
                return None
            return 77

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server._verified_root_gateway_pid = Mock(side_effect=pid)
        self.server.dash.stop = AsyncMock(side_effect=lambda: events.append("dashboard-stop"))
        self.server.gw.stop = AsyncMock(side_effect=lambda: events.append("owned-stop"))
        self.server._prepare_hermes = AsyncMock(return_value=False)
        response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 200)
        await self.server._preparation_task
        self.assertEqual(events, ["backup", "dashboard-stop", "owned-stop", "native-stop", "gateway-absent", "import"])

    async def test_restore_stop_failure_keeps_safety_copy_and_never_imports(self):
        commands = []

        async def cli(*args, **kwargs):
            commands.append(args[0])
            if args[0] == "backup":
                with zipfile.ZipFile(args[2], "w") as archive:
                    archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "ok"

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server.dash.stop = AsyncMock()
        self.server.dash.start = AsyncMock()
        self.server._stop_gateway_for_maintenance = AsyncMock(side_effect=RuntimeError("still draining"))
        response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(commands, ["backup"])
        self.assertEqual(len(list(self.server.BACKUP_DIR.glob("pre-restore-*.zip"))), 1)
        self.assertIsNone(self.server._restore_state())
        self.assertEqual(self.server.preparation["state"], "ready")
        self.server.dash.start.assert_awaited_once()

    async def test_apply_restarts_native_configured_gateway_without_template_model_env(self):
        (self.home / "config.yaml").write_text(
            "_config_version: 50\nmodel:\n  provider: custom\n  default: synthetic-model\n"
            "  base_url: https://synthetic.example.invalid/v1\n  api_key: synthetic-key\n"
        )
        self.server.ENV_FILE.write_text("")
        self.assertFalse(self.server.is_config_complete())
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server.dash.stop = AsyncMock(return_value=True)
        self.server.dash.start = AsyncMock(return_value=True)
        self.server._stop_gateway_for_maintenance = AsyncMock()
        self.server.gw.start = AsyncMock(return_value=True)
        response = await self.server.api_dependencies_apply(object())
        self.assertEqual(response.status_code, 200)
        self.server.gw.start.assert_awaited_once_with(write_config=False)

    async def test_dashboard_restart_keeps_native_action_registry_when_busy(self):
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, JSONResponse({"error": "pending restart"}, status_code=409)))
        self.server.dash.stop = AsyncMock()
        self.server.dash.start = AsyncMock()
        self.assertFalse(await self.server.dash.restart())
        self.server.dash.stop.assert_not_awaited()
        self.server.dash.start.assert_not_awaited()

    async def test_restore_after_controlled_dashboard_stop_can_recover_in_same_process(self):
        class Process:
            returncode = None

            def terminate(self):
                self.returncode = 0

            async def wait(self):
                return self.returncode

        self.server.dash.proc = Process()
        self.server._verified_root_gateway_pid = Mock(return_value=None)
        self.server._get_hermes_session_token = AsyncMock(return_value="synthetic-token")

        async def get(url, **kwargs):
            name = url.split("/actions/", 1)[1].split("/", 1)[0]
            return httpx.Response(200, json={"name": name, "running": False, "pid": None})

        client = SimpleNamespace(get=AsyncMock(side_effect=get))
        self.server.get_http_client = Mock(return_value=client)

        async def cli(*args, **kwargs):
            if args[0] == "backup":
                with zipfile.ZipFile(args[2], "w") as archive:
                    archive.writestr("config.yaml", "_config_version: 50\n")
                return 0, "synthetic safety snapshot"
            return 1, "synthetic failed import"

        self.server._run_hermes_cli = cli
        first = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(first.status_code, 500)
        self.assertTrue(self.server.dash._maintenance_stopped)
        original = self.server._restore_state()["snapshot"]
        second = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(second.status_code, 500)  # Import ran and failed, not a preflight503.
        self.assertEqual(self.server._restore_state()["snapshot"], original)
        self.assertTrue(self.server.preparation["requires_restore"])
        self.assertIsNone(self.server._preparation_task)
        self.assertEqual(client.get.await_count, 4)  # No lost registry is queried during recovery.

    async def test_controlled_dashboard_stop_is_not_authority_for_a_new_gateway(self):
        self.server.dash.proc = SimpleNamespace(returncode=0)
        self.server.dash._maintenance_stopped = True
        self.assertIsNone(await self.server._native_gateway_action_busy(None))
        with self.assertRaisesRegex(RuntimeError, "Dashboard exited"):
            await self.server._native_gateway_action_busy(77)
        self.server.dash._maintenance_stopped = False
        with self.assertRaisesRegex(RuntimeError, "Dashboard exited"):
            await self.server._native_gateway_action_busy(None)

    async def test_new_dashboard_spawn_clears_maintenance_authority(self):
        self.server.dash._maintenance_stopped = True
        stream = asyncio.StreamReader()
        stream.feed_eof()
        process = SimpleNamespace(pid=77, returncode=None, stdout=stream)
        with patch.object(self.server.asyncio, "create_subprocess_exec", new=AsyncMock(return_value=process)):
            self.assertTrue(await self.server.dash.start())
        self.assertFalse(self.server.dash._maintenance_stopped)
        await self.server.dash._drain_task

    async def test_restore_marker_write_failure_does_not_stop_live_children(self):
        commands = []

        async def cli(*args, **kwargs):
            commands.append(args[0])
            with zipfile.ZipFile(args[2], "w") as archive:
                archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "synthetic safety snapshot"

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(77, None))
        self.server.dash.stop = AsyncMock(wraps=self.server.dash.stop)
        self.server.gw.stop = AsyncMock(wraps=self.server.gw.stop)
        with patch.object(self.server, "_write_restore_state", side_effect=OSError("synthetic marker failure")):
            response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(commands, ["backup"])
        self.server.dash.stop.assert_not_awaited()
        self.server.gw.stop.assert_not_awaited()
        self.assertEqual(self.server.preparation["state"], "failed")
        self.assertTrue(self.server.preparation["requires_restore"])

    async def test_restore_publication_directory_sync_failure_restores_prior_authority(self):
        commands = []

        async def cli(*args, **kwargs):
            commands.append(args[0])
            with zipfile.ZipFile(args[2], "w") as archive:
                archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "synthetic safety snapshot"

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(None, None))
        self.server.dash.stop = AsyncMock()
        self.server.gw.stop = AsyncMock()
        fsync = os.fsync
        for prior in (None, {"phase": "importing", "snapshot": "pre-restore-100-aaaaaaaa.zip"}):
            self.server.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            self.server._write_restore_state(prior)
            self.server.preparation.update(state="failed" if prior else "ready", phase="failed" if prior else "complete",
                                           requires_restore=bool(prior))
            before = dict(self.server.preparation)
            directory_syncs = 0

            def fail_first_directory_sync(fd):
                nonlocal directory_syncs
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    directory_syncs += 1
                    if directory_syncs == 1:
                        # Actual _atomic_write_text has already renamed it.
                        self.assertEqual(self.server._restore_state()["phase"], "importing")
                        raise OSError("synthetic post-rename directory sync failure")
                return fsync(fd)

            with patch.object(self.server.os, "fsync", side_effect=fail_first_directory_sync):
                response = await self.server.api_backup_restore(self.restore_request())
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.server._restore_state(), prior)
            self.assertEqual(self.server.preparation, before)
            self.assertEqual(directory_syncs, 2)
        self.assertEqual(commands, ["backup", "backup"])
        self.server.dash.stop.assert_not_awaited()
        self.server.gw.stop.assert_not_awaited()

    async def test_restore_publication_and_rollback_sync_failure_keep_import_gate(self):
        commands = []

        async def cli(*args, **kwargs):
            commands.append(args[0])
            with zipfile.ZipFile(args[2], "w") as archive:
                archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "synthetic safety snapshot"

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(None, None))
        self.server.dash.stop = AsyncMock()
        self.server.gw.stop = AsyncMock()
        fsync = os.fsync

        def fail_directory_sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("synthetic persistent directory sync failure")
            return fsync(fd)

        with patch.object(self.server.os, "fsync", side_effect=fail_directory_sync):
            response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(commands, ["backup"])
        self.assertEqual(self.server._restore_state()["phase"], "importing")
        self.assertEqual(self.server.preparation["state"], "failed")
        self.assertTrue(self.server.preparation["requires_restore"])
        self.server.dash.stop.assert_not_awaited()
        self.server.gw.stop.assert_not_awaited()

    async def test_recovery_stop_failure_restores_prior_marker_without_starting_partial_state(self):
        self.server.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        original = {"phase": "importing", "snapshot": "pre-restore-100-aaaaaaaa.zip"}
        self.server._write_restore_state(original)
        self.server.preparation.update(state="failed", phase="failed", requires_restore=True)
        before = dict(self.server.preparation)

        async def cli(*args, **kwargs):
            self.assertEqual(args[0], "backup")
            with zipfile.ZipFile(args[2], "w") as archive:
                archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "synthetic partial-state safety snapshot"

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(None, None))
        self.server._stop_gateway_for_maintenance = AsyncMock(side_effect=RuntimeError("ambiguous identity"))
        self.server.dash.start = AsyncMock()
        response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.server._restore_state(), original)
        self.assertEqual(self.server.preparation, before)
        self.server.dash.start.assert_not_awaited()

    async def test_cancelled_dashboard_stop_keeps_durable_import_gate_without_importing(self):
        stopped = asyncio.Event()
        release = asyncio.Event()

        class Process:
            returncode = None

            def terminate(self):
                self.returncode = 0

            async def wait(self):
                stopped.set()
                await release.wait()
                return self.returncode

        self.server.dash.proc = Process()
        self.server._gateway_maintenance_check = AsyncMock(return_value=(None, None))
        commands = []

        async def cli(*args, **kwargs):
            commands.append(args[0])
            with zipfile.ZipFile(args[2], "w") as archive:
                archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "synthetic safety snapshot"

        self.server._run_hermes_cli = cli
        action = asyncio.create_task(self.server.api_backup_restore(self.restore_request()))
        await stopped.wait()
        action.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await action
        self.assertEqual(commands, ["backup"])
        self.assertEqual(self.server._restore_state()["phase"], "importing")
        self.assertEqual(self.server.preparation["state"], "failed")
        self.assertTrue(self.server.preparation["requires_restore"])

    async def test_stop_failure_and_record_rollback_failure_keep_activation_blocked(self):
        commands = []

        async def cli(*args, **kwargs):
            commands.append(args[0])
            with zipfile.ZipFile(args[2], "w") as archive:
                archive.writestr("config.yaml", "_config_version: 50\n")
            return 0, "synthetic safety snapshot"

        write_record = self.server._write_restore_state

        def fail_rollback(state):
            if state is None:
                raise OSError("synthetic rollback sync failure")
            write_record(state)

        self.server._run_hermes_cli = cli
        self.server._gateway_maintenance_check = AsyncMock(return_value=(None, None))
        self.server._stop_gateway_for_maintenance = AsyncMock(side_effect=RuntimeError("still draining"))
        self.server.dash.start = AsyncMock()
        with patch.object(self.server, "_write_restore_state", side_effect=fail_rollback):
            response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(commands, ["backup"])
        self.assertEqual(self.server._restore_state()["phase"], "importing")
        self.assertTrue(self.server.preparation["requires_restore"])
        self.assertEqual(self.server.preparation["state"], "failed")
        self.server.dash.start.assert_not_awaited()

    async def test_clock_rollback_keeps_original_anchor_and_new_verified_snapshot(self):
        self.server.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        original = self.server.BACKUP_DIR / "pre-restore-100-aaaaaaaa.zip"
        with zipfile.ZipFile(original, "w") as archive:
            archive.writestr("config.yaml", "_config_version: 50\noriginal: synthetic\n")
        original_bytes = original.read_bytes()
        os.utime(original, (100, 100))
        self.server._write_restore_state({"phase": "importing", "snapshot": original.name})
        self.server.preparation.update(state="failed", requires_restore=True)
        future = time.time() + 86400
        for i in range(self.server.PRE_RESTORE_KEEP):
            old = self.server.BACKUP_DIR / f"pre-restore-{200+i}-bbbbbbbb.zip"
            old.write_bytes(b"synthetic future-dated snapshot")
            os.utime(old, (future+i, future+i))
        self.server._gateway_maintenance_check = AsyncMock(return_value=(None, None))
        self.server._verified_root_gateway_pid = Mock(return_value=None)
        created = []

        async def cli(*args, **kwargs):
            if args[0] == "backup":
                created.append(Path(args[2]))
                with zipfile.ZipFile(args[2], "w") as archive:
                    archive.writestr("config.yaml", "_config_version: 50\n")
                return 0, "synthetic current-state snapshot"
            self.assertTrue(created[0].is_file())
            self.assertEqual(original.read_bytes(), original_bytes)
            return 1, "synthetic recovery import failure"

        self.server._run_hermes_cli = cli
        response = await self.server.api_backup_restore(self.restore_request())
        self.assertEqual(response.status_code, 500)
        self.assertTrue(created[0].is_file())
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(self.server._restore_state()["snapshot"], original.name)
