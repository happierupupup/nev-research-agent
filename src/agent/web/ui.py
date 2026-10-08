"""Gradio UI：登录 → 对话 / 用户 / 爬虫（管理员三页；普通用户仅对话）。"""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from agent.auth import User, is_admin
from agent.display import compose_streaming_bubble, format_assistant_message
from agent.web.crawl_admin import (
    approve_and_promote,
    list_staging,
    run_config_crawl,
    run_manual_crawl,
)
from agent.web.session_store import STORE
from agent.web.stream_research import history_for_gradio, iter_research_events
from agent.web.tokens import issue_token, resolve_token, revoke_token
from agent.web.users_admin import (
    delete_user,
    get_user_public,
    list_users_public,
    upsert_user,
    verify_login,
)

CSS = """
/* 宽度适应浏览器；不做会裁掉输入框的 100vh 套娃 */
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  padding: 10px 16px 16px !important;
  box-sizing: border-box !important;
}
.nev-login {max-width: 400px; margin: 3.5rem auto; padding: 1.5rem;}
.nev-header {margin-bottom: 6px;}

/* 左侧会话 */
.nev-side {
  background: #f7f8fa;
  border-right: 1px solid #e8eaed;
  padding: 12px 10px !important;
  max-height: 75vh;
  overflow-y: auto !important;
  border-radius: 12px 0 0 12px;
}
.nev-side label,
.nev-side .wrap label,
.nev-side span {
  background: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 8px 10px !important;
  font-size: 0.9rem !important;
  color: #1f2937 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.nev-side label:hover {background: #eef0f3 !important; color: #111827 !important;}
.nev-side input:checked + label,
.nev-side label.selected,
.nev-side .selected,
.nev-side [aria-checked="true"],
.nev-side label:has(input:checked),
.nev-side .primary,
.nev-side .primary *,
.nev-side label.primary,
.nev-side label.primary * {
  background: #dbeafe !important;
  color: #1e3a8a !important;
  font-weight: 600 !important;
  border-color: #93c5fd !important;
}

.nev-chat-col {padding: 4px 8px 0 !important;}

/* 聊天区随视口，但给输入框留出空间 */
#nev-chatbot {
  border: none !important;
  box-shadow: none !important;
  height: 58vh !important;
  min-height: 320px !important;
  max-height: 70vh !important;
}
#nev-chatbot .wrap,
#nev-chatbot > .wrapper {
  height: 58vh !important;
  max-height: 70vh !important;
  overflow-y: auto !important;
}
#nev-chatbot .message {border-radius: 14px !important; padding: 10px 14px !important;}
#nev-chatbot .bot {background: #f3f4f6 !important;}
#nev-chatbot .user {background: #e8f0fe !important;}

/* 输入区始终可见，不被聊天区盖住 */
.nev-input-row {
  margin-top: 10px !important;
  margin-bottom: 8px !important;
  position: relative !important;
  z-index: 5 !important;
}

footer {display: none !important;}
"""

# 发送过程中忽略会话 Radio 的 change，避免冲掉流式气泡
_IGNORE_SESSION_CHANGE = {"on": False}


def _user_from_state(token: str | None) -> User | None:
    return resolve_token(token)


def _session_choices(user_id: str) -> tuple[list[tuple[str, str]], str | None]:
    rows = STORE.list(user_id)
    choices = [(f"{r['title']}", r["id"]) for r in rows]
    value = choices[0][1] if choices else None
    return choices, value


def _users_table() -> list[list[str]]:
    """管理页只展示 yaml 中的用户。"""
    return [
        [
            u["id"],
            u["name"],
            u["role"],
            u["note"],
            "已设置" if u["has_password"] else "默认=id",
        ]
        for u in list_users_public(include_builtin=False)
    ]


def _user_pick_update(selected: str | None = None):
    choices = [
        (f"{u['id']} · {u['name']}（{u['role']}）", u["id"])
        for u in list_users_public(include_builtin=False)
    ]
    value = selected if selected and any(c[1] == selected for c in choices) else (
        choices[0][1] if choices else None
    )
    return gr.update(choices=choices, value=value)


def _edit_form_for(uid: str | None):
    row = get_user_public(uid) if uid else None
    if not row:
        return "", "", "user", "", ""
    return row["id"], row["name"], row["role"], row["note"], ""


def _staging_table() -> list[list[Any]]:
    return [
        [d["id"], d["status"], d["title"][:40], d["query"][:30], d["url"][:48], d["chars"]]
        for d in list_staging()
    ]


def build_ui() -> gr.Blocks:
    try:
        _theme = gr.themes.Soft()
    except Exception:  # noqa: BLE001
        _theme = None
    blocks_kwargs: dict[str, Any] = {"title": "NEV Research Agent", "css": CSS}
    if _theme is not None:
        blocks_kwargs["theme"] = _theme

    with gr.Blocks(**blocks_kwargs) as demo:
        token_state = gr.State("")
        user_id_state = gr.State("")
        role_state = gr.State("")
        session_id_state = gr.State("")

        # ---------- Login ----------
        with gr.Column(visible=True, elem_classes=["nev-login"]) as login_view:
            gr.Markdown("# NEV Research Agent\n新能源汽车行业研究助手")
            login_user = gr.Textbox(label="用户 ID", placeholder="admin / alice / dev")
            login_pass = gr.Textbox(label="密码", type="password", placeholder="默认与用户 ID 相同")
            login_btn = gr.Button("登录", variant="primary")
            login_msg = gr.Markdown("")
            gr.Markdown("测试账号见 `config/users.yaml`（如 `alice/alice`、`admin/admin`）。")

        # ---------- Main ----------
        with gr.Column(visible=False) as main_view:
            with gr.Row(elem_classes=["nev-header"]):
                header = gr.Markdown("### ")
                logout_btn = gr.Button("退出", scale=0)

            with gr.Tabs() as main_tabs:
                # ===== 对话页 =====
                with gr.Tab("对话"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=220, elem_classes=["nev-side"]):
                            new_btn = gr.Button("＋ 新对话", variant="primary")
                            session_radio = gr.Radio(
                                label="会话",
                                choices=[],
                                interactive=True,
                            )
                            with gr.Row():
                                refresh_btn = gr.Button("刷新")
                                del_btn = gr.Button("删除", variant="secondary")

                        with gr.Column(scale=4, elem_classes=["nev-chat-col"]):
                            chatbot = gr.Chatbot(
                                label="对话",
                                height=480,
                                elem_id="nev-chatbot",
                            )
                            # 输入框单独在 chatbot 下方，避免被视口高度 CSS 裁掉
                            with gr.Row(elem_classes=["nev-input-row"]):
                                msg = gr.Textbox(
                                    label="输入",
                                    show_label=False,
                                    placeholder="发消息…（新能源汽车相关）",
                                    lines=2,
                                    scale=8,
                                )
                                send_btn = gr.Button("发送", variant="primary", scale=1)

                # ===== 用户页（仅管理员可见）=====
                with gr.Tab("用户", visible=False) as tab_users:
                    gr.Markdown("### 用户列表\n数据来自 `config/users.yaml`。选中一条后可修改或删除。")
                    users_df = gr.Dataframe(
                        headers=["用户ID", "显示名", "角色", "备注", "密码"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                    )
                    u_refresh = gr.Button("刷新列表")

                    gr.Markdown("### 编辑 / 删除")
                    user_pick = gr.Dropdown(
                        label="选择用户",
                        choices=[],
                        interactive=True,
                    )
                    with gr.Row():
                        e_id = gr.Textbox(label="用户 ID", interactive=False)
                        e_name = gr.Textbox(label="显示名")
                        e_role = gr.Dropdown(
                            choices=["user", "admin"], value="user", label="角色"
                        )
                    e_note = gr.Textbox(label="备注")
                    e_pass = gr.Textbox(
                        label="新密码（留空则不改）",
                        type="password",
                        placeholder="留空保持原密码",
                    )
                    with gr.Row():
                        e_save = gr.Button("保存修改", variant="primary")
                        e_del = gr.Button("删除该用户", variant="secondary")
                    u_msg = gr.Markdown("")

                    gr.Markdown("### 新增用户")
                    with gr.Row():
                        n_id = gr.Textbox(label="用户 ID", placeholder="必填，如 bob")
                        n_name = gr.Textbox(label="显示名", placeholder="可选，默认=ID")
                        n_role = gr.Dropdown(
                            choices=["user", "admin"], value="user", label="角色"
                        )
                    n_note = gr.Textbox(label="备注")
                    n_pass = gr.Textbox(
                        label="密码",
                        type="password",
                        placeholder="留空则默认=用户ID",
                    )
                    n_add = gr.Button("新增用户", variant="primary")
                    n_msg = gr.Markdown("")

                # ===== 爬虫页（仅管理员可见）=====
                with gr.Tab("爬虫", visible=False) as tab_crawl:
                    gr.Markdown("扩库：抓取 → `data/kb_staging/`，审核后再入库。")
                    with gr.Row():
                        c_kw = gr.Textbox(label="关键词", placeholder="比亚迪 海外")
                        c_from = gr.Textbox(label="开始", placeholder="YYYY-MM-DD")
                        c_to = gr.Textbox(label="结束", placeholder="YYYY-MM-DD")
                        c_topk = gr.Slider(1, 8, value=4, step=1, label="top_k")
                    with gr.Row():
                        c_run = gr.Button("手动爬取", variant="primary")
                        c_cfg = gr.Button("按配置批量")
                        c_refresh = gr.Button("刷新待审")
                    c_msg = gr.Markdown("")
                    staging_df = gr.Dataframe(
                        headers=["id", "status", "title", "query", "url", "chars"],
                        datatype=["str", "str", "str", "str", "str", "number"],
                        interactive=False,
                    )
                    promote_ids = gr.Textbox(
                        label="通过并入库的 id（逗号分隔）",
                        placeholder="doc_id_1, doc_id_2",
                    )
                    promote_btn = gr.Button("审核通过并 promote", variant="primary")

        # ----- login / logout -----
        def do_login(uid: str, pw: str):
            user = verify_login(uid, pw)
            if user is None:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "",
                    "",
                    "",
                    "",
                    "登录失败：用户名或密码错误",
                    "### ",
                    gr.update(choices=[], value=None),
                    [],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    [],
                    gr.update(choices=[], value=None),
                    "",
                    "",
                    "user",
                    "",
                    "",
                    [],
                )
            token = issue_token(user)
            sess = STORE.create(user.id)
            choices, _ = _session_choices(user.id)
            admin = is_admin(user)
            role_cn = "管理员" if admin else "普通用户"
            if admin:
                rows = list_users_public(include_builtin=False)
                first_id = rows[0]["id"] if rows else None
                pick = _user_pick_update(first_id)
                eid, ename, erole, enote, epass = _edit_form_for(first_id)
                users_rows = _users_table()
                staging_rows = _staging_table()
            else:
                pick = gr.update(choices=[], value=None)
                eid, ename, erole, enote, epass = "", "", "user", "", ""
                users_rows, staging_rows = [], []
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                token,
                user.id,
                user.role,
                sess.id,
                "",
                f"**{user.name}** · {role_cn}",
                gr.update(choices=choices, value=sess.id),
                [],
                gr.update(visible=admin),
                gr.update(visible=admin),
                users_rows,
                pick,
                eid,
                ename,
                erole,
                enote,
                epass,
                staging_rows,
            )

        login_btn.click(
            do_login,
            [login_user, login_pass],
            [
                login_view,
                main_view,
                token_state,
                user_id_state,
                role_state,
                session_id_state,
                login_msg,
                header,
                session_radio,
                chatbot,
                tab_users,
                tab_crawl,
                users_df,
                user_pick,
                e_id,
                e_name,
                e_role,
                e_note,
                e_pass,
                staging_df,
            ],
        )

        def do_logout(token: str):
            revoke_token(token)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                "",
                "",
                "",
                "",
                "### ",
                gr.update(choices=[], value=None),
                [],
                gr.update(visible=False),
                gr.update(visible=False),
            )

        logout_btn.click(
            do_logout,
            [token_state],
            [
                login_view,
                main_view,
                token_state,
                user_id_state,
                role_state,
                session_id_state,
                login_msg,
                header,
                session_radio,
                chatbot,
                tab_users,
                tab_crawl,
            ],
        )

        def do_new(token: str):
            user = _user_from_state(token)
            if user is None:
                raise gr.Error("请先登录")
            _IGNORE_SESSION_CHANGE["on"] = True
            try:
                sess = STORE.create(user.id)
                choices, _ = _session_choices(user.id)
                return sess.id, gr.update(choices=choices, value=sess.id), []
            finally:
                _IGNORE_SESSION_CHANGE["on"] = False

        new_btn.click(do_new, [token_state], [session_id_state, session_radio, chatbot])

        def do_refresh(token: str, sid: str):
            user = _user_from_state(token)
            if user is None:
                raise gr.Error("请先登录")
            choices, default_sid = _session_choices(user.id)
            value = sid if any(c[1] == sid for c in choices) else default_sid
            sess = STORE.get(user.id, value) if value else None
            _IGNORE_SESSION_CHANGE["on"] = True
            try:
                return (
                    value or "",
                    gr.update(choices=choices, value=value),
                    history_for_gradio(sess),
                )
            finally:
                _IGNORE_SESSION_CHANGE["on"] = False

        refresh_btn.click(
            do_refresh,
            [token_state, session_id_state],
            [session_id_state, session_radio, chatbot],
        )

        def do_select(token: str, sid: str, history: list):
            # 发送流式过程中 Radio 被更新会误触发：直接保留当前气泡
            if _IGNORE_SESSION_CHANGE["on"]:
                return sid or "", history
            user = _user_from_state(token)
            if user is None or not sid:
                return sid or "", history or []
            sess = STORE.get(user.id, sid)
            return sid, history_for_gradio(sess)

        session_radio.change(
            do_select,
            [token_state, session_radio, chatbot],
            [session_id_state, chatbot],
        )

        def do_delete(token: str, sid: str):
            user = _user_from_state(token)
            if user is None:
                raise gr.Error("请先登录")
            if sid:
                STORE.delete(user.id, sid)
            rows = STORE.list(user.id)
            if not rows:
                sess = STORE.create(user.id)
                sid2 = sess.id
            else:
                sid2 = rows[0]["id"]
            choices, _ = _session_choices(user.id)
            sess = STORE.get(user.id, sid2)
            _IGNORE_SESSION_CHANGE["on"] = True
            try:
                return sid2, gr.update(choices=choices, value=sid2), history_for_gradio(sess)
            finally:
                _IGNORE_SESSION_CHANGE["on"] = False

        del_btn.click(
            do_delete,
            [token_state, session_id_state],
            [session_id_state, session_radio, chatbot],
        )

        def do_send(token: str, sid: str, text: str, history: list):
            user = _user_from_state(token)
            if user is None:
                raise gr.Error("请先登录")
            q = (text or "").strip()
            if not q:
                yield history, "", sid
                return
            if not sid:
                sess = STORE.create(user.id)
                sid = sess.id

            admin = is_admin(user)
            history = [list(pair) for pair in (history or [])]
            history.append(
                [
                    q,
                    compose_streaming_bubble(
                        thinking_lines=[],
                        stage="开始研究…",
                        admin=admin,
                        thinking_open=True,
                    ),
                ]
            )
            thinking_lines: list[str] = []
            # 流式过程中不要更新 Radio，否则会触发 change 冲掉气泡
            _IGNORE_SESSION_CHANGE["on"] = True
            yield history, "", sid

            report_acc = ""
            try:
                for ev in iter_research_events(user=user, session_id=sid, query=q):
                    kind = ev.get("kind")
                    if kind in {"status", "event"}:
                        continue
                    if kind == "thinking":
                        line = str(ev.get("line") or "")
                        if line:
                            thinking_lines.append(line)
                        prog = str(ev.get("progress") or line or "思考中…")
                        history[-1][1] = compose_streaming_bubble(
                            thinking_lines=thinking_lines,
                            stage=prog,
                            admin=admin,
                            thinking_open=True,
                        )
                        yield history, "", sid
                    elif kind == "delta":
                        report_acc += str(ev.get("text") or "")
                        history[-1][1] = compose_streaming_bubble(
                            thinking_lines=thinking_lines,
                            stage="生成回答…",
                            report=report_acc,
                            admin=admin,
                            thinking_open=False,
                        )
                        yield history, "", sid
                    elif kind == "done":
                        content = str(ev.get("content") or "")
                        if not content:
                            content = format_assistant_message(
                                str(ev.get("report") or report_acc),
                                thinking_lines=list(
                                    ev.get("thinking_lines") or thinking_lines
                                ),
                                user=user,
                                sources=list(ev.get("sources") or []),
                                meta=ev.get("meta") if admin else None,
                                thinking_open=False,
                                stage="已完成",
                            )
                        history[-1][1] = content
                        yield history, "", sid
                    elif kind == "error":
                        msg_e = str(ev.get("message") or "未知错误")
                        history[-1][1] = f"**错误：** {msg_e}"
                        yield history, "", sid
                        return
            except Exception as exc:  # noqa: BLE001
                history[-1][1] = f"**错误：** {exc}"
                yield history, "", sid
            finally:
                _IGNORE_SESSION_CHANGE["on"] = False

            # 结束后再刷新左侧标题（此时 STORE 已有完整消息）
            choices, _ = _session_choices(user.id)
            _IGNORE_SESSION_CHANGE["on"] = True
            try:
                yield history, "", sid
            finally:
                _IGNORE_SESSION_CHANGE["on"] = False

        # 发送：不把 session_radio 放进 outputs，避免流式时被 change 覆盖
        send_btn.click(
            do_send,
            [token_state, session_id_state, msg, chatbot],
            [chatbot, msg, session_id_state],
        ).then(
            # 发送结束后刷新会话标题列表
            do_refresh,
            [token_state, session_id_state],
            [session_id_state, session_radio, chatbot],
        )
        msg.submit(
            do_send,
            [token_state, session_id_state, msg, chatbot],
            [chatbot, msg, session_id_state],
        ).then(
            do_refresh,
            [token_state, session_id_state],
            [session_id_state, session_radio, chatbot],
        )

        # ----- admin users -----
        def _require_admin(token: str) -> User:
            user = _user_from_state(token)
            if user is None or not is_admin(user):
                raise gr.Error("需要管理员")
            return user

        def admin_refresh_users(token: str):
            _require_admin(token)
            rows = list_users_public(include_builtin=False)
            first = rows[0]["id"] if rows else None
            eid, ename, erole, enote, epass = _edit_form_for(first)
            return (
                _users_table(),
                _user_pick_update(first),
                eid,
                ename,
                erole,
                enote,
                epass,
                "",
            )

        def admin_pick_user(token: str, uid: str):
            _require_admin(token)
            eid, ename, erole, enote, epass = _edit_form_for(uid)
            return eid, ename, erole, enote, epass, ""

        def admin_save_user(token: str, uid, name, role, note, pw):
            _require_admin(token)
            try:
                upsert_user(
                    user_id=uid,
                    name=name or "",
                    role=role or "user",
                    note=note or "",
                    password=pw if (pw or "").strip() else None,
                )
            except ValueError as exc:
                return _users_table(), _user_pick_update(uid), f"失败：{exc}", ""
            return (
                _users_table(),
                _user_pick_update(uid),
                f"已保存修改：`{uid}`",
                "",
            )

        def admin_del_user(token: str, uid):
            _require_admin(token)
            try:
                ok = delete_user(uid)
            except ValueError as exc:
                return (
                    _users_table(),
                    _user_pick_update(uid),
                    "",
                    "",
                    "user",
                    "",
                    "",
                    f"失败：{exc}",
                )
            if not ok:
                return (
                    _users_table(),
                    _user_pick_update(uid),
                    uid,
                    "",
                    "user",
                    "",
                    "",
                    "用户不存在（可能未写入 yaml）",
                )
            rows = list_users_public(include_builtin=False)
            first = rows[0]["id"] if rows else None
            eid, ename, erole, enote, epass = _edit_form_for(first)
            return (
                _users_table(),
                _user_pick_update(first),
                eid,
                ename,
                erole,
                enote,
                epass,
                f"已删除用户 `{uid}`",
            )

        def admin_add_user(token: str, uid, name, role, note, pw):
            _require_admin(token)
            uid = (uid or "").strip()
            keep = (uid, name or "", role or "user", note or "", pw or "")
            if not uid:
                return (
                    _users_table(),
                    _user_pick_update(),
                    "用户 ID 不能为空",
                    *keep,
                )
            existing = {u["id"] for u in list_users_public(include_builtin=False)}
            if uid in existing:
                return (
                    _users_table(),
                    _user_pick_update(uid),
                    f"用户 `{uid}` 已存在，请用上方编辑",
                    *keep,
                )
            try:
                upsert_user(
                    user_id=uid,
                    name=name or uid,
                    role=role or "user",
                    note=note or "",
                    password=pw if (pw or "").strip() else uid,
                )
            except ValueError as exc:
                return _users_table(), _user_pick_update(), f"失败：{exc}", *keep
            return (
                _users_table(),
                _user_pick_update(uid),
                f"已新增用户 `{uid}`",
                "",
                "",
                "user",
                "",
                "",
            )

        u_refresh.click(
            admin_refresh_users,
            [token_state],
            [users_df, user_pick, e_id, e_name, e_role, e_note, e_pass, u_msg],
        )
        user_pick.change(
            admin_pick_user,
            [token_state, user_pick],
            [e_id, e_name, e_role, e_note, e_pass, u_msg],
        )
        e_save.click(
            admin_save_user,
            [token_state, e_id, e_name, e_role, e_note, e_pass],
            [users_df, user_pick, u_msg, e_pass],
        )
        e_del.click(
            admin_del_user,
            [token_state, e_id],
            [users_df, user_pick, e_id, e_name, e_role, e_note, e_pass, u_msg],
        )
        n_add.click(
            admin_add_user,
            [token_state, n_id, n_name, n_role, n_note, n_pass],
            [users_df, user_pick, n_msg, n_id, n_name, n_role, n_note, n_pass],
        )

        # ----- admin crawl -----
        def admin_refresh_staging(token: str):
            user = _user_from_state(token)
            if user is None or not is_admin(user):
                raise gr.Error("需要管理员")
            return _staging_table(), ""

        def admin_crawl(token: str, kw, df, dt, topk):
            user = _user_from_state(token)
            if user is None or not is_admin(user):
                raise gr.Error("需要管理员")
            try:
                results = run_manual_crawl(
                    keyword=kw, date_from=df or "", date_to=dt or "", top_k=int(topk)
                )
            except Exception as exc:  # noqa: BLE001
                return _staging_table(), f"爬取失败：{exc}"
            ok = sum(1 for r in results if r.get("ok"))
            return (
                _staging_table(),
                f"手动爬取完成：成功 {ok}/{len(results)} 条\n```json\n"
                f"{json.dumps(results, ensure_ascii=False, indent=2)[:2000]}\n```",
            )

        def admin_crawl_cfg(token: str):
            user = _user_from_state(token)
            if user is None or not is_admin(user):
                raise gr.Error("需要管理员")
            try:
                results = run_config_crawl()
            except Exception as exc:  # noqa: BLE001
                return _staging_table(), f"批量爬取失败：{exc}"
            ok = sum(1 for r in results if r.get("ok"))
            return _staging_table(), f"配置批量爬取：成功 {ok}/{len(results)} 条"

        def admin_promote(token: str, ids_text: str):
            user = _user_from_state(token)
            if user is None or not is_admin(user):
                raise gr.Error("需要管理员")
            ids = [
                x.strip()
                for x in (ids_text or "").replace("，", ",").split(",")
                if x.strip()
            ]
            try:
                written = approve_and_promote(ids)
            except Exception as exc:  # noqa: BLE001
                return _staging_table(), f"失败：{exc}"
            return _staging_table(), f"已 promote {len(written)} 个文件：\n" + "\n".join(written)

        c_refresh.click(admin_refresh_staging, [token_state], [staging_df, c_msg])
        c_run.click(admin_crawl, [token_state, c_kw, c_from, c_to, c_topk], [staging_df, c_msg])
        c_cfg.click(admin_crawl_cfg, [token_state], [staging_df, c_msg])
        promote_btn.click(admin_promote, [token_state, promote_ids], [staging_df, c_msg])

    return demo
