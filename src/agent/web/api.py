"""FastAPI routes including SSE chat stream."""

import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.auth import User, is_admin
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
    list_users_public,
    upsert_user,
    verify_login,
)


class LoginBody(BaseModel):
    user_id: str
    password: str = ""


class NewSessionBody(BaseModel):
    title: str = "新会话"


class ChatBody(BaseModel):
    query: str
    session_id: str


class UserUpsertBody(BaseModel):
    user_id: str
    name: str = ""
    role: str = "user"
    note: str = ""
    password: Optional[str] = None


class CrawlManualBody(BaseModel):
    keyword: str
    date_from: str = ""
    date_to: str = ""
    top_k: int = Field(default=4, ge=1, le=8)


class PromoteBody(BaseModel):
    ids: List[str]


def _user_from_auth(
    authorization: Optional[str], x_token: Optional[str] = None
) -> Tuple[User, Optional[str]]:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif x_token:
        token = x_token.strip()
    user = resolve_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user, token


def _sse_pack(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def mount_api(app: FastAPI) -> None:
    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/login")
    def login(body: LoginBody) -> Dict[str, Any]:
        user = verify_login(body.user_id, body.password)
        if user is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = issue_token(user)
        return {
            "token": token,
            "user": {"id": user.id, "name": user.name, "role": user.role, "note": user.note},
        }

    @app.post("/api/logout")
    def logout(
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, bool]:
        _, token = _user_from_auth(authorization, x_token)
        revoke_token(token)
        return {"ok": True}

    @app.get("/api/me")
    def me(
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, str]:
        user, _ = _user_from_auth(authorization, x_token)
        return {"id": user.id, "name": user.name, "role": user.role, "note": user.note}

    @app.get("/api/sessions")
    def sessions(
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        return {"sessions": STORE.list(user.id)}

    @app.post("/api/sessions")
    def create_session(
        body: NewSessionBody,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        sess = STORE.create(user.id, title=body.title or "新会话")
        return sess.to_dict()

    @app.get("/api/sessions/{session_id}")
    def get_session(
        session_id: str,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        sess = STORE.get(user.id, session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        data = sess.to_dict()
        if not is_admin(user):
            for m in data.get("messages") or []:
                m["events"] = []
            data["events"] = []
        data["history"] = history_for_gradio(sess)
        return data

    @app.delete("/api/sessions/{session_id}")
    def delete_session(
        session_id: str,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, bool]:
        user, _ = _user_from_auth(authorization, x_token)
        ok = STORE.delete(user.id, session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/stream")
    def stream_chat(
        session_id: str,
        body: ChatBody,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ):
        """SSE：思考过程 / 增量回答 / 完成。"""
        user, _ = _user_from_auth(authorization, x_token)
        if body.session_id and body.session_id != session_id:
            raise HTTPException(status_code=400, detail="session_id 不一致")

        def gen():
            for ev in iter_research_events(user=user, session_id=session_id, query=body.query):
                yield _sse_pack(ev)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/admin/users")
    def admin_users(
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        return {"users": list_users_public()}

    @app.post("/api/admin/users")
    def admin_upsert_user(
        body: UserUpsertBody,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, str]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        try:
            row = upsert_user(
                user_id=body.user_id,
                name=body.name,
                role=body.role,
                note=body.note,
                password=body.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return row

    @app.delete("/api/admin/users/{user_id}")
    def admin_delete_user(
        user_id: str,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, bool]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        try:
            ok = delete_user(user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not ok:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"ok": True}

    @app.get("/api/admin/staging")
    def admin_staging(
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        return {"docs": list_staging()}

    @app.post("/api/admin/crawl/manual")
    def admin_crawl_manual(
        body: CrawlManualBody,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        try:
            results = run_manual_crawl(
                keyword=body.keyword,
                date_from=body.date_from,
                date_to=body.date_to,
                top_k=body.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ok = sum(1 for r in results if r.get("ok"))
        return {"ok": ok, "total": len(results), "results": results}

    @app.post("/api/admin/crawl/config")
    def admin_crawl_config(
        include_optional: bool = Query(default=False),
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        results = run_config_crawl(include_optional=include_optional)
        ok = sum(1 for r in results if r.get("ok"))
        return {"ok": ok, "total": len(results), "results": results}

    @app.post("/api/admin/staging/promote")
    def admin_promote(
        body: PromoteBody,
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None, alias="X-Token"),
    ) -> Dict[str, Any]:
        user, _ = _user_from_auth(authorization, x_token)
        if not is_admin(user):
            raise HTTPException(status_code=403, detail="需要管理员")
        try:
            written = approve_and_promote(body.ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"promoted": written}
