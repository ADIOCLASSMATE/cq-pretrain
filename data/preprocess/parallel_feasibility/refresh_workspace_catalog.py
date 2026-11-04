"""Refresh Inspire's legacy workspace cache from the current ListWorkspaces API.

The current platform's GetRoutes response omits userWorkspaceList. This helper
updates only local session/catalog metadata from a checked live response; it
does not change CLI code, submit workloads, or print authentication material.
Run with the installed Inspire Python interpreter.
"""
from inspire.platform.web.session import get_web_session
from inspire.platform.web.browser_api.core import _request_json, _v2_result, _get_base_url
from inspire.services.catalog.resource_index import ResourceIdentity, ResourceIndex, scope_for_session

ACCOUNT = "lipengfei"
WORKSPACE = "昇腾卡公共空间"


def main():
    session = get_web_session(account=ACCOUNT)
    result = _v2_result(_request_json(session, "POST",
        "/api/v2/workspace?Action=ListWorkspaces",
        referer=_get_base_url() + "/jobs/spacesOverview",
        body={"PageNumber": 1, "page_size": 100}, timeout=15))
    rows = result["items"]
    if len(rows) != int(result["total"]):
        raise ValueError("Incomplete workspace catalog")
    target = [r for r in rows if r["name"] == WORKSPACE]
    if len(target) != 1 or any(type(r["is_fair_workspace"]) is not bool for r in rows):
        raise ValueError("Workspace identity or scheduling policy not confirmed")
    session.all_workspace_ids = [r["id"] for r in rows]
    session.all_workspace_names = {r["id"]: r["name"] for r in rows}
    session.all_workspace_fair_scheduling = {r["id"]: r["is_fair_workspace"] for r in rows}
    session.workspace_id = target[0]["id"]
    session.save(account=ACCOUNT)
    index = ResourceIndex.for_account()
    scope = scope_for_session(session, resource_type="workspace")
    if index is None or scope is None:
        raise RuntimeError("No account-scoped workspace catalog")
    index.upsert(scope, [ResourceIdentity(resource_id=r["id"], name=r["name"]) for r in rows])
    print("Live workspace catalog refreshed:", [r["name"] for r in rows])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Transport exceptions can embed authenticated request headers.
        raise SystemExit(f"Workspace refresh failed: {type(error).__name__}") from None
