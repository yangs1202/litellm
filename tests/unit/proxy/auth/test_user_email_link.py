"""How a sign-in's email claim links to, creates and fills LiteLLM user rows."""

import asyncio
import re
import uuid
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.auth_checks import get_user_object, read_email_verified
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.types import CustomOpenID
from litellm.proxy.management_endpoints.ui_sso import (
    generic_response_convertor,
    get_user_info_from_db,
)
from litellm.types.proxy.auth.auth_checks import UserNotFoundError


@pytest.fixture(autouse=True)
def _forget_user_lookups() -> None:
    auth_checks.last_db_access_time.clear()
    auth_checks.email_in_use_checked_at.clear()


def _ilike(value: str, pattern: str) -> bool:
    """Postgres ILIKE, which Prisma uses for an insensitive `equals`: `%` and `_` are wildcards, `\\` escapes."""
    regex = ""
    chars = iter(pattern)
    for char in chars:
        if char == "\\":
            regex += re.escape(next(chars, ""))
        elif char == "%":
            regex += ".*"
        elif char == "_":
            regex += "."
        else:
            regex += re.escape(char)
    return re.fullmatch(regex, value, flags=re.IGNORECASE | re.DOTALL) is not None


@pytest.fixture
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_checks.time, "time", lambda: 1_000_000.0)


def _matches(row: LiteLLM_UserTable, where: Mapping[str, object]) -> bool:
    for field, condition in where.items():
        value = getattr(row, field)
        if isinstance(condition, Mapping) and "equals" in condition:
            if not isinstance(value, str) or not _ilike(value, str(condition["equals"])):
                return False
        elif isinstance(condition, Mapping) and "not" in condition:
            if value == condition["not"]:
                return False
        elif value != condition:
            return False
    return True


class _UserTable:
    """The user-table calls the sign-in lookup makes, over a list of rows."""

    def __init__(self, *rows: LiteLLM_UserTable) -> None:
        self.rows = list(rows)
        self.first_lookups: list[Mapping[str, object]] = []

    def row(self, user_id: str) -> LiteLLM_UserTable:
        return next(r for r in self.rows if r.user_id == user_id)

    def with_email(self, email: str) -> list[str]:
        return [r.user_id for r in self.rows if (r.user_email or "").casefold() == email.casefold()]

    async def find_unique(self, where: Mapping[str, object], include: object = None) -> LiteLLM_UserTable | None:
        return next((r for r in self.rows if _matches(r, where)), None)

    async def find_first(self, where: Mapping[str, object], include: object = None) -> LiteLLM_UserTable | None:
        self.first_lookups.append(where)
        return next((r for r in self.rows if _matches(r, where)), None)

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> LiteLLM_UserTable | None:
        row = await self.find_unique(where)
        if row is None:
            return None
        updated = row.model_copy(update=dict(data))
        self.rows[self.rows.index(row)] = updated
        return updated

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        matched = [r for r in self.rows if _matches(r, where)]
        for row in matched:
            self.rows[self.rows.index(row)] = row.model_copy(update=dict(data))
        return len(matched)

    async def create(self, data: Mapping[str, object], include: object = None) -> LiteLLM_UserTable:
        row = LiteLLM_UserTable(**{"organization_memberships": [], **data})
        self.rows.append(row)
        return row


def _user(user_id: str, email: str | None, sso_user_id: str | None = None) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id,
        user_email=email,
        sso_user_id=sso_user_id,
        user_role="proxy_admin",
        organization_memberships=[],
    )


def _prisma(table: _UserTable) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_usertable = table
    return prisma


async def _sign_in(
    table: _UserTable,
    *,
    claim_id: str,
    email: str | None,
    email_verified: bool | None = None,
    upsert: bool = False,
    cache: UserApiKeyCache | None = None,
) -> LiteLLM_UserTable | None:
    result = await get_user_object(
        user_id=claim_id,
        prisma_client=_prisma(table),
        user_api_key_cache=cache or UserApiKeyCache(),
        user_id_upsert=upsert,
        sso_user_id=claim_id,
        user_email=email,
        email_verified=email_verified,
    )
    await asyncio.sleep(0)  # let the background sso_user_id write run
    return result


def _claim_id() -> str:
    return f"idp-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("email_verified", [None, True, False])
@pytest.mark.parametrize("email", ["owner@example.com", "OWNER@Example.com", "Owner@EXAMPLE.com"])
async def test_email_never_selects_a_row_linked_to_another_sso_id(email: str, email_verified: bool | None) -> None:
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id="idp-owner"))
    claim_id = _claim_id()

    with pytest.raises(UserNotFoundError):
        await _sign_in(table, claim_id=claim_id, email=email, email_verified=email_verified)

    assert table.row("owner").sso_user_id == "idp-owner"
    assert [r.user_id for r in table.rows] == ["owner"]


@pytest.mark.asyncio
async def test_new_row_after_a_refused_email_carries_no_email() -> None:
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id="idp-owner"))
    claim_id = _claim_id()

    user = await _sign_in(table, claim_id=claim_id, email="owner@example.com", upsert=True)

    assert user is not None and user.user_id == claim_id
    assert user.user_email is None
    assert table.with_email("owner@example.com") == ["owner"]
    assert table.row("owner").sso_user_id == "idp-owner"


@pytest.mark.asyncio
async def test_new_row_never_fills_in_the_refused_email_on_later_sign_ins() -> None:
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id="idp-owner"))
    claim_id = _claim_id()
    cache = UserApiKeyCache()

    for _ in range(3):
        user = await _sign_in(table, claim_id=claim_id, email="owner@example.com", upsert=True, cache=cache)
        assert user is not None and user.user_id == claim_id

    assert table.with_email("owner@example.com") == ["owner"]


@pytest.mark.asyncio
@pytest.mark.parametrize("upsert", [False, True])
async def test_email_marked_unverified_never_selects_an_unlinked_row(upsert: bool) -> None:
    table = _UserTable(_user("owner", "owner@example.com"))
    claim_id = _claim_id()

    if upsert:
        user = await _sign_in(table, claim_id=claim_id, email="owner@example.com", email_verified=False, upsert=True)
        assert user is not None and user.user_id == claim_id and user.user_email is None
    else:
        with pytest.raises(UserNotFoundError):
            await _sign_in(table, claim_id=claim_id, email="owner@example.com", email_verified=False)

    assert table.row("owner").sso_user_id is None
    assert table.with_email("owner@example.com") == ["owner"]


@pytest.mark.asyncio
@pytest.mark.parametrize("email_verified", [None, True])
async def test_email_links_an_unlinked_row_when_not_marked_unverified(email_verified: bool | None) -> None:
    table = _UserTable(_user("owner", "owner@example.com"))
    claim_id = _claim_id()

    user = await _sign_in(table, claim_id=claim_id, email="Owner@Example.com", email_verified=email_verified)

    assert user is not None and user.user_id == "owner"
    assert table.row("owner").sso_user_id == claim_id


@pytest.mark.asyncio
@pytest.mark.parametrize("email_verified", [None, True, False])
async def test_linked_owner_keeps_signing_in_by_sso_id(email_verified: bool | None) -> None:
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id="idp-owner"))

    user = await _sign_in(table, claim_id="idp-owner", email="owner@example.com", email_verified=email_verified)

    assert user is not None and user.user_id == "owner"


@pytest.mark.asyncio
@pytest.mark.parametrize("email_verified", [None, True, False])
async def test_owner_keeps_signing_in_by_user_id(email_verified: bool | None) -> None:
    table = _UserTable(_user("owner", None))

    user = await _sign_in(table, claim_id="owner", email="owner@example.com", email_verified=email_verified)

    assert user is not None and user.user_id == "owner"
    assert user.user_email == "owner@example.com"


@pytest.mark.asyncio
async def test_new_row_keeps_an_email_no_other_row_holds() -> None:
    table = _UserTable(_user("someone-else", "other@example.com"))
    claim_id = _claim_id()

    user = await _sign_in(table, claim_id=claim_id, email="new@example.com", upsert=True)

    assert user is not None and user.user_email == "new@example.com"


@pytest.mark.asyncio
async def test_null_email_is_not_filled_with_an_email_another_row_holds() -> None:
    table = _UserTable(
        _user("owner", "owner@example.com", sso_user_id="idp-owner"),
        _user("second", None),
    )

    user = await _sign_in(table, claim_id="second", email="OWNER@example.com")

    assert user is not None and user.user_id == "second" and user.user_email is None
    assert table.row("second").user_email is None


@pytest.mark.asyncio
async def test_cached_row_is_not_filled_with_an_email_another_row_holds() -> None:
    table = _UserTable(_user("owner", "owner@example.com"), _user("second", None))
    cache = UserApiKeyCache()
    await _sign_in(table, claim_id="second", email=None, cache=cache)

    user = await _sign_in(table, claim_id="second", email="owner@example.com", cache=cache)

    assert user is not None and user.user_email is None
    assert table.row("second").user_email is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("FALSE", False),
        (" False ", False),
        ("True", True),
        ("yes", None),
        ("", None),
        (0, None),
        (1, None),
        (None, None),
    ],
)
def test_read_email_verified(value: object, expected: bool | None) -> None:
    assert read_email_verified(value) is expected


@pytest.mark.parametrize("value, expected", [(False, False), ("false", False), (True, True), (None, None)])
def test_jwt_handler_reads_email_verified_from_global_and_issuer_tokens(value: object, expected: bool | None) -> None:
    handler = JWTHandler()
    claims = {} if value is None else {"email_verified": value}
    issuer_normalized = {**claims, JWTHandler.LITELLM_USER_EMAIL_CLAIM: "owner@example.com"}

    assert handler.get_email_verified(claims) is expected
    assert handler.get_email_verified(issuer_normalized) is expected


@pytest.mark.parametrize("value, expected", [(False, False), ("false", False), ("true", True), (None, None)])
def test_generic_sso_response_carries_email_verified(value: object, expected: bool | None) -> None:
    response = {"preferred_username": "user-1", "email": "owner@example.com", "sub": "user-1"}
    if value is not None:
        response["email_verified"] = value
    jwt_handler = MagicMock()
    jwt_handler.get_all_jwt_team_ids.return_value = []

    result = generic_response_convertor(response=response, jwt_handler=jwt_handler)

    assert result.email_verified is expected


async def _sso_sign_in(table: _UserTable, result: CustomOpenID | dict[str, object]) -> object:
    async def keep_lookup_result(**kwargs: object) -> object:
        return kwargs["user_info"]

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.upsert_sso_user",
        side_effect=keep_lookup_result,
    ):
        user = await get_user_info_from_db(
            result=result,
            prisma_client=_prisma(table),
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=MagicMock(),
            user_email="owner@example.com",
            user_defined_values=None,
        )
    await asyncio.sleep(0)
    return user


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        CustomOpenID(id="idp-new", email="owner@example.com", team_ids=[], email_verified=False),
        {"id": "idp-new", "email": "owner@example.com", "email_verified": "false"},
    ],
)
async def test_sso_sign_in_with_unverified_email_leaves_the_unlinked_row_alone(
    result: CustomOpenID | dict[str, object],
) -> None:
    table = _UserTable(_user("owner", "owner@example.com"))

    user = await _sso_sign_in(table, result)

    assert user is None
    assert table.row("owner").sso_user_id is None


@pytest.mark.asyncio
async def test_sso_sign_in_with_verified_email_links_the_unlinked_row() -> None:
    table = _UserTable(_user("owner", "owner@example.com"))

    user = await _sso_sign_in(
        table, CustomOpenID(id="idp-new", email="owner@example.com", team_ids=[], email_verified=True)
    )

    assert isinstance(user, LiteLLM_UserTable) and user.user_id == "owner"
    assert table.row("owner").sso_user_id == "idp-new"


@pytest.mark.asyncio
@pytest.mark.parametrize("value, expected", [(False, False), ("false", False), (True, True), (None, None)])
async def test_jwt_sign_in_passes_email_verified_to_the_lookup(value: object, expected: bool | None) -> None:
    from litellm.proxy.auth.handle_jwt import JWTAuthManager

    captured: dict[str, object] = {}

    async def load_user(**kwargs: object) -> None:
        captured.update(kwargs)
        return None

    handler = MagicMock()
    handler.is_upsert_user_id.return_value = False
    with patch("litellm.proxy.auth.handle_jwt.get_user_object", side_effect=load_user):
        await JWTAuthManager.get_objects(
            user_id="idp-1",
            user_email="owner@example.com",
            org_id=None,
            end_user_id=None,
            team_id=None,
            valid_user_email=None,
            jwt_handler=handler,
            prisma_client=SimpleNamespace(),
            user_api_key_cache=UserApiKeyCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
            route="/v1/models",
            email_verified=JWTHandler().get_email_verified({} if value is None else {"email_verified": value}),
        )

    assert captured["email_verified"] is expected
    assert captured["sso_user_id"] == "idp-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "email, expected_team_email",
    [("owner@example.com", None), ("new@example.com", "new@example.com")],
)
async def test_default_team_join_uses_the_email_the_new_row_stored(email: str, expected_team_email: str | None) -> None:
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id="idp-owner"))
    endpoints = "litellm.proxy.management_endpoints.internal_user_endpoints"
    with (
        patch(f"{endpoints}.check_if_default_team_set", return_value=["team-1"]),
        patch(f"{endpoints}.add_new_user_to_default_team") as join,
    ):
        user = await _sign_in(table, claim_id=_claim_id(), email=email, upsert=True)

    assert user is not None and user.user_email == expected_team_email
    assert join.call_args.kwargs["user_email"] == expected_team_email


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_sso_user_id, email_verified",
    [("idp-owner", None), (None, False), ("idp-owner", False)],
)
async def test_opt_out_restores_linking_any_row_by_email(
    monkeypatch: pytest.MonkeyPatch, owner_sso_user_id: str | None, email_verified: bool | None
) -> None:
    monkeypatch.setenv("LITELLM_EMAIL_FALLBACK_LINKS_ANY_USER", "true")
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id=owner_sso_user_id))
    claim_id = _claim_id()

    user = await _sign_in(table, claim_id=claim_id, email="owner@example.com", email_verified=email_verified)

    assert user is not None and user.user_id == "owner"
    assert table.row("owner").sso_user_id == claim_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "emails", [["owner@example.com"] * 3, ["owner@example.com", "OWNER@example.com", "Owner@Example.COM"]]
)
async def test_email_in_use_check_runs_once_per_window_on_cached_sign_ins(emails: list[str], fixed_clock: None) -> None:
    table = _UserTable(_user("owner", "owner@example.com"), _user("second", None))
    cache = UserApiKeyCache()
    await _sign_in(table, claim_id="second", email=None, cache=cache)
    table.first_lookups.clear()
    for email in emails:
        user = await _sign_in(table, claim_id="second", email=email, cache=cache)
        assert user is not None and user.user_email is None

    assert len(table.first_lookups) == 1


@pytest.mark.asyncio
async def test_opt_out_restores_storing_an_email_another_row_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_EMAIL_FALLBACK_LINKS_ANY_USER", "true")
    table = _UserTable(_user("owner", "owner@example.com"), _user("second", None))

    user = await _sign_in(table, claim_id="second", email="owner@example.com")

    assert user is not None and user.user_email == "owner@example.com"
    assert table.with_email("owner@example.com") == ["owner", "second"]


@pytest.mark.asyncio
async def test_email_in_use_check_stays_once_per_window_while_other_users_load(fixed_clock: None) -> None:
    table = _UserTable(_user("owner", "owner@example.com"), _user("second", None))
    cache = UserApiKeyCache()
    await _sign_in(table, claim_id="second", email=None, cache=cache)
    table.first_lookups.clear()

    for round_number in range(3):
        for i in range(150):
            auth_checks.last_db_access_time[f"user_id:other-{round_number}-{i}"] = (None, 0.0)
        user = await _sign_in(table, claim_id="second", email="owner@example.com", cache=cache)
        assert user is not None and user.user_email is None

    assert len(table.first_lookups) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("other_email", ["ann.lee@example.com", "ann-lee@example.com", "annXlee@example.com"])
async def test_email_with_an_underscore_matches_only_itself(other_email: str) -> None:
    table = _UserTable(_user("other", other_email, sso_user_id="idp-other"), _user("own", None))

    linked = await _sign_in(table, claim_id=_claim_id(), email="ann_lee@example.com", upsert=True)
    own = await _sign_in(table, claim_id="own", email="ann_lee@example.com")

    assert linked is not None and linked.user_email == "ann_lee@example.com"
    assert table.row("other").sso_user_id == "idp-other"
    assert own is not None and own.user_email is None
    assert sorted(table.with_email("ann_lee@example.com")) == [linked.user_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("email", ["a%b@example.com", "back\\slash@example.com"])
async def test_email_fallback_finds_an_email_with_a_wildcard_character(email: str) -> None:
    table = _UserTable(_user("owner", email), _user("other", "aXb@example.com"))
    claim_id = _claim_id()

    user = await _sign_in(table, claim_id=claim_id, email=email.upper())

    assert user is not None and user.user_id == "owner"
    assert table.row("owner").sso_user_id == claim_id


@pytest.mark.asyncio
async def test_stale_cached_row_picks_up_the_email_its_own_row_already_holds() -> None:
    table = _UserTable(_user("own", None))
    cache = UserApiKeyCache()
    await _sign_in(table, claim_id="own", email=None, cache=cache)
    await table.update(where={"user_id": "own"}, data={"user_email": "own@example.com"})

    user = await _sign_in(table, claim_id="own", email="own@example.com", cache=cache)

    assert user is not None and user.user_email == "own@example.com"


@pytest.mark.asyncio
@pytest.mark.parametrize("email", ["%@example.com", "own%@example.com", "_wn@example.com"])
async def test_email_with_a_wildcard_character_links_no_other_row(email: str) -> None:
    table = _UserTable(_user("owner", "own@example.com"))
    claim_id = _claim_id()

    user = await _sign_in(table, claim_id=claim_id, email=email, upsert=True)

    assert user is not None and user.user_id == claim_id and user.user_email == email
    assert table.row("owner").sso_user_id is None


@pytest.mark.asyncio
async def test_concurrent_cached_sign_ins_share_one_email_in_use_check(fixed_clock: None) -> None:
    table = _UserTable(_user("owner", "owner@example.com"), _user("second", None))
    cache = UserApiKeyCache()
    await _sign_in(table, claim_id="second", email=None, cache=cache)
    table.first_lookups.clear()

    users = await asyncio.gather(
        *(_sign_in(table, claim_id="second", email="owner@example.com", cache=cache) for _ in range(8))
    )

    assert all(user is not None and user.user_email is None for user in users)
    assert len(table.first_lookups) == 1


@pytest.mark.asyncio
async def test_one_rows_skipped_email_does_not_hold_back_another_rows_fill() -> None:
    table = _UserTable(_user("owner", "owner@example.com"), _user("second", None), _user("third", None))
    second_cache, third_cache = UserApiKeyCache(), UserApiKeyCache()
    await _sign_in(table, claim_id="second", email=None, cache=second_cache)
    await _sign_in(table, claim_id="third", email=None, cache=third_cache)
    await _sign_in(table, claim_id="second", email="owner@example.com", cache=second_cache)
    await table.update(where={"user_id": "owner"}, data={"user_email": "moved@example.com"})

    third = await _sign_in(table, claim_id="third", email="owner@example.com", cache=third_cache)

    assert third is not None and third.user_email == "owner@example.com"


async def _sso_new_user_emails(table: _UserTable, user_id: str) -> list[str | None]:
    stored: list[str | None] = []

    async def record_new_user(**kwargs: object) -> None:
        values = kwargs["user_defined_values"]
        assert isinstance(values, dict)
        stored.extend([kwargs["user_email"], values["user_email"]])

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.upsert_sso_user",
        side_effect=record_new_user,
    ):
        await get_user_info_from_db(
            result=CustomOpenID(id=user_id, email="owner@example.com", team_ids=[]),
            prisma_client=_prisma(table),
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=MagicMock(),
            user_email="owner@example.com",
            user_defined_values={
                "models": [],
                "user_id": user_id,
                "user_email": "owner@example.com",
                "max_budget": None,
                "user_role": None,
                "budget_duration": None,
            },
        )
    return stored


@pytest.mark.asyncio
async def test_sso_new_user_gets_no_email_another_user_holds() -> None:
    table = _UserTable(_user("owner", "owner@example.com", sso_user_id="idp-owner"))

    stored = await _sso_new_user_emails(table, "idp-new")

    assert stored == [None, None]
    assert table.row("owner").sso_user_id == "idp-owner"


@pytest.mark.asyncio
async def test_sso_new_user_keeps_an_email_no_other_user_holds() -> None:
    table = _UserTable(_user("other", "other@example.com"))

    stored = await _sso_new_user_emails(table, "idp-new")

    assert stored == ["owner@example.com", "owner@example.com"]
