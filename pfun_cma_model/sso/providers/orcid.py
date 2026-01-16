"""ORCiD SSO Oauth Helper class."""

from typing import TYPE_CHECKING, ClassVar, Optional

from fastapi_sso.sso.base import DiscoveryDocument, OpenID, SSOBase

if TYPE_CHECKING:
    import httpx  # pragma: no cover


class OrcidSSO(SSOBase):
    """Class providing login via Orcid OAuth."""

    provider = "orcid"
    scope: ClassVar = ["profile"]

    async def openid_from_response(self, response: dict, session: Optional["httpx.AsyncClient"] = None) -> OpenID:
        """Return OpenID from user information provided by Google."""
        info = response.get("user")
        if not info:
            raise SSOLoginError(401, "Failed to process login via Orcid")
        return OpenID(
            id=info["encodedId"],
            first_name=info["fullName"],
            display_name=info["displayName"],
            picture=info["avatar"],
            provider=self.provider,
        )

    async def get_discovery_document(self) -> DiscoveryDocument:
        """Get document containing handy urls."""
        return {
            "authorization_endpoint": "https://www.Orcid.com/oauth2/authorize?response_type=code",
            "token_endpoint": "https://api.Orcid.com/oauth2/token",
            "userinfo_endpoint": "https://api.Orcid.com/1/user/-/profile.json",
        }
