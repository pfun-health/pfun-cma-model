"""ORCiD SSO Oauth Helper class."""

from typing import ClassVar, Optional
from fastapi_sso.sso.base import DiscoveryDocument, OpenID, SSOBase, SSOLoginError
import httpx


class OrcidSSO(SSOBase):
    """Class providing login via Orcid OAuth.

    ref: https://github.com/ORCID/ORCID-Source/tree/development/orcid-api-web
    """

    provider = "orcid"
    scope: ClassVar = ["openid"]

    async def openid_from_response(self, response: dict, session: Optional["httpx.AsyncClient"] = None) -> OpenID:
        """Return OpenID user information, as provided by ORCiD."""
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

    async def openid_from_token(self, id_token: dict, session: Optional[httpx.AsyncClient] = None) -> OpenID:
        """Converts an ID token from the provider's token endpoint to an OpenID object.

        Args:
            id_token (dict): The id token data retrieved from the token endpoint.
            session: (Optional[httpx.AsyncClient]): The HTTPX AsyncClient session.

        Returns:
            OpenID: The user information in a standardized format.
        """
        raise NotImplementedError(f"Note yet implemented for Provider {self.provider}.")

    async def get_discovery_document(self) -> DiscoveryDocument:
        """Get document containing handy urls."""
        return {
            "authorization_endpoint": "https://orcid.org/oauth/authorize?response_type=code",
            "token_endpoint": "https://orcid.org/oauth/token",
            "userinfo_endpoint": "https://orcid.org/1/user/-/profile.json",
        }
