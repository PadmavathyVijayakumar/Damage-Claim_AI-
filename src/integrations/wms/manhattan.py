"""
Manhattan Active WMS connector.

Authentication: OAuth 2.0 client_credentials flow.
API: REST (receiving records, PO lookup) + GraphQL (shipment manifests).
Event subscription: Manhattan Webhook API → Kafka topic.

Key operations:
  - get_po_details()         — fetch PO line items and expected quantities
  - attach_inspection()      — attach inspection result to receiving record
  - create_damage_hold()     — flag a pallet for damage hold (HOLD_DAMAGE location)
  - get_shipment_manifest()  — GraphQL query for outbound pallet details
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

TOKEN_EXPIRY_BUFFER_S = 60   # refresh token 60 seconds before expiry


class ManhattanWMSClient:
    """
    REST + GraphQL client for Manhattan Active WMS.

    Configuration (from settings / env):
        MANHATTAN_BASE_URL      — e.g. https://api.manh.com/wms/v2
        MANHATTAN_CLIENT_ID
        MANHATTAN_CLIENT_SECRET
        MANHATTAN_TOKEN_URL     — OAuth2 token endpoint
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.timeout = timeout
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    @classmethod
    def from_config(cls, settings) -> "ManhattanWMSClient":
        return cls(
            base_url=settings.manhattan_base_url,
            client_id=settings.manhattan_client_id,
            client_secret=settings.manhattan_client_secret,
            token_url=settings.manhattan_token_url,
        )

    def _get_token(self) -> str:
        """Return cached token or refresh via OAuth2 client_credentials."""
        if self._access_token and time.monotonic() < self._token_expiry:
            return self._access_token

        resp = httpx.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.monotonic() + data.get("expires_in", 3600) - TOKEN_EXPIRY_BUFFER_S
        logger.debug("Manhattan token refreshed")
        return self._access_token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ── REST endpoints ──────────────────────────────────────────────────

    def get_po_details(self, po_number: str) -> dict:
        """Fetch PO details including line items and expected quantities."""
        url = f"{self.base_url}/purchase-orders/{po_number}"
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def attach_inspection(
        self,
        receiving_record_id: str,
        pallet_sscc: str,
        inspection_payload: dict,
    ) -> dict:
        """
        Attach AI inspection result to a receiving record.

        inspection_payload should match to_wms_payload() from DockCameraPipeline.
        """
        url = f"{self.base_url}/receiving-records/{receiving_record_id}/inspections"
        payload = {
            "pallet_sscc": pallet_sscc,
            "inspection_source": "mindwave_ai",
            **inspection_payload,
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Inspection attached to WMS record %s", receiving_record_id)
        return resp.json()

    def create_damage_hold(
        self,
        pallet_sscc: str,
        reason_code: str = "DAMAGE_AI",
        notes: str = "",
    ) -> dict:
        """Move pallet to HOLD_DAMAGE location in WMS."""
        url = f"{self.base_url}/pallets/{pallet_sscc}/holds"
        payload = {
            "hold_type": "DAMAGE",
            "reason_code": reason_code,
            "notes": notes or "AI damage detection flagged this pallet for hold.",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Damage hold created for pallet %s", pallet_sscc)
        return resp.json()

    def release_hold(self, pallet_sscc: str) -> dict:
        """Release a damage hold (after claim resolution)."""
        url = f"{self.base_url}/pallets/{pallet_sscc}/holds"
        resp = httpx.delete(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── GraphQL endpoint ────────────────────────────────────────────────

    def get_shipment_manifest(self, shipment_id: str) -> dict:
        """
        Fetch outbound shipment manifest via GraphQL.
        Returns list of pallets with SSCC, SKU, and load sequence.
        """
        url = f"{self.base_url}/graphql"
        query = """
        query GetShipmentManifest($shipmentId: ID!) {
          shipment(id: $shipmentId) {
            id
            bolNumber
            carrier { scac name }
            pallets {
              sscc
              sku
              quantity
              loadSequence
              sealNumber
            }
          }
        }
        """
        resp = httpx.post(
            url,
            headers=self._headers(),
            json={"query": query, "variables": {"shipmentId": shipment_id}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        return data["data"]["shipment"]
