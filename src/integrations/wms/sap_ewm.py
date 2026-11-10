"""
SAP Extended Warehouse Management (EWM) connector.

Protocol:  OData v4 (primary) + SAP Event Mesh (event subscription).
Auth:      OAuth 2.0 (SAP BTP XSUAA).
Transport: HTTPS with SAP BTP destination service or direct ERP URL.

Key entities:
  - DeliveryItem       — inbound goods receipt (GR) line
  - StorageBin         — bin/zone location for a handling unit (pallet)
  - HandlingUnit       — the pallet, tagged by SSCC
  - QualityInspection  — EWM QM inspection result (we write here on damage)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class SAPEWMClient:
    """
    OData v4 client for SAP EWM.

    Configuration (from settings / env):
        SAP_EWM_BASE_URL        — e.g. https://<host>/sap/opu/odata4/ewm/
        SAP_EWM_CLIENT_ID
        SAP_EWM_CLIENT_SECRET
        SAP_EWM_TOKEN_URL       — XSUAA token endpoint
        SAP_EWM_WAREHOUSE       — Warehouse number (e.g. "0001")
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        warehouse: str = "0001",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.warehouse = warehouse
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @classmethod
    def from_config(cls, settings) -> "SAPEWMClient":
        return cls(
            base_url=settings.sap_ewm_base_url,
            client_id=settings.sap_ewm_client_id,
            client_secret=settings.sap_ewm_client_secret,
            token_url=settings.sap_ewm_token_url,
            warehouse=getattr(settings, "sap_ewm_warehouse", "0001"),
        )

    def _get_token(self) -> str:
        if self._token and time.monotonic() < self._token_expiry:
            return self._token
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
        self._token = data["access_token"]
        self._token_expiry = time.monotonic() + data.get("expires_in", 3600) - 60
        return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "odata-version": "4.0",
        }

    def get_handling_unit(self, sscc: str) -> dict:
        """
        Fetch handling unit (pallet) details by SSCC.
        Returns bin location, SKU, quantity, and EWM status.
        """
        url = (
            f"{self.base_url}/HandlingUnits"
            f"?$filter=SSCC eq '{sscc}' and WarehouseNumber eq '{self.warehouse}'"
            f"&$expand=StorageBin,Material"
        )
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        value = data.get("value", [])
        return value[0] if value else {}

    def create_quality_inspection(
        self,
        sscc: str,
        defect_code: str,
        severity: str,
        notes: str,
        ai_confidence: float,
    ) -> dict:
        """
        Create a QM inspection result in SAP EWM.

        defect_code: SAP defect catalog code (e.g. "P001" for puncture).
        severity:    SAP usage decision: "A" (accept) | "R" (reject) | "B" (block).
        """
        severity_map = {"none": "A", "minor": "A", "moderate": "B", "severe": "R"}
        usage_decision = severity_map.get(severity.lower(), "B")

        url = f"{self.base_url}/QualityInspections"
        payload = {
            "WarehouseNumber": self.warehouse,
            "SSCC": sscc,
            "DefectCode": defect_code,
            "UsageDecision": usage_decision,
            "InspectionText": notes,
            "AIConfidence": str(round(ai_confidence, 4)),
            "InspectionSource": "MINDWAVE_AI",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("SAP EWM QI created sscc=%s decision=%s", sscc, usage_decision)
        return resp.json()

    def block_handling_unit(self, sscc: str, block_reason: str = "0010") -> dict:
        """
        Block a handling unit in EWM (stock block type for damage).
        block_reason "0010" = quality problem in SAP standard.
        """
        url = f"{self.base_url}/HandlingUnits(SSCC='{sscc}',WarehouseNumber='{self.warehouse}')"
        payload = {
            "BlockReason": block_reason,
            "BlockedForPicking": True,
        }
        resp = httpx.patch(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("HU blocked in SAP EWM sscc=%s", sscc)
        return resp.json()

    def get_delivery_item(self, delivery_number: str, item_number: str = "10") -> dict:
        """Fetch GR delivery item details (expected vs actual quantities)."""
        url = (
            f"{self.base_url}/DeliveryItems"
            f"(DeliveryNumber='{delivery_number}',DeliveryItemNumber='{item_number}')"
        )
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def post_event_mesh_message(
        self,
        event_type: str,
        payload: dict,
        topic: str = "mindwave/damage/detected",
    ) -> bool:
        """
        Publish a message to SAP Event Mesh (BTP).
        Used for real-time event-driven integration with SAP S/4HANA.
        """
        # In production: use SAP Event Mesh REST API
        # Requires separate Event Mesh service credentials
        logger.info("SAP Event Mesh publish type=%s topic=%s", event_type, topic)
        return True  # placeholder
