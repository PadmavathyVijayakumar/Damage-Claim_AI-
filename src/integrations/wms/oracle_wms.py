"""
Oracle WMS Cloud connector.

Protocol: REST API + Oracle Integration Cloud (OIC) for event-driven flows.
Auth:     OAuth 2.0 (Oracle IDCS).

Key operations:
  - get_lpn_details()         — LPN (licence plate number = pallet) details
  - create_damage_lpn_hold()  — place LPN on damage hold
  - attach_receipt_note()     — write inspection note to inbound receipt
  - trigger_oic_event()       — fire an OIC integration to downstream ERP
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OracleWMSClient:
    """
    Oracle WMS Cloud REST client.

    Configuration:
        ORACLE_WMS_BASE_URL       — e.g. https://<host>/wms/lgfapi/v10
        ORACLE_WMS_CLIENT_ID
        ORACLE_WMS_CLIENT_SECRET
        ORACLE_WMS_TOKEN_URL      — IDCS token endpoint
        ORACLE_WMS_FACILITY       — Facility code (e.g. "DC01")
        ORACLE_WMS_COMPANY        — Company code
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        facility: str = "DC01",
        company: str = "ACME",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.facility = facility
        self.company = company
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @classmethod
    def from_config(cls, settings) -> "OracleWMSClient":
        return cls(
            base_url=settings.oracle_wms_base_url,
            client_id=settings.oracle_wms_client_id,
            client_secret=settings.oracle_wms_client_secret,
            token_url=settings.oracle_wms_token_url,
            facility=getattr(settings, "oracle_wms_facility", "DC01"),
            company=getattr(settings, "oracle_wms_company", "ACME"),
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
                "scope": "wms_api",
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
            "facility": self.facility,
            "company": self.company,
        }

    def get_lpn_details(self, lpn: str) -> dict:
        """
        Fetch LPN details (Oracle's equivalent of SSCC/pallet ID).
        Returns location, item details, and status.
        """
        url = f"{self.base_url}/lpn/{lpn}"
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create_damage_lpn_hold(
        self,
        lpn: str,
        damage_code: str = "DMG",
        notes: str = "",
    ) -> dict:
        """
        Place an LPN on damage hold — prevents picking and shipping.
        """
        url = f"{self.base_url}/lpn/{lpn}/holds"
        payload = {
            "hold_code": damage_code,
            "hold_comment": notes or "AI damage detection flagged for hold.",
            "created_by": "MINDWAVE_AI",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Oracle WMS hold created lpn=%s", lpn)
        return resp.json()

    def attach_receipt_note(
        self,
        receipt_number: str,
        lpn: str,
        severity: str,
        trigger_claim: bool,
        evidence_s3_key: str = "",
    ) -> dict:
        """
        Attach an inspection note to an inbound receipt.
        """
        url = f"{self.base_url}/receipts/{receipt_number}/notes"
        payload = {
            "note_type": "AI_INSPECTION",
            "lpn": lpn,
            "severity": severity.upper(),
            "trigger_claim": trigger_claim,
            "evidence_reference": evidence_s3_key,
            "source_system": "MINDWAVE_AI",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Receipt note attached receipt=%s lpn=%s", receipt_number, lpn)
        return resp.json()

    def trigger_oic_event(self, event_type: str, payload: dict) -> dict:
        """
        Trigger an Oracle Integration Cloud (OIC) integration flow.
        Used to cascade damage events to Oracle ERP / TMS.
        """
        # OIC REST trigger endpoint format:
        # POST /ic/api/integration/v1/flows/rest/<INTEGRATION_ID>/<VERSION>/trigger
        oic_url = f"{self.base_url}/oic/trigger/{event_type}"
        resp = httpx.post(
            oic_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        logger.info("OIC event triggered type=%s", event_type)
        return resp.json()
