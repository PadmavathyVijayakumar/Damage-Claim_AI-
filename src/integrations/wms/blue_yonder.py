"""
Blue Yonder (JDA) Luminate WMS connector.

Supports both:
  - REST API (Luminate Platform — modern cloud deployments)
  - SOAP legacy (JDA WMS 2020 on-premise)

Auth: API Key (REST) or WS-Security UsernameToken (SOAP).
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class BlueYonderRESTClient:
    """
    Blue Yonder Luminate Platform REST client.

    Configuration:
        BY_BASE_URL       — e.g. https://<tenant>.luminate.jda.com/api/v1
        BY_API_KEY        — Luminate API key
        BY_TENANT_ID      — Tenant identifier
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        tenant_id: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout

    @classmethod
    def from_config(cls, settings) -> "BlueYonderRESTClient":
        return cls(
            base_url=settings.by_base_url,
            api_key=settings.by_api_key,
            tenant_id=settings.by_tenant_id,
        )

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "x-tenant-id": self.tenant_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def get_receipt(self, receipt_id: str) -> dict:
        """Fetch GR receipt details."""
        url = f"{self.base_url}/receipts/{receipt_id}"
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def attach_damage_note(
        self,
        receipt_id: str,
        sscc: str,
        severity: str,
        trigger_claim: bool,
        ai_inference_ms: float,
    ) -> dict:
        """Attach AI damage note to a receipt line."""
        url = f"{self.base_url}/receipts/{receipt_id}/damage-notes"
        payload = {
            "sscc": sscc,
            "severity": severity,
            "triggerClaim": trigger_claim,
            "source": "mindwave_ai",
            "aiInferenceMs": ai_inference_ms,
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("BY damage note attached receipt=%s sscc=%s", receipt_id, sscc)
        return resp.json()

    def get_pallet_status(self, sscc: str) -> dict:
        """Get current pallet location and status."""
        url = f"{self.base_url}/pallets/{sscc}"
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class BlueYonderSOAPClient:
    """
    SOAP client for legacy JDA WMS on-premise deployments.

    Uses WS-Security UsernameToken header.
    Only implements the minimum set of operations needed for damage claim integration.
    """

    SOAP_NS = "http://jda.com/wms/services"

    def __init__(
        self,
        wsdl_url: str,
        username: str,
        password: str,
        timeout: float = 30.0,
    ) -> None:
        self.wsdl_url = wsdl_url
        self.username = username
        self.password = password
        self.timeout = timeout

    @classmethod
    def from_config(cls, settings) -> "BlueYonderSOAPClient":
        return cls(
            wsdl_url=settings.by_soap_wsdl_url,
            username=settings.by_soap_username,
            password=settings.by_soap_password,
        )

    def _ws_security_header(self) -> str:
        return f"""
        <wsse:Security xmlns:wsse="http://schemas.xmlsoap.org/ws/2002/12/secext">
          <wsse:UsernameToken>
            <wsse:Username>{self.username}</wsse:Username>
            <wsse:Password>{self.password}</wsse:Password>
          </wsse:UsernameToken>
        </wsse:Security>"""

    def call(self, operation: str, body_xml: str) -> str:
        """Make a raw SOAP call; returns response XML string."""
        envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope
    xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
    xmlns:jda="{self.SOAP_NS}">
  <soapenv:Header>
    {self._ws_security_header()}
  </soapenv:Header>
  <soapenv:Body>
    <jda:{operation}>
      {body_xml}
    </jda:{operation}>
  </soapenv:Body>
</soapenv:Envelope>"""

        resp = httpx.post(
            self.wsdl_url,
            content=envelope.encode("utf-8"),
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f"{self.SOAP_NS}/{operation}",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.text

    def create_hold(self, sscc: str, reason: str) -> str:
        """Create a damage hold via SOAP (legacy on-premise)."""
        body = f"<sscc>{sscc}</sscc><holdReason>{reason}</holdReason>"
        return self.call("CreateDamageHold", body)
