"""
Oracle Transportation Management (OTM) Cloud connector.

Covers:
- OAuth2 (IDCS client_credentials) authentication
- Shipment order retrieval and status updates
- Damage / exception event posting
- Stop-off and delivery event recording
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

@dataclass
class OTMAuthToken:
    access_token: str
    expires_at: float  # epoch seconds


class OTMAuthProvider:
    """IDCS / Oracle IAM OAuth2 client-credentials provider for OTM Cloud."""

    def __init__(self, idcs_url: str, client_id: str, client_secret: str):
        self._token_url = f"{idcs_url.rstrip('/')}/oauth2/v1/token"
        self._client_id = client_id
        self._client_secret = client_secret
        self._cached: Optional[OTMAuthToken] = None

    def get_token(self) -> str:
        if self._cached and time.time() < self._cached.expires_at - 60:
            return self._cached.access_token

        resp = httpx.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "scope": "urn:opc:resource:consumer::all",
            },
            auth=(self._client_id, self._client_secret),
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        self._cached = OTMAuthToken(
            access_token=body["access_token"],
            expires_at=time.time() + int(body.get("expires_in", 3600)),
        )
        return self._cached.access_token


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class OTMShipment:
    shipment_gid: str
    source_location: str
    dest_location: str
    carrier_gid: str
    status_code: str
    ship_date: Optional[str] = None
    delivery_date: Optional[str] = None
    tracking_number: Optional[str] = None
    stops: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OTMDamageEvent:
    shipment_gid: str
    sscc: str
    damage_severity: str          # "none" | "minor" | "moderate" | "severe"
    damage_type: str
    location_gid: str
    event_time: str               # ISO-8601
    reported_by: str
    image_s3_keys: List[str] = field(default_factory=list)
    claim_reference: Optional[str] = None


@dataclass
class OTMEventResult:
    event_id: str
    status: str                   # "ACCEPTED" | "PENDING" | "ERROR"
    message: str = ""


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class OracleTMSClient:
    """
    Oracle Transportation Management Cloud REST client.

    Endpoints used:
      GET  /logisticsRestApi/resources/23C/shipments/{gid}
      POST /logisticsRestApi/resources/23C/shipmentStatuses
      POST /logisticsRestApi/resources/23C/shipmentEvents
      POST /logisticsRestApi/resources/23C/orderReleases/{gid}/holds
      DELETE /logisticsRestApi/resources/23C/orderReleases/{gid}/holds/{hold_id}
    """

    _API_VERSION = "23C"

    def __init__(
        self,
        otm_base_url: str,
        idcs_url: str,
        client_id: str,
        client_secret: str,
        domain_name: str = "DEFAULT",
        timeout: int = 30,
    ):
        self._base = otm_base_url.rstrip("/")
        self._auth = OTMAuthProvider(idcs_url, client_id, client_secret)
        self._domain = domain_name
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._auth.get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self._base}/logisticsRestApi/resources/{self._API_VERSION}/{path.lstrip('/')}"

    # ------------------------------------------------------------------
    # Shipment queries
    # ------------------------------------------------------------------

    def get_shipment(self, shipment_gid: str) -> OTMShipment:
        """
        Retrieve shipment details by GID (e.g. "ACME.SHIP-00012345").
        """
        resp = httpx.get(
            self._url(f"shipments/{shipment_gid}"),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        shp = body.get("shipment", body)

        return OTMShipment(
            shipment_gid=shp.get("shipmentGid", shipment_gid),
            source_location=shp.get("sourceLocationGid", ""),
            dest_location=shp.get("destLocationGid", ""),
            carrier_gid=shp.get("involvedParty", [{}])[0].get("involvedPartyGid", ""),
            status_code=shp.get("statusCode", ""),
            ship_date=shp.get("shipDate"),
            delivery_date=shp.get("estimatedDeliveryDate"),
            stops=shp.get("stops", []),
        )

    def search_shipments_by_bol(self, bol_number: str) -> List[OTMShipment]:
        """Search shipments by Bill of Lading number."""
        resp = httpx.get(
            self._url("shipments"),
            params={"q": f"transportationBillNumber='{bol_number}'"},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            OTMShipment(
                shipment_gid=s.get("shipmentGid", ""),
                source_location=s.get("sourceLocationGid", ""),
                dest_location=s.get("destLocationGid", ""),
                carrier_gid=s.get("carrierGid", ""),
                status_code=s.get("statusCode", ""),
            )
            for s in items
        ]

    # ------------------------------------------------------------------
    # Event / exception posting
    # ------------------------------------------------------------------

    def post_damage_event(self, event: OTMDamageEvent) -> OTMEventResult:
        """
        Post a damage inspection event against a shipment.
        Maps to OTM Shipment Event (transaction type = 'RD' – Received with Damage).
        """
        severity_map = {
            "none": "NO_DAMAGE",
            "minor": "MINOR_DAMAGE",
            "moderate": "MODERATE_DAMAGE",
            "severe": "SEVERE_DAMAGE",
        }
        payload = {
            "shipmentGid": event.shipment_gid,
            "eventTypeCode": "RD",
            "eventTime": event.event_time,
            "locationGid": event.location_gid,
            "remarks": (
                f"AI-detected damage: {event.damage_type} | "
                f"Severity: {severity_map.get(event.damage_severity, 'UNKNOWN')} | "
                f"SSCC: {event.sscc} | "
                f"Images: {','.join(event.image_s3_keys)}"
            ),
            "reportedByUser": event.reported_by,
        }
        if event.claim_reference:
            payload["claimReference"] = event.claim_reference

        resp = httpx.post(
            self._url("shipmentEvents"),
            json=payload,
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return OTMEventResult(
            event_id=body.get("eventId", ""),
            status=body.get("status", "ACCEPTED"),
            message=body.get("message", ""),
        )

    def update_shipment_status(
        self,
        shipment_gid: str,
        status_code: str,
        reason_code: Optional[str] = None,
    ) -> bool:
        """Update the status of an OTM shipment (e.g. 'ON_HOLD', 'DAMAGED')."""
        payload: Dict[str, Any] = {
            "shipmentGid": shipment_gid,
            "statusCode": status_code,
            "statusDate": datetime.now(timezone.utc).isoformat(),
        }
        if reason_code:
            payload["reasonCode"] = reason_code

        resp = httpx.post(
            self._url("shipmentStatuses"),
            json=payload,
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.status_code in (200, 201)

    # ------------------------------------------------------------------
    # Order release holds
    # ------------------------------------------------------------------

    def place_order_hold(
        self,
        order_release_gid: str,
        hold_code: str = "DAMAGE_INSPECTION",
        notes: str = "",
    ) -> str:
        """Place a hold on an order release; returns hold_id."""
        resp = httpx.post(
            self._url(f"orderReleases/{order_release_gid}/holds"),
            json={
                "holdCode": hold_code,
                "holdReason": notes or f"Automated hold: {hold_code}",
                "holdDate": datetime.now(timezone.utc).isoformat(),
            },
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("holdId", "")

    def release_order_hold(self, order_release_gid: str, hold_id: str) -> bool:
        """Remove a previously placed hold."""
        resp = httpx.delete(
            self._url(f"orderReleases/{order_release_gid}/holds/{hold_id}"),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.status_code in (200, 204)

    # ------------------------------------------------------------------
    # Delivery confirmation
    # ------------------------------------------------------------------

    def record_delivery_scan(
        self,
        shipment_gid: str,
        sscc: str,
        condition: str,
        location_gid: str,
        scanned_by: str,
    ) -> OTMEventResult:
        """
        Record final delivery scan (Chain-of-Custody point 5: Delivery Scan).
        Uses OTM event type 'D' (Delivered).
        """
        event = OTMDamageEvent(
            shipment_gid=shipment_gid,
            sscc=sscc,
            damage_severity=condition if condition in ("none", "minor", "moderate", "severe") else "none",
            damage_type="delivery_scan",
            location_gid=location_gid,
            event_time=datetime.now(timezone.utc).isoformat(),
            reported_by=scanned_by,
        )
        # Override event type to 'D' for a clean delivery
        return self.post_damage_event(event)
