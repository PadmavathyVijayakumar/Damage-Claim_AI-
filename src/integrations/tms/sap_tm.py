"""
SAP Transportation Management (SAP TM) connector.

Covers:
- XSUAA OAuth2 client-credentials authentication
- OData v4 freight order / delivery order retrieval
- Damage notification creation via SAP TM Business Object layer
- Event messaging via SAP Event Mesh (REST)
- Freight unit blocking / unblocking for damage holds
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
# XSUAA auth (same pattern as SAP EWM / S/4HANA clients)
# ---------------------------------------------------------------------------

@dataclass
class XSUAAToken:
    access_token: str
    expires_at: float


class SAPTMAuthProvider:
    """SAP XSUAA OAuth2 client-credentials provider for SAP TM on BTP."""

    def __init__(self, xsuaa_url: str, client_id: str, client_secret: str):
        self._token_url = f"{xsuaa_url.rstrip('/')}/oauth/token"
        self._client_id = client_id
        self._client_secret = client_secret
        self._cached: Optional[XSUAAToken] = None

    def get_token(self) -> str:
        if self._cached and time.time() < self._cached.expires_at - 60:
            return self._cached.access_token

        resp = httpx.post(
            self._token_url,
            data={"grant_type": "client_credentials"},
            auth=(self._client_id, self._client_secret),
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        self._cached = XSUAAToken(
            access_token=body["access_token"],
            expires_at=time.time() + int(body.get("expires_in", 3600)),
        )
        return self._cached.access_token


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SAPTMFreightOrder:
    freight_order_id: str
    carrier_id: str
    shipper_id: str
    consignee_id: str
    status: str
    source_location: str
    dest_location: str
    planned_departure: Optional[str] = None
    actual_departure: Optional[str] = None
    planned_arrival: Optional[str] = None
    actual_arrival: Optional[str] = None
    freight_units: List[str] = field(default_factory=list)   # FreightUnit IDs


@dataclass
class SAPTMDamageNotification:
    freight_order_id: str
    freight_unit_id: str
    sscc: str
    damage_severity: str          # none / minor / moderate / severe
    damage_description: str
    detected_at_location: str
    detected_at_time: str         # ISO-8601
    detected_by: str
    claim_id: Optional[str] = None
    image_references: List[str] = field(default_factory=list)


@dataclass
class SAPTMNotificationResult:
    notification_id: str
    status: str                   # "CREATED" | "ERROR"
    message: str = ""


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class SAPTMClient:
    """
    SAP Transportation Management OData v4 client (BTP / Cloud-hosted TM).

    Key OData services used:
      /sap/tm/api/v1/FreightOrders
      /sap/tm/api/v1/FreightUnits
      /sap/tm/api/v1/DamageNotifications    (custom or extensibility)
      /sap/tm/api/v1/EventMesh/publish      (SAP Event Mesh)

    Freight Blocking uses the SAP TM "PlanningBlock" action on FreightUnit.
    """

    def __init__(
        self,
        tm_base_url: str,
        xsuaa_url: str,
        client_id: str,
        client_secret: str,
        event_mesh_namespace: str = "",
        timeout: int = 30,
    ):
        self._base = tm_base_url.rstrip("/")
        self._auth = SAPTMAuthProvider(xsuaa_url, client_id, client_secret)
        self._em_namespace = event_mesh_namespace
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self, with_csrf: bool = False) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Authorization": f"Bearer {self._auth.get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if with_csrf:
            h["x-csrf-token"] = self._fetch_csrf()
        return h

    def _fetch_csrf(self) -> str:
        resp = httpx.get(
            f"{self._base}/sap/tm/api/v1/",
            headers={
                "Authorization": f"Bearer {self._auth.get_token()}",
                "x-csrf-token": "Fetch",
            },
            timeout=10,
        )
        return resp.headers.get("x-csrf-token", "")

    def _odata_url(self, entity: str) -> str:
        return f"{self._base}/sap/tm/api/v1/{entity}"

    # ------------------------------------------------------------------
    # Freight order retrieval
    # ------------------------------------------------------------------

    def get_freight_order(self, freight_order_id: str) -> SAPTMFreightOrder:
        """Retrieve a Freight Order by its SAP TM internal ID."""
        resp = httpx.get(
            self._odata_url(f"FreightOrders('{freight_order_id}')"),
            params={"$expand": "FreightUnits"},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        d = resp.json().get("value", resp.json())
        fu_ids = [fu.get("FreightUnitId", "") for fu in d.get("FreightUnits", [])]

        return SAPTMFreightOrder(
            freight_order_id=d.get("FreightOrderId", freight_order_id),
            carrier_id=d.get("CarrierId", ""),
            shipper_id=d.get("ShipperId", ""),
            consignee_id=d.get("ConsigneeId", ""),
            status=d.get("LifecycleStatusCode", ""),
            source_location=d.get("SourceLocationId", ""),
            dest_location=d.get("DestinationLocationId", ""),
            planned_departure=d.get("PlannedDepartureDateTime"),
            actual_departure=d.get("ActualDepartureDateTime"),
            planned_arrival=d.get("PlannedArrivalDateTime"),
            actual_arrival=d.get("ActualArrivalDateTime"),
            freight_units=fu_ids,
        )

    def find_freight_order_by_sscc(self, sscc: str) -> Optional[SAPTMFreightOrder]:
        """Find the freight order containing a given SSCC/handling unit."""
        resp = httpx.get(
            self._odata_url("FreightOrders"),
            params={
                "$filter": f"contains(HandlingUnitIds, '{sscc}')",
                "$top": "1",
            },
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        items = resp.json().get("value", [])
        if not items:
            return None
        return self.get_freight_order(items[0]["FreightOrderId"])

    # ------------------------------------------------------------------
    # Damage notifications
    # ------------------------------------------------------------------

    def create_damage_notification(
        self, notif: SAPTMDamageNotification
    ) -> SAPTMNotificationResult:
        """
        Create a damage notification against a freight unit.

        Severity mapping to SAP TM Damage Category:
          none     → "1" (No Damage)
          minor    → "2" (Minor Damage)
          moderate → "3" (Moderate Damage)
          severe   → "4" (Total Loss)
        """
        severity_code_map = {
            "none": "1",
            "minor": "2",
            "moderate": "3",
            "severe": "4",
        }
        payload = {
            "FreightOrderId": notif.freight_order_id,
            "FreightUnitId": notif.freight_unit_id,
            "HandlingUnitId": notif.sscc,
            "DamageCategoryCode": severity_code_map.get(notif.damage_severity, "2"),
            "DamageDescription": notif.damage_description,
            "LocationId": notif.detected_at_location,
            "DetectionDateTime": notif.detected_at_time,
            "DetectedBy": notif.detected_by,
            "ExternalReferenceId": notif.claim_id or "",
            "ImageReferences": ",".join(notif.image_references),
        }

        resp = httpx.post(
            self._odata_url("DamageNotifications"),
            json=payload,
            headers=self._headers(with_csrf=True),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return SAPTMNotificationResult(
            notification_id=body.get("DamageNotificationId", ""),
            status="CREATED" if resp.status_code in (200, 201) else "ERROR",
        )

    # ------------------------------------------------------------------
    # Freight unit blocking
    # ------------------------------------------------------------------

    def block_freight_unit(
        self,
        freight_unit_id: str,
        blocking_reason: str = "DAMAGE_INSPECTION",
    ) -> bool:
        """
        Set a planning block on a Freight Unit to prevent re-assignment.
        Uses SAP TM PlanningBlock action.
        """
        resp = httpx.post(
            self._odata_url(f"FreightUnits('{freight_unit_id}')/PlanningBlock"),
            json={"BlockingReason": blocking_reason},
            headers=self._headers(with_csrf=True),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.status_code in (200, 204)

    def unblock_freight_unit(self, freight_unit_id: str) -> bool:
        """Remove planning block from a Freight Unit."""
        resp = httpx.post(
            self._odata_url(f"FreightUnits('{freight_unit_id}')/RemovePlanningBlock"),
            json={},
            headers=self._headers(with_csrf=True),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.status_code in (200, 204)

    # ------------------------------------------------------------------
    # SAP Event Mesh publishing
    # ------------------------------------------------------------------

    def publish_damage_event(
        self,
        freight_order_id: str,
        sscc: str,
        damage_severity: str,
        claim_id: Optional[str] = None,
    ) -> bool:
        """
        Publish a damage event to SAP Event Mesh so downstream systems
        (S/4HANA, Ariba, etc.) can react in near-real-time.
        """
        if not self._em_namespace:
            logger.warning("Event Mesh namespace not configured; skipping publish")
            return False

        event_payload: Dict[str, Any] = {
            "specversion": "1.0",
            "type": "sap.tm.freight.damage.detected.v1",
            "source": f"/damage-claim-ai/{self._em_namespace}",
            "id": f"{freight_order_id}-{sscc}-{int(time.time())}",
            "time": datetime.now(timezone.utc).isoformat(),
            "datacontenttype": "application/json",
            "data": {
                "freightOrderId": freight_order_id,
                "sscc": sscc,
                "damageSeverity": damage_severity,
                "claimId": claim_id,
            },
        }

        resp = httpx.post(
            f"{self._base}/messagingrest/v1/topics/{self._em_namespace}/messages",
            json=event_payload,
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.status_code in (200, 202)
