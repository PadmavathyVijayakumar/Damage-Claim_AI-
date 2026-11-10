"""
SAP S/4HANA ERP connector.

Protocol: OData v4 (SAP Business Accelerator Hub APIs).
Auth:     OAuth 2.0 (SAP BTP XSUAA).

Key operations:
  - create_quality_notification()  — QM notification for inbound damage
  - post_goods_receipt()           — MIGO-equivalent GR posting
  - create_purchase_order_hold()   — block PO line for quality
  - get_vendor_details()           — vendor master for dispute letters
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class SAPS4Client:
    """
    SAP S/4HANA OData v4 client for damage claim integration.

    Configuration:
        SAP_S4_BASE_URL        — e.g. https://<host>/sap/opu/odata4/sap/
        SAP_S4_CLIENT_ID
        SAP_S4_CLIENT_SECRET
        SAP_S4_TOKEN_URL
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        plant: str = "1000",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.plant = plant
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @classmethod
    def from_config(cls, settings) -> "SAPS4Client":
        return cls(
            base_url=settings.sap_s4_host,
            client_id=settings.sap_s4_client_id,
            client_secret=settings.sap_s4_client_secret,
            token_url=settings.sap_s4_token_url,
            plant=getattr(settings, "sap_s4_plant", "1000"),
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

    def create_quality_notification(
        self,
        material: str,
        vendor: str,
        purchase_order: str,
        defect_description: str,
        defect_code: str = "0001",
        quantity_defective: float = 1.0,
    ) -> dict:
        """
        Create a QM quality notification in SAP S/4HANA.
        Notification type Q2 (complaint to vendor).
        """
        url = f"{self.base_url}/api_qualitynotification/v1/QualityNotification"
        payload = {
            "NotificationType": "Q2",
            "Material": material,
            "Vendor": vendor,
            "PurchaseOrder": purchase_order,
            "Plant": self.plant,
            "DefectDescription": defect_description,
            "DefectCode": defect_code,
            "DefectiveQty": str(quantity_defective),
            "DefectiveQtyUnit": "EA",
            "CreatedBySystem": "MINDWAVE_AI",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        notification_id = resp.json().get("QualityNotification")
        logger.info("SAP S4 QN created %s material=%s", notification_id, material)
        return resp.json()

    def block_purchase_order_item(
        self,
        purchase_order: str,
        item: str = "00010",
        block_reason: str = "09",   # "09" = quality hold in SAP standard
    ) -> dict:
        """
        Set a delivery block on a PO item — prevents GR posting for the blocked item.
        """
        url = (
            f"{self.base_url}/api_purchaseorder_process/v1"
            f"/PurchaseOrder('{purchase_order}')/to_PurchaseOrderItem(PurchaseOrderItem='{item}')"
        )
        payload = {"DeliveryBlockReason": block_reason}
        resp = httpx.patch(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("PO item blocked PO=%s item=%s", purchase_order, item)
        return resp.json()

    def get_vendor_details(self, vendor_id: str) -> dict:
        """Fetch vendor master data for dispute letters."""
        url = (
            f"{self.base_url}/api_business_partner/v1"
            f"/A_BusinessPartner('{vendor_id}')"
            f"?$select=BusinessPartnerFullName,EmailAddress,BusinessPartnerAddress"
        )
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def post_material_document(
        self,
        material: str,
        quantity: float,
        movement_type: str = "101",   # 101 = GR for PO
        purchase_order: str = "",
        storage_location: str = "0001",
    ) -> dict:
        """
        Post a material document (goods movement) in SAP S/4HANA.
        Used to record actual received quantity after damage adjustment.
        """
        url = f"{self.base_url}/api_material_document/v1/MaterialDocumentHeader"
        payload = {
            "GoodsMovementCode": movement_type,
            "to_MaterialDocumentItem": [
                {
                    "Material": material,
                    "Plant": self.plant,
                    "StorageLocation": storage_location,
                    "QuantityInEntryUnit": str(quantity),
                    "EntryUnit": "EA",
                    "PurchaseOrder": purchase_order,
                    "GoodsMovementType": movement_type,
                }
            ],
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
