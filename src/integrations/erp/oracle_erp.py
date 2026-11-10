"""
Oracle Fusion ERP connector.

Protocol: REST (Oracle Fusion Applications REST APIs v1).
Auth:     OAuth 2.0 (Oracle IDCS / OCI IAM).

Key operations:
  - create_supplier_quality_incident()  — record inbound damage
  - create_ap_debit_memo()             — debit memo against supplier invoice
  - get_purchase_order()               — fetch PO details for validation
  - create_receiving_exception()       — log receiving exception for damage
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OracleFusionERPClient:
    """
    Oracle Fusion ERP REST client.

    Configuration:
        ORACLE_ERP_HOST           — e.g. https://<host>.oraclecloud.com
        ORACLE_ERP_CLIENT_ID
        ORACLE_ERP_CLIENT_SECRET
        ORACLE_ERP_TOKEN_URL
    """

    def __init__(
        self,
        host: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @classmethod
    def from_config(cls, settings) -> "OracleFusionERPClient":
        return cls(
            host=settings.oracle_erp_host,
            client_id=settings.oracle_erp_client_id,
            client_secret=settings.oracle_erp_client_secret,
            token_url=settings.oracle_erp_token_url,
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
        }

    def get_purchase_order(self, po_number: str) -> dict:
        """Fetch PO header and lines from Oracle Fusion Procurement."""
        url = (
            f"{self.host}/fscmRestApi/resources/11.13.18.05"
            f"/purchaseOrders?q=OrderNumber={po_number}"
            f"&fields=OrderNumber,Supplier,Lines,Status"
        )
        resp = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return items[0] if items else {}

    def create_receiving_exception(
        self,
        po_number: str,
        item_number: str,
        quantity: float,
        exception_code: str = "DAMAGE",
        notes: str = "",
    ) -> dict:
        """
        Log a receiving exception in Oracle Fusion Receiving.
        Used when physical damage is detected at inbound.
        """
        url = f"{self.host}/fscmRestApi/resources/11.13.18.05/receivingTransactions"
        payload = {
            "PONumber": po_number,
            "POLineNumber": item_number,
            "TransactionType": "EXCEPTION",
            "ExceptionCode": exception_code,
            "Quantity": quantity,
            "Comments": notes or "Mindwave AI detected physical damage.",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Oracle receiving exception created PO=%s", po_number)
        return resp.json()

    def create_supplier_quality_incident(
        self,
        supplier_id: str,
        material_code: str,
        po_number: str,
        defect_description: str,
        severity: str,
        evidence_url: str = "",
    ) -> dict:
        """
        Create a supplier quality incident (SQM) in Oracle Fusion Quality.
        """
        url = f"{self.host}/fscmRestApi/resources/11.13.18.05/qualityIncidents"
        payload = {
            "SupplierId": supplier_id,
            "ItemNumber": material_code,
            "PurchaseOrderNumber": po_number,
            "Description": defect_description,
            "Severity": severity.upper(),
            "Source": "MINDWAVE_AI",
            "EvidenceURL": evidence_url,
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        incident_id = resp.json().get("IncidentId")
        logger.info("Oracle quality incident created %s", incident_id)
        return resp.json()

    def create_ap_debit_memo(
        self,
        supplier_id: str,
        invoice_number: str,
        amount: float,
        description: str,
        currency: str = "USD",
    ) -> dict:
        """
        Create an AP debit memo to recover damage costs from the supplier.
        """
        url = f"{self.host}/fscmRestApi/resources/11.13.18.05/invoices"
        payload = {
            "InvoiceNumber": f"DM-{invoice_number}",
            "InvoiceType": "DEBIT_MEMO",
            "SupplierId": supplier_id,
            "InvoiceAmount": amount,
            "InvoiceCurrencyCode": currency,
            "Description": description,
            "Source": "MINDWAVE_AI_DAMAGE_CLAIM",
        }
        resp = httpx.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Oracle AP debit memo created supplier=%s amount=%.2f", supplier_id, amount)
        return resp.json()
