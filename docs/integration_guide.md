# Integration Guide — WMS / TMS / ERP Connectors

This guide explains how to configure and use each enterprise connector.
All credentials must be stored in `config/.env` (never committed to source control).

---

## Table of Contents

1. [Manhattan Active WM (WMS)](#1-manhattan-active-wm)
2. [SAP EWM (WMS)](#2-sap-ewm)
3. [Blue Yonder / JDA (WMS)](#3-blue-yonder--jda)
4. [Oracle WMS Cloud](#4-oracle-wms-cloud)
5. [SAP S/4HANA (ERP)](#5-sap-s4hana)
6. [Oracle Fusion ERP](#6-oracle-fusion-erp)
7. [Oracle Transportation Management (TMS)](#7-oracle-transportation-management)
8. [SAP Transportation Management (TMS)](#8-sap-transportation-management)
9. [Carrier EDI (214 / 210)](#9-carrier-edi-214--210)

---

## 1. Manhattan Active WM

**File:** `src/integrations/wms/manhattan.py`

### Authentication
- OAuth2 `client_credentials` grant
- Token endpoint: `https://<tenant>.manh.com/oauth2/token`

### Environment variables
```dotenv
MANHATTAN_BASE_URL=https://<tenant>.manh.com/api/v1
MANHATTAN_OAUTH_URL=https://<tenant>.manh.com/oauth2/token
MANHATTAN_CLIENT_ID=<client_id>
MANHATTAN_CLIENT_SECRET=<secret>
MANHATTAN_SITE_ID=DC01
```

### Key operations
| Method | Description |
|--------|-------------|
| `get_po_details(po_number)` | Retrieve purchase order header + lines |
| `get_shipment_manifest(sscc)` | GraphQL query for pallet manifest |
| `attach_inspection(sscc, payload)` | Attach AI inspection result to receipt line |
| `create_damage_hold(sscc, severity, reason)` | Place damage hold on SSCC |
| `release_hold(sscc, hold_id)` | Release hold after inspection resolution |

### Sample usage
```python
from src.integrations.wms.manhattan import ManhattanWMSClient

client = ManhattanWMSClient(
    base_url=settings.MANHATTAN_BASE_URL,
    oauth_url=settings.MANHATTAN_OAUTH_URL,
    client_id=settings.MANHATTAN_CLIENT_ID,
    client_secret=settings.MANHATTAN_CLIENT_SECRET,
    site_id=settings.MANHATTAN_SITE_ID,
)

# After dock inspection
client.attach_inspection(sscc="003600000012345678901", payload=pipeline_result.to_wms_payload())
client.create_damage_hold(sscc="003600000012345678901", severity="moderate", reason="AI_DETECTED")
```

---

## 2. SAP EWM

**File:** `src/integrations/wms/sap_ewm.py`

### Authentication
- XSUAA OAuth2 `client_credentials`
- Token endpoint: `https://<subaccount>.authentication.eu10.hana.ondemand.com/oauth/token`

### Environment variables
```dotenv
SAP_EWM_BASE_URL=https://<sid>-api.cfapps.eu10.hana.ondemand.com
SAP_EWM_XSUAA_URL=https://<subaccount>.authentication.eu10.hana.ondemand.com
SAP_EWM_CLIENT_ID=<client_id>
SAP_EWM_CLIENT_SECRET=<secret>
SAP_EWM_WAREHOUSE_NUMBER=1710
```

### Key operations
| Method | Description |
|--------|-------------|
| `get_handling_unit(hu_number)` | Get HU (pallet) details including product and quantity |
| `create_quality_inspection(hu, severity, findings)` | Create QI task; maps severity to SAP usage decision (A/B/R) |
| `block_handling_unit(hu, reason)` | Block HU for movement |
| `post_event_mesh_message(event)` | Publish event to SAP Integration Suite / Event Mesh |

### Severity → Usage Decision mapping
| Severity | SAP Usage Decision |
|----------|--------------------|
| none | A (Accept) |
| minor | A (Accept with note) |
| moderate | B (Partial Accept) |
| severe | R (Reject) |

---

## 3. Blue Yonder / JDA

**File:** `src/integrations/wms/blue_yonder.py`

Supports both cloud (REST + API key) and on-premise (SOAP + WS-Security).

### REST (Cloud)
```dotenv
BLUE_YONDER_BASE_URL=https://<tenant>.api.blueyonder.cloud/api/v1
BLUE_YONDER_API_KEY=<api_key>
```

### SOAP (On-Premise / Legacy JDA)
```dotenv
BLUE_YONDER_SOAP_WSDL=https://<server>/jdaservice/services/InboundReceiving?wsdl
BLUE_YONDER_SOAP_USERNAME=<username>
BLUE_YONDER_SOAP_PASSWORD=<password>
```

### Key operations
| Method | Description |
|--------|-------------|
| `get_lpn_details(lpn)` | Retrieve LPN contents (REST) |
| `create_damage_lpn(lpn, severity, reason)` | Create damage LPN record (REST) |
| `release_lpn(lpn)` | Release LPN from hold (REST) |
| `soap_attach_receipt_note(asn, note, severity)` | Attach receipt note via SOAP (on-premise) |

---

## 4. Oracle WMS Cloud

**File:** `src/integrations/wms/oracle_wms.py`

### Authentication
- Oracle IDCS OAuth2 `client_credentials`

### Environment variables
```dotenv
ORACLE_WMS_BASE_URL=https://<tenancy>.wms.ocs.oraclecloud.com
ORACLE_WMS_IDCS_URL=https://<tenancy>.identity.oraclecloud.com
ORACLE_WMS_CLIENT_ID=<client_id>
ORACLE_WMS_CLIENT_SECRET=<secret>
ORACLE_WMS_OIC_INSTANCE_URL=https://<tenancy>.integration.ocp.oraclecloud.com
```

### Key operations
| Method | Description |
|--------|-------------|
| `get_lpn_details(lpn)` | LPN contents and location |
| `create_damage_lpn_hold(lpn, severity, notes)` | Damage hold with reason code |
| `attach_receipt_note(receipt, sscc, note, severity)` | Attach inspection note to receipt |
| `trigger_oic_event(event_name, payload)` | Trigger Oracle Integration Cloud workflow |

---

## 5. SAP S/4HANA

**File:** `src/integrations/erp/sap_s4.py`

### Authentication
- XSUAA OAuth2 (same provider as SAP EWM)

### Environment variables
```dotenv
SAP_S4_BASE_URL=https://<sid>-api.cfapps.eu10.hana.ondemand.com
SAP_S4_XSUAA_URL=https://<subaccount>.authentication.eu10.hana.ondemand.com
SAP_S4_CLIENT_ID=<client_id>
SAP_S4_CLIENT_SECRET=<secret>
```

### Key operations
| Method | Description |
|--------|-------------|
| `create_quality_notification(po_item, severity, description)` | Q2-type QN with damage findings |
| `block_purchase_order_item(po, item, reason)` | Block PO item for delivery |
| `get_vendor_details(vendor_id)` | Vendor master data for scorecard |
| `post_material_document(movement_type, plant, material, qty)` | Post goods movement (e.g., scrapping) |

---

## 6. Oracle Fusion ERP

**File:** `src/integrations/erp/oracle_erp.py`

### Environment variables
```dotenv
ORACLE_ERP_BASE_URL=https://<tenancy>.fa.ocs.oraclecloud.com
ORACLE_ERP_IDCS_URL=https://<tenancy>.identity.oraclecloud.com
ORACLE_ERP_CLIENT_ID=<client_id>
ORACLE_ERP_CLIENT_SECRET=<secret>
```

### Key operations
| Method | Description |
|--------|-------------|
| `create_receiving_exception(po, line, qty, reason)` | Record over/short/damaged receipt exception |
| `create_supplier_quality_incident(po, severity, findings)` | Quality incident for vendor scorecard |
| `create_ap_debit_memo(invoice, vendor, amount, reason)` | Debit memo for freight damage recovery |

---

## 7. Oracle Transportation Management

**File:** `src/integrations/tms/oracle_tms.py`

### Environment variables
```dotenv
ORACLE_TMS_BASE_URL=https://<tenancy>.otm.ocs.oraclecloud.com
ORACLE_TMS_IDCS_URL=https://<tenancy>.identity.oraclecloud.com
ORACLE_TMS_CLIENT_ID=<client_id>
ORACLE_TMS_CLIENT_SECRET=<secret>
ORACLE_TMS_DOMAIN=DEFAULT
```

### Key operations
| Method | Description |
|--------|-------------|
| `get_shipment(gid)` | Full shipment record including stops |
| `search_shipments_by_bol(bol)` | Find shipment(s) by BOL number |
| `post_damage_event(event)` | Post RD (Received with Damage) event |
| `update_shipment_status(gid, status)` | Update shipment lifecycle status |
| `place_order_hold(gid, code)` | Hold order release for damage inspection |
| `release_order_hold(gid, hold_id)` | Remove hold |
| `record_delivery_scan(gid, sscc, condition, location, by)` | Delivery confirmation event |

---

## 8. SAP Transportation Management

**File:** `src/integrations/tms/sap_tm.py`

### Environment variables
```dotenv
SAP_TM_BASE_URL=https://<sid>-api.cfapps.eu10.hana.ondemand.com
SAP_TM_XSUAA_URL=https://<subaccount>.authentication.eu10.hana.ondemand.com
SAP_TM_CLIENT_ID=<client_id>
SAP_TM_CLIENT_SECRET=<secret>
SAP_TM_EVENT_MESH_NAMESPACE=sap/tm/damage/v1
```

### Key operations
| Method | Description |
|--------|-------------|
| `get_freight_order(id)` | Freight order header + freight units |
| `find_freight_order_by_sscc(sscc)` | Search by handling unit SSCC |
| `create_damage_notification(notif)` | Damage notification (severity code 1–4) |
| `block_freight_unit(id, reason)` | Planning block on freight unit |
| `unblock_freight_unit(id)` | Remove planning block |
| `publish_damage_event(fo_id, sscc, severity)` | CloudEvent to SAP Event Mesh |

---

## 9. Carrier EDI (214 / 210)

**File:** `src/integrations/carriers/edi.py`

### EDI 214 — Shipment Status
- Parsed to `ShipmentStatusEvent` dataclass
- Damage exception codes: `X6` (Refused), `D15` (Damage), `X3` (Shortage), `X8` (Damaged)
- Use `EDI214Document.damage_exceptions` to get filtered damage events

### EDI 210 — Freight Invoice
- Parsed to `FreightInvoiceLine` list
- Use `total_charges` for AP reconciliation against expected freight cost

### Integration pattern
```python
from src.integrations.carriers.edi import EDI214Parser, EDI210Parser

parser = EDI214Parser()
doc = parser.parse(raw_edi_text)
for exc in doc.damage_exceptions:
    logger.warning(f"Damage exception from carrier: {exc.tracking_number} "
                   f"code={exc.status_code} at {exc.event_location}")
```

---

## Common Patterns

### Lazy initialisation
All clients take credentials at construction time but establish connections
lazily (first request). This allows them to be constructed during app startup
even if the remote system is temporarily unavailable.

### Token caching
All OAuth2 providers cache the access token and refresh it 60 seconds before
expiry. No action needed from callers.

### Error handling
All HTTP calls raise `httpx.HTTPStatusError` on 4xx/5xx responses.  Wrap calls
in `try/except` and log the `response.text` for debugging:
```python
try:
    result = client.create_damage_hold(sscc=sscc, ...)
except httpx.HTTPStatusError as e:
    logger.error(f"WMS error: {e.response.status_code} {e.response.text}")
```

### Timeouts
Default timeout for all connectors is 30 seconds.  Override at construction:
```python
client = ManhattanWMSClient(..., timeout=60)
```
