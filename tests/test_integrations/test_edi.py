"""Tests for EDI 214 / 210 parsers."""
import pytest

from src.integrations.carriers.edi import (
    EDI214Parser,
    EDI210Parser,
    DAMAGE_EXCEPTION_CODES,
)

SAMPLE_EDI_214 = """ISA*00*          *00*          *ZZ*CARRIER        *ZZ*SHIPPER        *240101*0800*^*00501*000000001*0*P*>~
GS*QM*CARRIER*SHIPPER*20240101*0800*1*X*005010~
ST*214*0001~
B10*123456*MYBOLO01*MYCARR~
AT7*X6*REFUSE*DAMAGE TO GOODS**20240101*0800~
MS3*DALLAS*TX~
SE*5*0001~
GE*1*1~
IEA*1*000000001~"""

SAMPLE_EDI_210 = """ISA*00*          *00*          *ZZ*CARRIER        *ZZ*SHIPPER        *240101*0800*^*00501*000000002*0*P*>~
GS*IM*CARRIER*SHIPPER*20240101*0800*2*X*005010~
ST*210*0002~
B3*MYCARR*MYBOLO01**CC*20240101**1250.00~
L9*450*1250.00~
L0***PALLETS*2*P*35000*L~
SE*4*0002~
GE*1*2~
IEA*1*000000002~"""


class TestEDI214Parser:
    def setup_method(self):
        self.parser = EDI214Parser()

    def test_parses_tracking_and_bol(self):
        doc = self.parser.parse(SAMPLE_EDI_214)
        assert doc.sender_id.strip() == "CARRIER"
        assert len(doc.events) == 1
        event = doc.events[0]
        assert event.bol_number == "MYBOLO01"
        assert event.tracking_number == "123456"

    def test_damage_exception_flagged(self):
        doc = self.parser.parse(SAMPLE_EDI_214)
        assert len(doc.damage_exceptions) == 1
        assert doc.damage_exceptions[0].status_code == "X6"
        assert doc.damage_exceptions[0].is_damage_exception

    def test_damage_exception_codes_set(self):
        assert "X6" in DAMAGE_EXCEPTION_CODES
        assert "D15" in DAMAGE_EXCEPTION_CODES


class TestEDI210Parser:
    def setup_method(self):
        self.parser = EDI210Parser()

    def test_parses_bol_and_charges(self):
        invoices = self.parser.parse(SAMPLE_EDI_210)
        assert len(invoices) == 1
        inv = invoices[0]
        assert inv.bol_number == "MYBOLO01"
        assert inv.total_charges == 1250.0

    def test_weight_parsed(self):
        invoices = self.parser.parse(SAMPLE_EDI_210)
        assert invoices[0].weight_lbs == 35000.0
