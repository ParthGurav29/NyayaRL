from nyayarl.models import *

# Valid construction
e = EvidenceItem(id="E001", description="Knife", type=EvidenceType.PHYSICAL, is_present=True)
w = WitnessStatement(id="W001", content="Saw the accused", reliability=0.8, is_contradicting=False)

# Boundary violations — both must raise ValueError
try:
    WitnessStatement(id="W002", content="x", reliability=1.5, is_contradicting=False)
    print("FAIL: reliability 1.5 should have raised")
except ValueError:
    print("PASS: reliability guard works")

try:
    CaseFile(case_id="C001", fir="x", accused_count=1, evidence_items=[], witness_statements=[], applicable_ipc_sections=[], curriculum_level=5, precedent_id="P001")
    print("FAIL: curriculum_level 5 should have raised")
except ValueError:
    print("PASS: curriculum_level guard works")

print("schema smoke test done")
