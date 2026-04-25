from nyayarl.models import *
from nyayarl.argument_chain import ArgumentChainValidator

case = CaseFile(
    case_id="C001", fir="Accused stabbed victim",
    accused_count=1,
    evidence_items=[
        EvidenceItem(id="E001", description="Knife", type=EvidenceType.PHYSICAL, is_present=True),
        EvidenceItem(id="E002", description="Contract", type=EvidenceType.DOCUMENTARY, is_present=True)
    ],
    witness_statements=[
        WitnessStatement(id="W001", content="Saw attack", reliability=0.8, is_contradicting=False),
        WitnessStatement(id="W002", content="Saw nothing", reliability=0.3, is_contradicting=True)
    ],
    applicable_ipc_sections=["IPC_302"],
    curriculum_level=1,
    precedent_id="P001"
)

v = ArgumentChainValidator()

# Test 1: valid Step 1
a1 = Action(step_type=StepType.ACTUS_REUS, cited_ipc_sections=[], anchored_evidence_ids=["E001"], anchored_witness_ids=[], judgment=None)
r1 = v.validate(a1, case, [])
assert r1.is_valid == True, f"FAIL: {r1.failure_reason}"
print("PASS: valid step 1")

# Test 2: Step 1 with documentary evidence — must fail
a_bad = Action(step_type=StepType.ACTUS_REUS, cited_ipc_sections=[], anchored_evidence_ids=["E002"], anchored_witness_ids=[], judgment=None)
r_bad = v.validate(a_bad, case, [])
assert r_bad.is_valid == False, "FAIL: documentary evidence should be rejected at step 1"
print("PASS: documentary evidence rejected at step 1")

# Test 3: Step 2 with low reliability witness — must fail
a2_bad = Action(step_type=StepType.MENS_REA, cited_ipc_sections=[], anchored_evidence_ids=[], anchored_witness_ids=["W002"], judgment=None)
r2_bad = v.validate(a2_bad, case, [a1])
assert r2_bad.is_valid == False, "FAIL: low reliability witness should be rejected"
print("PASS: low reliability witness rejected at step 2")

# Test 4: out of order step — must raise ValueError
try:
    a3 = Action(step_type=StepType.LINKAGE, cited_ipc_sections=["IPC_302"], anchored_evidence_ids=["E001"], anchored_witness_ids=[], judgment=None)
    v.validate(a3, case, [a1])
    print("FAIL: out of order step should raise ValueError")
except ValueError:
    print("PASS: out of order step raises ValueError")

print("validator smoke test done")
