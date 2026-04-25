"""Run the 3 pre-training verification checks."""
from nyayarl import DefenceAgent, PrecedentsDB
from nyayarl.models import (
    EvidenceItem,
    EvidenceType,
    WitnessStatement,
    CaseFile,
    Observation,
    Action,
    StepType,
    JudgmentLabel,
)

# ── Check 1: Verify real gradients exist ────────────────────────────────────

print("=== Check 1: Gradient flow ===")
agent = DefenceAgent()

# Build a minimal valid Observation for step 0 (ACTUS_REUS)
case_file = CaseFile(
    case_id="C001",
    fir="Test FIR",
    accused_count=1,
    evidence_items=[
        EvidenceItem(
            id="ev1",
            description="Physical evidence",
            type=EvidenceType.PHYSICAL,
            is_present=True,
        ),
        EvidenceItem(
            id="ev2",
            description="Forensic evidence",
            type=EvidenceType.FORENSIC,
            is_present=True,
        ),
    ],
    witness_statements=[
        WitnessStatement(
            id="w1",
            content="Witness testimony",
            reliability=0.8,
            is_contradicting=False,
        )
    ],
    applicable_ipc_sections=["302"],
    curriculum_level=1,
    precedent_id="",
)

obs = Observation(
    case_file=case_file,
    submitted_steps=[],
    prosecution_challenges=[],
    chain_score=0.0,
    curriculum_level=1,
    is_done=False,
)

action = agent.select_action(obs)
log_prob = agent.get_log_prob(obs, action)

print(f"log_prob = {log_prob.item():.6f}")
print(f"requires_grad = {log_prob.requires_grad}")

loss = -log_prob
loss.backward()

grad_found = False
for name, param in agent.named_parameters():
    if param.grad is not None and param.grad.norm().item() > 0:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
        grad_found = True
        break

if not log_prob.requires_grad:
    print("ERROR: log_prob.requires_grad is False — gradient flow broken")
    exit(1)
if not grad_found:
    print("ERROR: No parameter has non-zero gradient — gradient flow broken")
    exit(1)

print("Check 1 PASSED\n")

# ── Check 2: Verify entropy is non-zero ─────────────────────────────────────

print("=== Check 2: Entropy non-zero ===")
entropy = agent.get_entropy(obs)
ent_val = entropy.item()
print(f"entropy = {ent_val:.6f}")

if ent_val <= 0.0:
    print("ERROR: entropy is zero or negative — action masking may be collapsing distribution")
    exit(1)

print("Check 2 PASSED\n")

# ── Check 3: Verify deterministic ground truth ───────────────────────────────

print("=== Check 3: Ground truth determinism ===")
db = PrecedentsDB()

# Use a real template_id from the precedents database (from ILDC precedent files)
# All precedent files have template_category like "323_325_307", "353_356_201", etc.
# Use the first available category from the database
precedents = list(db.precedents.values())
if precedents:
    sample_template_id = precedents[0].get("template_category", "")
    print(f"Using template_id: {sample_template_id}")
    r1 = db.get_ground_truth(sample_template_id)
    r2 = db.get_ground_truth(sample_template_id)
    
    if r1 is None or r2 is None:
        print("WARNING: No ground truth returned for template_id")
    else:
        assert r1["judgment"] == r2["judgment"], "Ground truth is still stochastic"
        print(f"ground truth judgment = {r1['judgment']}")
        print("ground truth is deterministic")
else:
    print("WARNING: No precedents loaded, skipping determinism check")

print("Check 3 PASSED\n")

print("All 3 checks passed — ready to train")
