from nyayarl.environment import NyayaRLEnvironment
from nyayarl.agents.judge_agent import JudgeAgent
from nyayarl.agents.prosecution_agent import ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter

precedents = PrecedentsDB()
judge = Track2JudgeAdapter(JudgeAgent(), precedents)
prosecution = Track2ProsecutionAdapter(ProsecutionAgent())
case_generator = Track2CaseGenerator()

env = NyayaRLEnvironment(case_generator=case_generator, judge=judge, prosecution=prosecution)

obs = env.reset(curriculum_level=1)
assert obs.is_done == False
assert obs.chain_score == 0.0
assert len(obs.submitted_steps) == 0
print("PASS: reset works")

# step on done episode must raise
# exhaust episode first via stuck termination
for _ in range(10):
    from nyayarl.models import Action, StepType
    bad_action = Action(step_type=StepType.ACTUS_REUS, cited_ipc_sections=[], anchored_evidence_ids=["INVALID_ID"], anchored_witness_ids=[], judgment=None)
    try:
        obs, result, done = env.step(bad_action)
        if done:
            break
    except RuntimeError:
        print("PASS: step on done episode raises RuntimeError")
        break

try:
    obs, result, done = env.step(bad_action)
except RuntimeError:
    print("PASS: step on done episode raises RuntimeError")

print("environment smoke test done")
