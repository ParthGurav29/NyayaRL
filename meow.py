import urllib.request, json

BASE = "http://127.0.0.1:8000"

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    return json.loads(res.read())

# 🔹 Reset
r = post("/reset", {"session_id": "test1", "curriculum_level": 1})
case = r["case_file"]
print("RESET OK:", case["template_id"])

# (Optional but smart) — print valid IDs
print("\nAvailable Evidence (usable):")
for e in case["evidence_items"]:
    if e["is_present"]:
        print(" -", e["id"])

print("\nAvailable Witness (reliable):")
for w in case["witness_statements"]:
    if w["reliability"] > 0.5:
        print(" -", w["id"], f"(rel={w['reliability']:.2f})")

# 🔹 Steps (FIXED — flattened payload)
steps = [
    ("actus_reus", {
        "anchored_evidence_ids": ["stolen_property_recovered"],
        "anchored_witness_ids": [],
        "cited_ipc_sections": [],
        "judgment": None
    }),

    ("mens_rea", {
        "anchored_evidence_ids": [],
        "anchored_witness_ids": ["police_investigator"],
        "cited_ipc_sections": [],
        "judgment": None
    }),

    ("linkage", {
        "anchored_evidence_ids": ["cctv_robbery"],
        "anchored_witness_ids": [],
        "cited_ipc_sections": ["392"],
        "judgment": None
    }),

    ("counter_argument", {
        "anchored_evidence_ids": ["stolen_property_recovered"],
        "anchored_witness_ids": ["victim_robbery"],
        "cited_ipc_sections": [],
        "judgment": None
    }),

    ("ipc_application", {
        "anchored_evidence_ids": [],
        "anchored_witness_ids": [],
        "cited_ipc_sections": ["392"],
        "judgment": None
    }),

    ("precedent_citation", {
        "anchored_evidence_ids": [],
        "anchored_witness_ids": [],
        "cited_ipc_sections": ["392"],
        "judgment": "convict"
    }),
]

# 🔹 Execute steps
for step_type, action in steps:
    payload = {
        "session_id": "test1",
        "step_type": step_type,
        **action   # ⭐ IMPORTANT FIX (flattening)
    }

    print("\nSENDING:", payload)

    r = post("/step", payload)
    result = r["step_result"]

    print(f"{step_type:25s}  valid={result['is_valid']}  reward={result['reward']}  {result.get('failure_reason','')}")

    if not result["is_valid"]:
        print("❌ FAILED — stopping")
        break

print("\nDONE:", r.get("done"))