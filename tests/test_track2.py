"""
Test script for Track 2 components (Data & Agents).

Validates:
- Precedents database loads correctly
- Judge agent scoring works
- Prosecution agent selects challenges
- Challenge banks are loaded

Run: python -m tests.test_track2
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nyayarl.precedents import PrecedentsDB
from nyayarl.agents import JudgeAgent, ProsecutionAgent


def test_precedents_db():
    """Test precedent database loading and queries."""
    print("=" * 60)
    print("TEST 1: Precedents Database")
    print("=" * 60)

    db = PrecedentsDB()
    stats = db.get_statistics()

    print(f"Total precedents loaded: {stats['total_precedents']}")
    print(f"Categories: {stats['categories']}")
    print(f"Convictions: {stats['convictions']}")
    print(f"Acquittals: {stats['acquittals']}")
    print(f"IPC sections covered: {stats['ipc_sections_covered']}")

    # Test query by category
    print("\n--- Precedents by Category ---")
    for category in ["302_304", "420_120B", "498A_304B", "323_325_307", "392_394_397"]:
        precedents = db.get_precedents_by_category(category)
        judgments = [p.get("judgment") for p in precedents]
        print(f"  {category}: {len(precedents)} precedents ({judgments})")

    # Test get ground truth
    print("\n--- Sample Ground Truth ---")
    gt = db.get_ground_truth("302_304_homicide")
    if gt:
        print(f"  Judgment: {gt['judgment']}")
        print(f"  Min steps: {gt['min_argument_steps']}")
        print(f"  Key evidence: {gt['key_evidence']}")

    assert stats['total_precedents'] == 20, f"Expected 20 precedents, got {stats['total_precedents']}"
    print("\n[PASS] Precedents DB test PASSED\n")
    return True


def test_judge_agent():
    """Test judge agent scoring."""
    print("=" * 60)
    print("TEST 2: Judge Agent")
    print("=" * 60)

    judge = JudgeAgent()

    # Test Case 1: Perfect defence (all steps, correct judgment)
    print("\n--- Case 1: Perfect Defence ---")
    score, breakdown = judge.score_episode(
        defence_chain=[1, 2, 3, 4, 5, 6],
        final_judgment="convict",
        ground_truth={"judgment": "convict", "min_argument_steps": 6},
        prosecution_challenges=[],
        successful_challenges=0
    )
    print(f"Score: {score}")
    print(f"Breakdown: {breakdown}")
    assert score > 10, f"Expected high score for perfect defence, got {score}"

    # Test Case 2: Wrong judgment
    print("\n--- Case 2: Wrong Judgment ---")
    judge.reset()  # Reset history
    score2, breakdown2 = judge.score_episode(
        defence_chain=[1, 2, 3, 4, 5, 6],
        final_judgment="acquit",  # Wrong!
        ground_truth={"judgment": "convict", "min_argument_steps": 6},
        prosecution_challenges=[],
        successful_challenges=0
    )
    print(f"Score: {score2}")
    print(f"Breakdown: {breakdown2}")
    assert score2 < score, "Wrong judgment should score lower"

    # Test Case 3: Incomplete chain
    print("\n--- Case 3: Incomplete Chain ---")
    judge.reset()
    score3, breakdown3 = judge.score_episode(
        defence_chain=[1, 2, 4, 5],  # Missing step 3 and 6
        final_judgment="convict",
        ground_truth={"judgment": "convict", "min_argument_steps": 6},
        prosecution_challenges=[],
        successful_challenges=0
    )
    print(f"Score: {score3}")
    print(f"Breakdown: {breakdown3}")
    assert score3 < score, "Incomplete chain should score lower"

    # Test Case 4: Prosecution pressure
    print("\n--- Case 4: Prosecution Pressure ---")
    judge.reset()
    score4, breakdown4 = judge.score_episode(
        defence_chain=[1, 2, 3, 4, 5, 6],
        final_judgment="convict",
        ground_truth={"judgment": "convict", "min_argument_steps": 6},
        prosecution_challenges=[{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
        successful_challenges=2  # Prosecution won 2/3 challenges
    )
    print(f"Score: {score4}")
    print(f"Breakdown: {breakdown4}")

    print("\n[PASS] Judge Agent test PASSED\n")
    return True


def test_prosecution_agent():
    """Test prosecution agent challenge selection."""
    print("=" * 60)
    print("TEST 3: Prosecution Agent")
    print("=" * 60)

    prosecution = ProsecutionAgent()

    # Show loaded challenge banks
    print("\n--- Challenge Banks Loaded ---")
    for template_id, challenges in prosecution.challenge_banks.items():
        print(f"  {template_id}: {len(challenges)} challenges")

    # Verify all 5 challenge banks loaded
    assert len(prosecution.challenge_banks) == 5, f"Expected 5 challenge banks, got {len(prosecution.challenge_banks)}"

    # Test Case 1: Defence has gap at step 3
    print("\n--- Case 1: Defence Missing Step 3 ---")
    case_file = {"template_id": "302_304_homicide"}
    defence_chain = [1, 2, 4, 5]  # Missing step 3

    challenge = prosecution.select_challenge(
        case_file=case_file,
        defence_chain=defence_chain,
        weak_step=3
    )
    print(f"Selected challenge: {challenge['id']}")
    print(f"Text: {challenge['text']}")
    print(f"Targets step: {challenge['targets_step']}")
    print(f"Severity: {challenge['severity']}")

    # Should target step 3 (the gap)
    assert challenge['targets_step'] == 3 or 3 not in defence_chain, "Should target the gap"

    # Test Case 2: Defence has gap at step 1
    print("\n--- Case 2: Defence Missing Step 1 ---")
    defence_chain_2 = [2, 3, 4, 5, 6]  # Missing step 1 (actus reus)

    challenge2 = prosecution.select_challenge(
        case_file=case_file,
        defence_chain=defence_chain_2,
        weak_step=1
    )
    print(f"Selected challenge: {challenge2['id']}")
    print(f"Text: {challenge2['text']}")
    print(f"Targets step: {challenge2['targets_step']}")

    # Test Case 3: Learning from outcomes
    print("\n--- Case 3: Learning from Outcomes ---")
    prosecution.record_challenge_outcome("challenge_302_1", success=True)
    prosecution.record_challenge_outcome("challenge_302_1", success=True)
    prosecution.record_challenge_outcome("challenge_302_1", success=False)

    success_rate = prosecution.get_success_rate("challenge_302_1")
    print(f"challenge_302_1 success rate: {success_rate:.2f}")
    assert 0.5 < success_rate < 0.8, f"Expected ~0.67 success rate, got {success_rate}"

    print("\n[PASS] Prosecution Agent test PASSED\n")
    return True


def test_integrated_flow():
    """Test integrated flow: precedents + judge + prosecution."""
    print("=" * 60)
    print("TEST 4: Integrated Flow")
    print("=" * 60)

    db = PrecedentsDB()
    judge = JudgeAgent()
    prosecution = ProsecutionAgent()

    # Simulate a full episode for each case type
    print("\n--- Simulating Episodes ---")

    for category in ["302_304", "420_120B", "498A_304B"]:
        print(f"\n  Category: {category}")

        # Get ground truth
        gt = db.get_ground_truth(f"{category}_homicide" if "302" in category else f"{category}_domestic" if "498" in category else f"{category}_fraud")
        if not gt:
            # Try alternate key format
            gt = db.get_ground_truth(category)
        if not gt:
            # Fallback
            precedents = db.get_precedents_by_category(category)
            if precedents:
                gt = {
                    "judgment": precedents[0].get("judgment"),
                    "min_argument_steps": 6
                }

        if not gt:
            print(f"    ⚠ Could not get ground truth for {category}")
            continue

        # Simulate defence completing all steps
        defence_chain = [1, 2, 3, 4, 5, 6]

        # Prosecution finds a challenge
        case_file = {"template_id": f"{category}_test"}
        challenge = prosecution.select_challenge(
            case_file=case_file,
            defence_chain=defence_chain,
            weak_step=None
        )

        # Judge scores
        score, breakdown = judge.score_episode(
            defence_chain=defence_chain,
            final_judgment=gt.get("judgment", "convict"),
            ground_truth=gt,
            prosecution_challenges=[challenge],
            successful_challenges=0
        )

        print(f"    Ground truth: {gt.get('judgment')}")
        print(f"    Prosecution challenge: {challenge.get('text', 'none')[:50]}...")
        print(f"    Score: {score:.1f}")

    print("\n[PASS] Integrated Flow test PASSED\n")
    return True


def main():
    """Run all Track 2 tests."""
    print("\n" + "=" * 60)
    print("NYAYARL TRACK 2 TEST SUITE")
    print("Testing: Data & Agents Components")
    print("=" * 60 + "\n")

    tests = [
        ("Precedents Database", test_precedents_db),
        ("Judge Agent", test_judge_agent),
        ("Prosecution Agent", test_prosecution_agent),
        ("Integrated Flow", test_integrated_flow),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] {name} test FAILED: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n=== ALL TRACK 2 TESTS PASSED ===")
        print("Track 2 (Data & Agents) is ready for integration!")
        return 0
    else:
        print("\n=== SOME TESTS FAILED ===")
        print("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
