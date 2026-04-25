from training.curriculum import CurriculumManager

cm = CurriculumManager(window_size=10, promotion_threshold=0.8, min_window_size=10)

# should not promote on sparse data
for i in range(5):
    cm.record_episode(True, 1)
assert cm.should_promote() == False, "FAIL: promoted on sparse window"
print("PASS: no promotion on sparse window")

# fill window with 80% success
for i in range(5):
    cm.record_episode(i < 4, 1)
assert cm.should_promote() == True, "FAIL: should promote at 80%"
new_level = cm.promote()
assert new_level == 2, f"FAIL: expected level 2, got {new_level}"
print("PASS: promotion works correctly")

# verify window cleared after promotion
assert cm.should_promote() == False, "FAIL: window not cleared after promotion"
print("PASS: window cleared after promotion")

print("curriculum test done")
