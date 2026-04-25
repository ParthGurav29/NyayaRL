import json
from collections import Counter

rewards = []
with open('logs/training.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        rewards.append(data['mean_reward'])

print(f"Total Steps Logged: {len(rewards)}")
print(f"Reward Distribution: {Counter(rewards)}")