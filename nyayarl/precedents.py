"""
Precedent Database for NyayaRL.

Loads and provides access to ILDC precedent judgments.
"""

import json
import random
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Any


class PrecedentsDB:
    """Database of ILDC precedent judgments."""

    def __init__(self, precedents_dir: str = None):
        """
        Initialize precedents database.

        Args:
            precedents_dir: Path to precedent JSON files
        """
        if precedents_dir is None:
            precedents_dir = Path(__file__).parent.parent / "data" / "precedents"

        self.precedents_dir = Path(precedents_dir)
        self.precedents: Dict[str, Dict] = {}  # case_id -> precedent data
        self.precedents_by_category: Dict[str, List[str]] = {}  # category -> [case_ids]
        self.precedents_by_ipc: Dict[int, List[str]] = {}  # ipc_section -> [case_ids]

        self._load_precedents()

    def _load_precedents(self):
        """Load all precedent JSON files."""
        if not self.precedents_dir.exists():
            return

        for file_path in self.precedents_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                case_id = data.get("case_id", file_path.stem)
                self.precedents[case_id] = data

                # Index by 2-part IPC prefix (e.g. "302_304" from "302_304_homicide")
                full_cat = data.get("template_category", "unknown")
                parts = full_cat.split("_")
                prefix = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else full_cat

                if prefix not in self.precedents_by_category:
                    self.precedents_by_category[prefix] = []
                self.precedents_by_category[prefix].append(case_id)

                # Index by IPC section
                for ipc in data.get("ipc_sections", []):
                    if ipc not in self.precedents_by_ipc:
                        self.precedents_by_ipc[ipc] = []
                    self.precedents_by_ipc[ipc].append(case_id)

    def get_precedent(self, case_id: str) -> Optional[Dict]:
        """Get precedent by case ID."""
        return self.precedents.get(case_id)

    def get_precedents_by_category(self, category: str) -> List[Dict]:
        """Get all precedents for a case category."""
        case_ids = self.precedents_by_category.get(category, [])
        return [self.precedents[cid] for cid in case_ids if cid in self.precedents]

    def get_precedents_by_ipc(self, ipc_section: int) -> List[Dict]:
        """Get all precedents for an IPC section."""
        case_ids = self.precedents_by_ipc.get(ipc_section, [])
        return [self.precedents[cid] for cid in case_ids if cid in self.precedents]

    def get_random_precedent(self, category: str = None) -> Optional[Dict]:
        """Get a random precedent, optionally filtered by category."""
        if category:
            candidates = self.get_precedents_by_category(category)
        else:
            candidates = list(self.precedents.values())

        if not candidates:
            return None
        return random.choice(candidates)

    def get_ground_truth(self, case_template_id: str) -> Optional[Dict]:
        """
        Get ground truth judgment for a case template.

        This returns the expected judgment that the agent should reach.

        Args:
            case_template_id: Template ID (e.g., "302_304_homicide")

        Returns:
            Ground truth dict with judgment, key_evidence, etc.
        """
        # Get precedents for this category
        category = case_template_id.split("_")[0] + "_" + case_template_id.split("_")[1]

        precedents = self.get_precedents_by_category(category)
        if not precedents:
            return None

        # Deterministic ground truth: same template_id -> same precedent choice.
        #
        # Randomness belongs in case generation (evidence presence, witness configs),
        # not in what the "correct answer" is.
        key = str(case_template_id)
        # IMPORTANT: Python's built-in hash() is salted per process unless PYTHONHASHSEED is fixed.
        # A salted hash silently changes "ground truth" between runs. Use a stable hash.
        idx = int(zlib.crc32(key.encode("utf-8", "ignore"))) % len(precedents)
        precedent = precedents[idx]

        return {
            "judgment": precedent.get("judgment"),
            "min_argument_steps": precedent.get("min_argument_steps", 6),
            "key_evidence": list(precedent.get("key_evidence_weights", {}).keys()),
            "key_witnesses": precedent.get("key_witnesses", []),
            "precedent_id": precedent.get("case_id")
        }

    def get_all_case_ids(self) -> List[str]:
        """Get all precedent case IDs."""
        return list(self.precedents.keys())

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        conviction_count = sum(1 for p in self.precedents.values() if p.get("judgment") == "convict")
        acquittal_count = sum(1 for p in self.precedents.values() if p.get("judgment") == "acquit")

        return {
            "total_precedents": len(self.precedents),
            "categories": len(self.precedents_by_category),
            "convictions": conviction_count,
            "acquittals": acquittal_count,
            "ipc_sections_covered": len(self.precedents_by_ipc)
        }
