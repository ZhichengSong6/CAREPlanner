import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"hierarchical9_boundary_audit_v1"))
import core as audit_core

class SerializationRegression(unittest.TestCase):
    def test_runtime_nonfinite_diagnostics_are_json_safe(self):
        row={"nan":float("nan"),"pos":float("inf"),"neg":-float("inf"),"nested":[1.0,float("nan")]}
        safe=audit_core.json_safe(row)
        self.assertIsNone(safe["nan"])
        self.assertIsNone(safe["pos"])
        self.assertIsNone(safe["neg"])
        self.assertIsNone(safe["nested"][1])
        json.dumps(safe,allow_nan=False)

if __name__=="__main__":
    unittest.main()
