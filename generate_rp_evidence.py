import json
import os
import hashlib

def calculate_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    base_dir = os.path.dirname(__file__)
    
    # 1. Verify proof artifact exists
    proof_path = os.path.join(base_dir, "formal_proofs", "rp_proof.txt")
    if not os.path.exists(proof_path):
        print(f"[ERROR] Proof artifact not found: {proof_path}")
        return
    
    proof_sha = calculate_sha256(proof_path)
    
    # 2. Get action spec sha
    action_spec_path = os.path.join(base_dir, "action_spec.json")
    action_sha = calculate_sha256(action_spec_path)
    
    # 3. Create evidence JSON with proper proof artifact binding
    evidence = {
        "schema": "yangmills.rp_evidence.v1",
        "action_spec": {
            "sha256": action_sha
        },
        "reflection": {
            "axis": "t (time)"
        },
        "provenance": {
            "method": "Character expansion (Osterwalder-Seiler 1978)",
            "proof_artifact_sha256": proof_sha,
            "description": "Reflection positivity proved via character expansion of Wilson plaquette action. "
                           "Positivity of character expansion coefficients c_R(beta) > 0 and orthogonality "
                           "of characters under Haar integration yield <(Theta A) A> >= 0.",
            "source": "formal_proofs/rp_proof.txt"
        },
        "proof": {
            "schema": "yangmills.rp_proof_artifact.v1",
            "sha256": proof_sha
        }
    }
    
    output_path = os.path.join(base_dir, "rp_evidence.json")
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=4)
        
    print(f"Generated rp_evidence.json with proof SHA: {proof_sha}")

if __name__ == "__main__":
    main()
