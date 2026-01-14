"""
Export Verification Results to LaTeX
====================================

This script runs all verification modules and exports the results to a 
LaTeX-compatible format that can be directly \input{} into the paper.

Usage:
    python export_results_to_latex.py

Output:
    - verification_results.tex  (LaTeX macros for all numerical results)
    - verification_results.json (Machine-readable backup)

The paper can then use macros like:
    \VerBetaStrong, \VerBetaIntermediate, \VerJIrrelevantMax, etc.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from interval_arithmetic import Interval
from ab_initio_jacobian import AbInitioJacobianEstimator

def run_full_verification():
    """
    Run the complete verification and collect all results.
    """
    results = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "version": "1.0",
            "status": "PENDING"
        },
        "regimes": {
            "strong_coupling_max": 0.40,
            "intermediate_min": 0.40,
            "intermediate_max": 6.0,
            "weak_coupling_min": 6.0
        },
        "verification_points": {},
        "summary": {}
    }
    
    jacobian_estimator = AbInitioJacobianEstimator()
    
    # Verification points
    check_points = [0.40, 0.50, 0.63, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    all_passed = True
    max_j_irr = 0.0
    min_j_irr = 1.0
    
    for beta_val in check_points:
        beta_interval = Interval(beta_val, beta_val)
        
        try:
            J = jacobian_estimator.compute_jacobian(beta_interval)
            j_pp = J[0][0]
            j_rr = J[1][1]
            
            # Determine regime
            if beta_val <= 0.40:
                regime = "strong"
            elif beta_val < 4.5:
                regime = "crossover"
            else:
                regime = "weak"
            
            # Check contraction
            contracts = j_rr.upper < 0.99
            if not contracts:
                all_passed = False
            
            # Track extremes
            if j_rr.upper > max_j_irr:
                max_j_irr = j_rr.upper
            if j_rr.upper < min_j_irr:
                min_j_irr = j_rr.upper
            
            results["verification_points"][f"{beta_val:.2f}"] = {
                "beta": beta_val,
                "regime": regime,
                "J_marginal_lower": round(j_pp.lower, 6),
                "J_marginal_upper": round(j_pp.upper, 6),
                "J_irrelevant_lower": round(j_rr.lower, 6),
                "J_irrelevant_upper": round(j_rr.upper, 6),
                "contracts": contracts,
                "status": "PASS" if contracts else "FAIL"
            }
            
        except Exception as e:
            results["verification_points"][f"{beta_val:.2f}"] = {
                "beta": beta_val,
                "status": "ERROR",
                "error": str(e)
            }
            all_passed = False
    
    # Summary statistics
    results["summary"] = {
        "all_passed": all_passed,
        "total_points": len(check_points),
        "passed_points": sum(1 for k, v in results["verification_points"].items() if v.get("status") == "PASS"),
        "max_J_irrelevant": round(max_j_irr, 6),
        "min_J_irrelevant": round(min_j_irr, 6),
        "contraction_margin": round(0.99 - max_j_irr, 6)
    }
    
    results["metadata"]["status"] = "PASS" if all_passed else "FAIL"
    
    return results


def number_to_macro_name(num_str):
    """
    Convert a numeric string to a valid LaTeX macro name (letters only).
    E.g., "0.40" -> "ZeroPointFourZero", "6.0" -> "SixPointZero"
    """
    digit_words = {
        '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
        '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
        '.': 'Point'
    }
    return ''.join(digit_words.get(c, c) for c in num_str)


def export_to_latex(results, output_path):
    """
    Export results to a LaTeX file with macro definitions.
    LaTeX \newcommand names can only contain letters, so we convert
    numeric values like 0.40 to words like ZeroPointFourZero.
    """
    lines = [
        "% =============================================================================",
        "% AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        f"% Generated: {results['metadata']['generated']}",
        f"% Status: {results['metadata']['status']}",
        "% =============================================================================",
        "",
        "% Guard against multiple inclusion",
        "\\ifx\\VerificationResultsLoaded\\undefined",
        "\\def\\VerificationResultsLoaded{}",
        "",
        "% --- Regime Boundaries ---",
        f"\\newcommand{{\\VerBetaStrongMax}}{{{results['regimes']['strong_coupling_max']:.2f}}}",
        f"\\newcommand{{\\VerBetaIntermediateMin}}{{{results['regimes']['intermediate_min']:.2f}}}",
        f"\\newcommand{{\\VerBetaIntermediateMax}}{{{results['regimes']['intermediate_max']:.1f}}}",
        f"\\newcommand{{\\VerBetaWeakMin}}{{{results['regimes']['weak_coupling_min']:.1f}}}",
        "",
        "% --- Summary Statistics ---",
        f"\\newcommand{{\\VerTotalPoints}}{{{results['summary']['total_points']}}}",
        f"\\newcommand{{\\VerPassedPoints}}{{{results['summary']['passed_points']}}}",
        f"\\newcommand{{\\VerMaxJIrrelevant}}{{{results['summary']['max_J_irrelevant']:.4f}}}",
        f"\\newcommand{{\\VerMinJIrrelevant}}{{{results['summary']['min_J_irrelevant']:.4f}}}",
        f"\\newcommand{{\\VerContractionMargin}}{{{results['summary']['contraction_margin']:.4f}}}",
        f"\\newcommand{{\\VerStatus}}{{{results['metadata']['status']}}}",
        "",
        "% --- Per-Beta Results ---",
        "% Macro names use words for digits (e.g., \\VerJIrrelevantAtZeroPointFourZero)",
    ]
    
    for key, data in results["verification_points"].items():
        if data.get("status") == "PASS" or data.get("status") == "FAIL":
            beta_str = number_to_macro_name(key)  # 0.40 -> ZeroPointFourZero
            lines.append(f"% Beta = {data['beta']}")
            lines.append(f"\\newcommand{{\\VerJMarginalAt{beta_str}}}{{{data['J_marginal_upper']:.4f}}}")
            lines.append(f"\\newcommand{{\\VerJIrrelevantAt{beta_str}}}{{{data['J_irrelevant_upper']:.4f}}}")
            lines.append(f"\\newcommand{{\\VerStatusAt{beta_str}}}{{{data['status']}}}")
            lines.append("")
    
    # Add a verification table environment
    # NOTE: We skip the table definition in preamble to avoid expansion issues
    # The table can be constructed manually in the document using the per-beta macros
    lines.extend([
        "% --- Verification Table ---",
        "% NOTE: Table data available via per-beta macros above.",
        "% Example usage in document:",
        "%   \\VerJIrrelevantAtZeroPointFourZero gives the J_irr value at beta=0.40",
        "%   \\VerJIrrelevantAtOnePointZeroZero gives the J_irr value at beta=1.0",
        "",
        "\\fi  % End of inclusion guard",
        "% =============================================================================",
        "% END OF AUTO-GENERATED FILE",
        "% =============================================================================",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"[OK] LaTeX macros written to: {output_path}")


def export_to_json(results, output_path):
    """
    Export results to JSON for programmatic access.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] JSON results written to: {output_path}")


def main():
    print("=" * 70)
    print("YANG-MILLS VERIFICATION: EXPORT TO LATEX")
    print("=" * 70)
    
    # Run verification
    print("\n[1/3] Running full verification...")
    results = run_full_verification()
    
    # Determine output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(script_dir, "..", "split")
    
    # Export to LaTeX (in paper directory)
    latex_path = os.path.join(paper_dir, "verification_results.tex")
    print(f"\n[2/3] Exporting to LaTeX...")
    export_to_latex(results, latex_path)
    
    # Export to JSON (in verification directory)
    json_path = os.path.join(script_dir, "verification_results.json")
    print(f"\n[3/3] Exporting to JSON...")
    export_to_json(results, json_path)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"VERIFICATION STATUS: {results['metadata']['status']}")
    print(f"Points Verified: {results['summary']['passed_points']}/{results['summary']['total_points']}")
    print(f"Max J_irrelevant: {results['summary']['max_J_irrelevant']:.4f} (margin: {results['summary']['contraction_margin']:.4f})")
    print("=" * 70)
    
    if results['metadata']['status'] == 'PASS':
        print("\n✓ Paper can now use \\input{verification_results.tex} to load all values.")
    else:
        print("\n✗ Verification FAILED. Fix issues before updating paper.")
    
    return 0 if results['metadata']['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
