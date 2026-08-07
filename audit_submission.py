"""
B-P-SAFE-AMSR — Root Auditor Entrypoint
Run with: python audit_submission.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from psafe.audit_submission import main

if __name__ == "__main__":
    main()
