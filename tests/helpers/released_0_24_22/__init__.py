"""A frozen copy of the released client 0.24.22 (origin/main 88207408), as a test oracle.

Only what decides what is uploaded: the turn cut (``messages``), the uploader
(``turn_upload``) and the Hermes record conversion (``hermes_messages``), copied verbatim
but for their imports. Tests run the same sequence through this copy and through the
branch and compare what the server would store (test_hermes_upload_fuzz.py).
"""
