"""A frozen copy of part of the released client 0.24.22 (tag v0.24.22, 4910645), as a
test oracle.

It holds the turn cut (``messages``), the uploader (``turn_upload``) and the Hermes record
conversion (``hermes_messages``), copied verbatim but for their imports. It does not hold
the released provider (its hooks, scope and private checks). Tests run the same sequence
through this copy and through the branch and compare what the server would store
(test_hermes_upload_fuzz.py). Equality at the provider level rests on the runs against
real Hermes, not on this copy.
"""
