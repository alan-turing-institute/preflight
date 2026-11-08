#!/usr/bin/env python3
"""Example `notify_command`: mail each notice to the user whose job it is about, through the host's `sendmail`.

Site file:  notify_command = ["/usr/bin/python3", "/opt/gpujobs/slurm/notify_mail.py", "--domain", "example.org"]

The agent runs the command once per notice with the notice in its environment (the contract is the `notify_command`
row of the README): GPUJOBS_EVENT (failed | idle | over_ask | memory_stop), GPUJOBS_JOB, GPUJOBS_NAME,
GPUJOBS_PROJECT, GPUJOBS_USER and GPUJOBS_MESSAGE, plus GPUJOBS_STATE for a failure, GPUJOBS_ROLE, GPUJOBS_OTHER_JOB
and GPUJOBS_OTHER_PROJECT for an idle finding, and GPUJOBS_ENDED for a memory stop. The subject names the event. Without `--domain` the mail goes to the local mailbox of GPUJOBS_USER. A notice with no user (the job
had already left slurm) is skipped. Any other channel (Slack, a chat bot) is a script reading the same variables.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from email.message import EmailMessage

#: A Unix user name as useradd accepts it; anything else is not mailed (the name comes from slurm, not the submitter).
USER = re.compile(r"[a-z_][a-z0-9_.-]{0,31}")
SENDMAIL_TIMEOUT_S = 20


def build(env: dict[str, str], domain: str | None, sender: str) -> EmailMessage | None:
    """The mail for one notice, or None when it has no valid user to go to."""
    user = env.get("GPUJOBS_USER", "")
    if not USER.fullmatch(user):
        return None
    mail = EmailMessage()
    mail["To"] = f"{user}@{domain}" if domain else user
    mail["From"] = sender
    event, job = env.get("GPUJOBS_EVENT", "notice"), env.get("GPUJOBS_JOB", "?")
    mail["Subject"] = f"[gpujobs] {event}: job #{job}"  # never the submitter's own text in a header
    mail.set_content(env.get("GPUJOBS_MESSAGE", "") + "\n")
    return mail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mail one gpujobs notice to the job's user.")
    parser.add_argument("--domain", help="mail domain added to the user name; local mailbox when omitted")
    parser.add_argument("--sender", default="gpujobs", help="From address (default: gpujobs)")
    parser.add_argument("--sendmail", default="/usr/sbin/sendmail", help="sendmail program (default: %(default)s)")
    args = parser.parse_args(argv)
    mail = build(dict(os.environ), args.domain, args.sender)
    if mail is None:
        print("[notify_mail] no valid user for this notice; not mailed", file=sys.stderr)
        return 0
    done = subprocess.run([args.sendmail, "-t", "-oi"], input=mail.as_bytes(), timeout=SENDMAIL_TIMEOUT_S, check=False)
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
