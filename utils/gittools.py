from subprocess import Popen, PIPE
import shlex

def run_command(command):
    p = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
    out, err = [s.decode('utf-8') for s in p.communicate()]
    if len(err)>0:
        raise RuntimeError(err)
    return out

def git_diff():
    return run_command("git diff")

def git_revision_hash():
    return run_command("git rev-parse HEAD")

def git_revision_short_hash():
    return run_command("git rev-parse --short HEAD")
