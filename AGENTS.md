# AGENTS.md

## Default Completion Workflow

For every completed code/edit task:

1. Run the relevant validation commands for the change.
2. Commit the finished changes with a clear, descriptive commit message.
3. Push the current branch to the configured remote.

Assume this workflow by default unless the user explicitly asks not to commit/push.
If push fails, report the exact error and required next step.
