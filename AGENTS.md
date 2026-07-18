# Repository instructions

- Make the smallest necessary change for the active milestone.
- Inspect relevant files before editing and run focused tests after each change.
- Never expose API keys, tokens, credentials, private data, papers, datasets, experiment details, or novel research ideas.
- Do not access or ingest complete local datasets. Tests and demos must use small synthetic or explicitly bounded samples.
- Do not batch-delete files or directories. If bulk deletion appears necessary, stop and ask the user to do it manually.
- Never use `del /s`, `rd /s`, `rmdir /s`, `Remove-Item -Recurse`, or `rm -rf`.
- Preserve the legacy `/healthz`, `/analyze`, and `/eval/run` API behavior.
- Agents may call only allowlisted typed tools. Never execute model-generated shell, Python, filesystem paths, or arbitrary SQL.
- Risk and evidence gates are deterministic and cannot be disabled or dynamically downweighted.
- Do not add RAG, Chroma, real-money trading, or order execution.
- Keep runtime secrets in ignored environment configuration and never commit or log their values.
- Add a Git checkpoint after every stable milestone. Never force-push or auto-merge.

