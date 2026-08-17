# Security

## Deployment Boundary

This is a local research toolkit, not a hardened hosted service. Bind HTTP
services to `127.0.0.1` by default. Tailscale or another trusted VPN is suitable
when host firewall rules restrict access to intended clients. Do not expose the
ports directly to the public internet.

The applications do not provide TLS termination, rate limiting, account
management, tenant isolation, or a production secret store.

## Data Exposure

Treat generated images, prompts, labels, session files, score databases, and
exports as private data. Review and grader APIs can expose dataset, task,
session, seed, workflow, checkpoint, and run identifiers. Checkpoint aliases
improve display readability but do not remove the underlying values. The
current interfaces are not strict blind-review systems.

Use a generated labeler invite token and do not commit `.env`, `.env.oss`,
`image_grader/config.local.json`, or local checkpoint overlays.

## Model And File Safety

- Keep grader model roots and allowed image roots narrow.
- Obtain model weights separately and verify their source and license.
- Do not enable a pickle-backed model unless the exact file is trusted.
- Keep `trusted_pickle` disabled for unverified files.
- Keep ComfyUI and custom nodes updated according to their own security policy.

## Wildcard Installer

The wildcard installer downloads only after explicit acceptance and validates
pinned sizes and SHA-256 hashes. A Civitai API token should be supplied through
the process environment, never added to repository files or command history.

Proxy CA and query-token fallbacks weaken normal transport protections. Enable
them only for proxy infrastructure you control. Query-token mode can expose the
first request URL to that proxy's logs.

## Reports

This one-time source release has no guaranteed private vulnerability response
channel or maintenance window. Before publication, configure the GitHub
repository's security contact or private vulnerability-reporting feature if
you intend to accept reports.
