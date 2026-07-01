//! Auth discovery for Claude Code and Codex.
//!
//! Walks the host machine looking for credentials we can forward into
//! the Terminal-Bench Docker container. Designed to be unattended:
//! returns the first thing it finds without prompting.
//!
//! Lookup order (first hit wins):
//!
//! Claude:
//!   1. `CLAUDE_CODE_OAUTH_TOKEN` env var
//!   2. `ANTHROPIC_API_KEY` env var (only accepts sk-ant-api03- keys; oat01 tokens
//!      are automatically treated as CLAUDE_CODE_OAUTH_TOKEN)
//!   3. `bench/.env` then `<repo_root>/.env` (both keys; oat01 format in
//!      ANTHROPIC_API_KEY field is auto-coerced to CLAUDE_CODE_OAUTH_TOKEN)
//!   4. `~/.claude/.credentials.json` claudeAiOauth.accessToken (auto-refreshed
//!      via refresh_token when expired; requires network)
//!   5. `~/.claude/auth.json` fields: oauthToken / accessToken /
//!      token / claudeAiOAuth
//!
//! Codex:
//!   1. `OPENAI_API_KEY` env var
//!   2. `bench/.env` then `<repo_root>/.env`
//!   3. `~/.codex/` directory (bind-mounted into container at
//!      `/root/.codex` so the in-container Codex CLI sees host login state)

use std::path::{Path, PathBuf};

use crate::drivers::terminal_bench::normalize_for_mount;

/// Anthropic Claude Code OAuth client ID (hardcoded in the CC binary).
const CC_OAUTH_CLIENT_ID: &str = "22422756-60c9-4084-8eb7-27705fd5cf9a";
/// Anthropic token refresh endpoint (extracted from CC binary).
const CC_OAUTH_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";

/// OAuth access tokens have the `oat01` prefix; standard API keys use `api03`.
/// When an oat01-format token is stored under ANTHROPIC_API_KEY (wrong field),
/// coerce it to CLAUDE_CODE_OAUTH_TOKEN so CC accepts it correctly.
fn coerce_var_name<'a>(var_name: &'a str, token: &str) -> &'a str {
    if var_name == "ANTHROPIC_API_KEY" && token.starts_with("sk-ant-oat") {
        "CLAUDE_CODE_OAUTH_TOKEN"
    } else {
        var_name
    }
}

/// Try to refresh the OAuth access token using the refresh_token from
/// ~/.claude/.credentials.json.  On success updates the file in place and
/// returns the new access token.  On any failure returns None silently so
/// callers can fall through to the next discovery method.
fn try_refresh_oauth_token(
    creds_path: &Path,
    refresh_token: &str,
    json_value: &serde_json::Value,
) -> Option<String> {
    let body = format!(
        "grant_type=refresh_token&refresh_token={refresh_token}&client_id={CC_OAUTH_CLIENT_ID}"
    );
    let response = ureq::post(CC_OAUTH_TOKEN_URL)
        .set("Content-Type", "application/x-www-form-urlencoded")
        .send_string(&body)
        .ok()?;

    let resp: serde_json::Value = response.into_json().ok()?;
    let new_access = resp.get("access_token").and_then(|x| x.as_str())?;
    if new_access.trim().is_empty() {
        return None;
    }
    let expires_in = resp
        .get("expires_in")
        .and_then(|x| x.as_i64())
        .unwrap_or(3600);
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let new_expires_at_ms = now_ms + expires_in * 1000;

    // Write the refreshed token back to the file.
    let mut updated = json_value.clone();
    if let Some(obj) = updated
        .get_mut("claudeAiOauth")
        .and_then(|v| v.as_object_mut())
    {
        obj.insert(
            "accessToken".to_string(),
            serde_json::Value::String(new_access.to_string()),
        );
        obj.insert(
            "expiresAt".to_string(),
            serde_json::Value::Number(new_expires_at_ms.into()),
        );
    }
    if let Ok(text) = serde_json::to_string(&updated) {
        let _ = std::fs::write(creds_path, text);
    }

    Some(new_access.to_string())
}

/// Refresh the OAuth access token in `~/.claude/.credentials.json` if it is
/// within `margin_secs` of expiring (or already expired). Used between bench
/// trials so long sweeps don't get killed by the ~6h access-token TTL on
/// the Anthropic Max subscription.
///
/// Returns `Some(new_token)` if a refresh happened and succeeded.
/// Returns `None` when:
///   - no `.credentials.json` exists (env-var or .env auth path)
///   - access token still has > `margin_secs` left
///   - no refresh_token is present
///   - the refresh HTTP call failed
///
/// On success the file is also updated in place by `try_refresh_oauth_token`.
#[allow(dead_code)]
pub fn refresh_credentials_if_needed(margin_secs: i64) -> Option<String> {
    if let Some(home) = candidate_home_dirs().into_iter().next() {
        let creds_path = home.join(".claude").join(".credentials.json");
        let text = std::fs::read_to_string(&creds_path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        let oauth = v.get("claudeAiOauth")?;
        let _current = oauth.get("accessToken").and_then(|x| x.as_str())?;
        let expires_at_ms = oauth.get("expiresAt").and_then(|x| x.as_i64()).unwrap_or(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        if expires_at_ms > now_ms + margin_secs * 1000 {
            return None;
        }
        let refresh_token = oauth
            .get("refreshToken")
            .and_then(|x| x.as_str())
            .unwrap_or("");
        if refresh_token.is_empty() {
            return None;
        }
        return try_refresh_oauth_token(&creds_path, refresh_token, &v);
    }
    None
}

#[derive(Debug, Clone)]
pub struct ClaudeAuth {
    pub var_name: String,
    pub token: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub enum CodexAuth {
    Env { token: String, source: String },
    DirMount { path: PathBuf, source: String },
    None,
}

pub fn discover_claude(bench_dir: &Path, repo_root: &Path) -> Option<ClaudeAuth> {
    for var_name in ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"] {
        if let Ok(token) = std::env::var(var_name)
            && !token.trim().is_empty()
        {
            let effective = coerce_var_name(var_name, &token);
            return Some(ClaudeAuth {
                var_name: effective.to_string(),
                token,
                source: format!("env {var_name}"),
            });
        }
    }
    for dir in [bench_dir, repo_root] {
        for var_name in ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"] {
            if let Some(token) = read_dotenv(dir, var_name) {
                let effective = coerce_var_name(var_name, &token);
                return Some(ClaudeAuth {
                    var_name: effective.to_string(),
                    token,
                    source: format!("{}/.env [{var_name}]", dir.display()),
                });
            }
        }
    }
    for home in candidate_home_dirs() {
        // New credential format: .credentials.json with claudeAiOauth.accessToken.
        // If the access token is expired but a refresh token exists, auto-refresh.
        let creds_path = home.join(".claude").join(".credentials.json");
        if let Ok(text) = std::fs::read_to_string(&creds_path)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
            && let Some(oauth) = v.get("claudeAiOauth")
            && let Some(token) = oauth.get("accessToken").and_then(|x| x.as_str())
            && !token.trim().is_empty()
        {
            let expires_at_ms = oauth.get("expiresAt").and_then(|x| x.as_i64()).unwrap_or(0);
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            if expires_at_ms > now_ms {
                return Some(ClaudeAuth {
                    var_name: "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
                    token: token.to_string(),
                    source: format!("{} [claudeAiOauth.accessToken]", creds_path.display()),
                });
            }
            // Access token expired — try to refresh with the refresh token.
            let refresh_token = oauth
                .get("refreshToken")
                .and_then(|x| x.as_str())
                .unwrap_or("");
            if !refresh_token.is_empty() {
                eprintln!(
                    "kbench: setup: {} accessToken expired (expiresAt={}); \
                     attempting OAuth refresh...",
                    creds_path.display(),
                    expires_at_ms
                );
                if let Some(new_token) = try_refresh_oauth_token(&creds_path, refresh_token, &v) {
                    eprintln!("kbench: setup: OAuth refresh succeeded");
                    return Some(ClaudeAuth {
                        var_name: "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
                        token: new_token,
                        source: format!(
                            "{} [claudeAiOauth.accessToken (refreshed)]",
                            creds_path.display()
                        ),
                    });
                }
                eprintln!(
                    "kbench: setup: OAuth refresh failed; \
                     falling through to next auth source"
                );
            } else {
                eprintln!(
                    "kbench: setup: {} has an expired accessToken (expiresAt={}) \
                     and no refreshToken; set ANTHROPIC_API_KEY in .env for \
                     reliable unattended auth",
                    creds_path.display(),
                    expires_at_ms
                );
            }
        }

        // Legacy format: auth.json with flat token fields.
        let path = home.join(".claude").join("auth.json");
        if let Ok(text) = std::fs::read_to_string(&path)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
        {
            for field in ["oauthToken", "accessToken", "token", "claudeAiOAuth"] {
                if let Some(token) = v.get(field).and_then(|x| x.as_str())
                    && !token.trim().is_empty()
                {
                    return Some(ClaudeAuth {
                        var_name: "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
                        token: token.to_string(),
                        source: format!("{} [{field}]", path.display()),
                    });
                }
            }
        }
    }
    None
}

pub fn discover_codex(bench_dir: &Path, repo_root: &Path) -> CodexAuth {
    if let Ok(token) = std::env::var("OPENAI_API_KEY")
        && !token.trim().is_empty()
    {
        return CodexAuth::Env {
            token,
            source: "env OPENAI_API_KEY".to_string(),
        };
    }
    for dir in [bench_dir, repo_root] {
        if let Some(token) = read_dotenv(dir, "OPENAI_API_KEY") {
            return CodexAuth::Env {
                token,
                source: format!("{}/.env [OPENAI_API_KEY]", dir.display()),
            };
        }
    }
    for home in candidate_home_dirs() {
        let path = home.join(".codex");
        if path.is_dir() {
            return CodexAuth::DirMount {
                source: format!("{} (bind-mounted)", path.display()),
                path,
            };
        }
    }
    CodexAuth::None
}

/// On WSL2 the user's `claude` / `codex` configs typically live on the
/// Windows side at `/mnt/c/Users/<name>/`, not in the WSL home. Yield both
/// the WSL home AND every `/mnt/c/Users/*/` candidate so discovery covers
/// either layout regardless of the Linux-side $USER.
fn candidate_home_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(h) = dirs::home_dir() {
        out.push(h);
    }
    if cfg!(target_os = "linux")
        && std::fs::read_to_string("/proc/version")
            .map(|s| s.to_lowercase().contains("microsoft"))
            .unwrap_or(false)
        && let Ok(entries) = std::fs::read_dir("/mnt/c/Users")
    {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            // Skip Windows system dirs that won't have user configs.
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(
                name,
                "Public" | "Default" | "Default User" | "All Users" | "desktop.ini"
            ) {
                continue;
            }
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Convert discovered auth into Harbor CLI args. Env-based tokens become
/// `--ae=KEY=VALUE` (Harbor forwards to container env); `~/.codex` becomes
/// `--mounts=[{...}]` (Harbor bind-mounts into the container).
pub fn to_harbor_args(claude: Option<&ClaudeAuth>, codex: &CodexAuth) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(c) = claude {
        args.push(format!("--ae={}={}", c.var_name, c.token));
    }
    match codex {
        CodexAuth::Env { token, .. } => {
            args.push(format!("--ae=OPENAI_API_KEY={token}"));
        }
        CodexAuth::DirMount { path, .. } => {
            let src = normalize_for_mount(path);
            let mounts = serde_json::json!([{
                "source": src,
                "target": "/root/.codex",
                "type": "bind",
            }]);
            args.push(format!("--mounts={mounts}"));
        }
        CodexAuth::None => {}
    }
    args
}

fn read_dotenv(dir: &Path, key: &str) -> Option<String> {
    let path = dir.join(".env");
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() != key {
            continue;
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_returns_quoted_value_unquoted() {
        let dir = std::env::temp_dir().join(format!("kbench-auth-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".env"), "FOO=\"bar baz\"\nOTHER=plain\n").unwrap();

        assert_eq!(read_dotenv(&dir, "FOO").as_deref(), Some("bar baz"));
        assert_eq!(read_dotenv(&dir, "OTHER").as_deref(), Some("plain"));
        assert_eq!(read_dotenv(&dir, "MISSING"), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dotenv_skips_comments_and_blank_lines() {
        let dir = std::env::temp_dir().join(format!("kbench-auth-test2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(".env"),
            "# header comment\n\n   # indented comment\nKEY=value\n",
        )
        .unwrap();

        assert_eq!(read_dotenv(&dir, "KEY").as_deref(), Some("value"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn to_harbor_args_emits_ae_for_env_tokens() {
        let claude = ClaudeAuth {
            var_name: "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
            token: "sk-ant-xyz".to_string(),
            source: "test".to_string(),
        };
        let codex = CodexAuth::Env {
            token: "sk-oai-abc".to_string(),
            source: "test".to_string(),
        };
        let args = to_harbor_args(Some(&claude), &codex);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "--ae=CLAUDE_CODE_OAUTH_TOKEN=sk-ant-xyz");
        assert_eq!(args[1], "--ae=OPENAI_API_KEY=sk-oai-abc");
    }

    #[test]
    fn to_harbor_args_emits_mounts_for_codex_dir() {
        let codex = CodexAuth::DirMount {
            path: PathBuf::from("/home/user/.codex"),
            source: "test".to_string(),
        };
        let args = to_harbor_args(None, &codex);
        assert_eq!(args.len(), 1);
        assert!(args[0].starts_with("--mounts="));
        assert!(args[0].contains("\"target\":\"/root/.codex\""));
        assert!(args[0].contains("\"type\":\"bind\""));
    }
}
