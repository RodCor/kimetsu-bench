//! Persistent production MCP client for reader-free measurement.
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

fn decode_tool_result(response: &Value, id: u64) -> Result<(Value, usize, usize), String> {
    if response.get("id").and_then(Value::as_u64) != Some(id) {
        return Err("MCP response ID mismatch".into());
    }
    if let Some(error) = response.get("error") {
        return Err(format!("MCP error: {error}"));
    }
    let result = response.get("result").ok_or("MCP result missing")?;
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return Err(format!("MCP tool failed: {result}"));
    }
    let blocks = result
        .get("content")
        .and_then(Value::as_array)
        .ok_or("MCP content missing")?;
    if blocks.len() != 1 || blocks[0].get("type").and_then(Value::as_str) != Some("text") {
        return Err("expected one MCP text content block".into());
    }
    let text = blocks[0]
        .get("text")
        .and_then(Value::as_str)
        .ok_or("MCP text missing")?;
    let payload: Value =
        serde_json::from_str(text).map_err(|e| format!("MCP text is not JSON: {e}"))?;
    if !payload.is_object() {
        return Err("MCP payload must be an object".into());
    }
    if payload.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(format!("context tool rejected request: {payload}"));
    }
    Ok((payload, text.len(), result.to_string().len()))
}

pub(super) struct McpMeasurement {
    pub payload: Value,
    pub text_bytes: usize,
    pub result_bytes: usize,
    pub wire_bytes: usize,
    pub latency_ms: f64,
    pub first_query: bool,
    pub server_startup_ms: f64,
    pub working_set_bytes: Option<u64>,
    pub peak_working_set_bytes: Option<u64>,
}

#[cfg(windows)]
fn process_memory(handle: windows_sys::Win32::Foundation::HANDLE) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // The caller retains the live process handle. The initialized C-layout
    // output buffer and cb match the Windows API's required structure size.
    if unsafe { K32GetProcessMemoryInfo(handle, &mut counters, counters.cb) } == 0 {
        None
    } else {
        Some((
            counters.WorkingSetSize as u64,
            counters.PeakWorkingSetSize as u64,
        ))
    }
}

fn child_memory(child: &Child) -> Option<(u64, u64)> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        process_memory(child.as_raw_handle())
    }
    #[cfg(not(windows))]
    {
        let _ = child;
        None // Unavailable is not zero; platform-specific collectors can extend this.
    }
}

pub(super) struct BrainMcp {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    queries: usize,
    startup_ms: f64,
}

impl BrainMcp {
    pub fn start(workspace: &Path, binary: &str) -> Result<Self, String> {
        let start = Instant::now();
        let mut child = Command::new(binary)
            .current_dir(workspace)
            .env("KIMETSU_USER_BRAIN", "0")
            .args(["mcp", "serve", "--workspace"])
            .arg(workspace)
            .arg("--no-user-skills")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("start MCP: {e}"))?;
        let stdin = child.stdin.take().ok_or("MCP stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("MCP stdout unavailable")?;
        let mut client = Self {
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            next_id: 1,
            queries: 0,
            startup_ms: 0.0,
        };
        let response = client
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion":"2024-11-05", "capabilities":{},
                    "clientInfo":{"name":"BrainBenchmark","version":"1"}
                }),
            )?
            .0;
        if response.get("error").is_some() {
            return Err(format!("MCP initialize failed: {response}"));
        }
        client.send(&serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}))?;
        client.startup_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(client)
    }

    fn send(&mut self, value: &Value) -> Result<(), String> {
        let stream = self.stdin.as_mut().ok_or("MCP stdin closed")?;
        writeln!(stream, "{value}")
            .and_then(|_| stream.flush())
            .map_err(|e| format!("MCP write: {e}"))
    }

    fn request(&mut self, method: &str, params: Value) -> Result<(Value, usize), String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&serde_json::json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        loop {
            let mut line = String::new();
            if self
                .stdout
                .read_line(&mut line)
                .map_err(|e| format!("MCP read: {e}"))?
                == 0
            {
                return Err("MCP exited before responding".into());
            }
            let value: Value =
                serde_json::from_str(&line).map_err(|e| format!("MCP stdout is not JSON: {e}"))?;
            if value.get("id").is_none() {
                continue;
            } // server notification
            if value.get("id").and_then(Value::as_u64) != Some(id) {
                return Err("MCP response ID mismatch".into());
            }
            return Ok((value, line.len()));
        }
    }

    pub fn context(&mut self, query: &str, budget: usize) -> Result<McpMeasurement, String> {
        let start = Instant::now();
        let id = self.next_id;
        let (response, wire_bytes) = self.request(
            "tools/call",
            serde_json::json!({
                "name":"kimetsu_brain_context", "arguments":{
                    "query":query, "budget_tokens":budget, "include_ambient":false,
                    "max_capsules":4
                }
            }),
        )?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (payload, text_bytes, result_bytes) = decode_tool_result(&response, id)?;
        let first_query = self.queries == 0;
        self.queries += 1;
        // Sample only the MCP child, after stopping the request latency clock.
        let memory = child_memory(&self.child);
        Ok(McpMeasurement {
            payload,
            text_bytes,
            result_bytes,
            wire_bytes,
            latency_ms,
            first_query,
            server_startup_ms: if first_query { self.startup_ms } else { 0.0 },
            working_set_bytes: memory.map(|m| m.0),
            peak_working_set_bytes: memory.map(|m| m.1),
        })
    }
}

impl Drop for BrainMcp {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn windows_process_memory_reports_live_working_set_and_peak() {
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let (current, peak) = process_memory(unsafe { GetCurrentProcess() }).unwrap();
        assert!(current > 0);
        assert!(peak >= current);
    }

    #[test]
    fn measures_serialized_result_and_utf8_text_without_losing_escapes() {
        let payload =
            serde_json::json!({"capsules":[{"summary":"配置\n\"quoted\""}],"used_tokens":12});
        let text = payload.to_string();
        let result = serde_json::json!({"content":[{"type":"text","text":text}]});
        let response = serde_json::json!({"jsonrpc":"2.0","id":2,"result":result});
        let (decoded, text_bytes, result_bytes) = decode_tool_result(&response, 2).unwrap();
        assert_eq!(decoded, payload);
        assert_eq!(text_bytes, text.len());
        assert_eq!(result_bytes, result.to_string().len());
        assert!(result_bytes > text_bytes);
    }

    #[test]
    fn malformed_or_failed_protocol_output_is_not_abstention() {
        for response in [
            serde_json::json!({"id":9,"result":{"content":[{"type":"text","text":"{}"}]}}),
            serde_json::json!({"id":2,"error":{"message":"failed"}}),
            serde_json::json!({"id":2,"result":{"isError":true,"content":[]}}),
            serde_json::json!({"id":2,"result":{"content":[]}}),
            serde_json::json!({"id":2,"result":{"content":[{"type":"text","text":"null"}]}}),
        ] {
            assert!(decode_tool_result(&response, 2).is_err());
        }
    }
}
