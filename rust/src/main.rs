// nsmb_planner: Rust port of the geometry queries in src/tilemap_planner.py.
//
// Built with rustc alone (no cargo, no crates): Smart App Control on this
// machine blocks cargo.exe, so this file stays within std on purpose.
//
// Modes:
//   nsmb_planner <geometry.bin> serve            line protocol on stdin/stdout
//   nsmb_planner <geometry.bin> batch <queries>  answer a file of queries
//   nsmb_planner <geometry.bin> bench <iters>    throughput of the 4 queries
//
// Queries (one per line, whitespace separated):
//   G                      -> G <min_col> <max_col> <reach> <feasible> <goal_px>
//   S <px> <look>          -> surface_ahead
//   W <px> <look> <min_h>  -> wall_ahead
//   P <px> <look> <min_h>  -> pipe_ahead
//   N <px>                 -> is_narrow_corridor

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::ExitCode;
use std::time::Instant;

const TILE: i32 = 16;
const FLOOR_DEFAULT: i32 = 30;

#[derive(Debug)]
struct Geometry {
    min_col: i32,
    max_col: i32,
    reach: i32,
    feasible: bool,
    surface: HashMap<i32, i32>,
    obstacle_h: HashMap<i32, i32>,
    ceilings: HashMap<i32, Vec<i32>>,
}

fn u32_at(b: &[u8], off: usize) -> Result<u32, String> {
    if off + 4 > b.len() {
        return Err("truncated".to_string());
    }
    Ok(u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]))
}

fn i32_at(b: &[u8], off: usize) -> Result<i32, String> {
    Ok(u32_at(b, off)? as i32)
}

fn load(path: &str) -> Result<Geometry, String> {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| format!("open {path}: {e}"))?
        .read_to_end(&mut b)
        .map_err(|e| format!("read {path}: {e}"))?;
    if b.len() < 8 || &b[..8] != b"NSMBGEO1" {
        return Err("bad magic (re-run tools/export_geometry.py)".to_string());
    }
    let version = u32_at(&b, 8)?;
    if version != 1 {
        return Err(format!("unsupported geometry version {version}"));
    }
    let mut o = 12usize;
    let min_col = i32_at(&b, o)?;
    let max_col = i32_at(&b, o + 4)?;
    let reach = i32_at(&b, o + 8)?;
    let feasible = i32_at(&b, o + 12)? != 0;
    o += 16;

    let mut surface = HashMap::new();
    let n = u32_at(&b, o)? as usize;
    o += 4;
    for _ in 0..n {
        let c = i32_at(&b, o)?;
        let v = i32_at(&b, o + 4)?;
        surface.insert(c, v);
        o += 8;
    }

    let mut obstacle_h = HashMap::new();
    let n = u32_at(&b, o)? as usize;
    o += 4;
    for _ in 0..n {
        let c = i32_at(&b, o)?;
        let v = i32_at(&b, o + 4)?;
        obstacle_h.insert(c, v);
        o += 8;
    }

    // Optional trailing sections: solid, pits, takeoffs, and 2D ceilings
    let mut ceilings: HashMap<i32, Vec<i32>> = HashMap::new();
    if o + 4 <= b.len() {
        let n_solid = u32_at(&b, o)? as usize;
        o += 4 + n_solid * 4;
    }
    if o + 4 <= b.len() {
        let n_pits = u32_at(&b, o)? as usize;
        o += 4 + n_pits * 8;
    }
    if o + 4 <= b.len() {
        let n_tk = u32_at(&b, o)? as usize;
        o += 4 + n_tk * 20;
    }
    if o + 4 <= b.len() {
        let n_ceil = u32_at(&b, o)? as usize;
        o += 4;
        for _ in 0..n_ceil {
            if o + 8 <= b.len() {
                let c = i32_at(&b, o)?;
                let r = i32_at(&b, o + 4)?;
                ceilings.entry(c).or_default().push(r);
                o += 8;
            }
        }
    }

    Ok(Geometry {
        min_col,
        max_col,
        reach,
        feasible,
        surface,
        obstacle_h,
        ceilings,
    })
}

impl Geometry {
    /// int(px / TILE): truncation toward zero, matching Python's int().
    fn col_of(&self, px: f64) -> i32 {
        (px / TILE as f64) as i32
    }

    fn goal_px(&self) -> i32 {
        self.max_col * TILE
    }

    fn surface_ahead(&self, px: f64, look: i32) -> (Option<i32>, Vec<(i32, Option<i32>)>) {
        let col = self.col_of(px);
        let cur = self.surface.get(&col).copied();
        let mut prof = Vec::with_capacity(look as usize);
        for d in 1..=look {
            prof.push((col + d, self.surface.get(&(col + d)).copied()));
        }
        (cur, prof)
    }

    fn wall_ahead(&self, px: f64, look: i32, min_wall: i32) -> i32 {
        let col = self.col_of(px);
        for d in 1..=look {
            if *self.obstacle_h.get(&(col + d)).unwrap_or(&0) >= min_wall {
                return d;
            }
        }
        0
    }

    fn pipe_ahead(&self, px: f64, look: i32, min_h: i32) -> Option<f64> {
        let col = self.col_of(px);
        let cur = self.surface.get(&col).copied().unwrap_or(FLOOR_DEFAULT);
        for d in 1..=look {
            let ahead = col + d;
            let h = *self.obstacle_h.get(&ahead).unwrap_or(&0);
            let s = self.surface.get(&ahead).copied();
            let rise = match s {
                Some(s) => cur - s,
                None => 0,
            };
            if h >= min_h || rise >= min_h {
                return Some((ahead * TILE) as f64 - px);
            }
        }
        None
    }

    fn is_narrow_corridor(&self, px: f64) -> bool {
        let col = self.col_of(px);
        (91..=97).contains(&col)
    }

    fn ceiling_ahead(&self, px: f64, mario_row: Option<i32>, look: i32, clearance: i32) -> i32 {
        let col = self.col_of(px);
        let base_row = mario_row.or_else(|| self.surface.get(&col).copied()).unwrap_or(FLOOR_DEFAULT);
        for d in 0..=look {
            if let Some(rows) = self.ceilings.get(&(col + d)) {
                for &cr in rows {
                    if cr >= base_row - clearance && cr < base_row {
                        return if d == 0 { 1 } else { d };
                    }
                }
            }
        }
        0
    }

    fn answer(&self, line: &str) -> String {
        let mut it = line.split_whitespace();
        let op = match it.next() {
            Some(o) => o,
            None => return "ERR empty".to_string(),
        };
        let num = |s: Option<&str>| s.and_then(|v| v.parse::<f64>().ok());
        match op {
            "G" => format!(
                "G {} {} {} {} {}",
                self.min_col,
                self.max_col,
                self.reach,
                self.feasible,
                self.goal_px()
            ),
            "S" => {
                let px = match num(it.next()) {
                    Some(v) => v,
                    None => return "ERR S needs px".to_string(),
                };
                let look = num(it.next()).unwrap_or(4.0) as i32;
                let (cur, prof) = self.surface_ahead(px, look);
                let mut out = String::from("S");
                out.push(' ');
                out.push_str(&match cur {
                    Some(c) => c.to_string(),
                    None => "NA".to_string(),
                });
                for (c, r) in prof {
                    out.push(' ');
                    match r {
                        Some(rr) => out.push_str(&format!("{c}:{rr}")),
                        None => out.push_str(&format!("{c}:NA")),
                    }
                }
                out
            }
            "W" => {
                let px = match num(it.next()) {
                    Some(v) => v,
                    None => return "ERR W needs px".to_string(),
                };
                let look = num(it.next()).unwrap_or(3.0) as i32;
                let minh = num(it.next()).unwrap_or(3.0) as i32;
                format!("W {}", self.wall_ahead(px, look, minh))
            }
            "P" => {
                let px = match num(it.next()) {
                    Some(v) => v,
                    None => return "ERR P needs px".to_string(),
                };
                let look = num(it.next()).unwrap_or(5.0) as i32;
                let minh = num(it.next()).unwrap_or(2.0) as i32;
                match self.pipe_ahead(px, look, minh) {
                    Some(d) => format!("P {:.6}", d),
                    None => "P NA".to_string(),
                }
            }
            "N" => {
                let px = match num(it.next()) {
                    Some(v) => v,
                    None => return "ERR N needs px".to_string(),
                };
                format!("N {}", self.is_narrow_corridor(px) as i32)
            }
            "C" => {
                let px = match num(it.next()) {
                    Some(v) => v,
                    None => return "ERR C needs px".to_string(),
                };
                let row_tok = it.next();
                let mrow = match row_tok {
                    Some("NA") | None => None,
                    Some(s) => s.parse::<f64>().ok().map(|v| v.round() as i32),
                };
                let look = num(it.next()).unwrap_or(4.0) as i32;
                let clr = num(it.next()).unwrap_or(3.0) as i32;
                format!("C {}", self.ceiling_ahead(px, mrow, look, clr))
            }
            other => format!("ERR unknown op {other}"),
        }
    }
}

fn bench(g: &Geometry, iters: u64) {
    let t0 = Instant::now();
    let mut acc_s = 0i64;
    let mut acc_w = 0i64;
    let mut acc_p = 0i64;
    let mut acc_n = 0i64;
    for i in 0..iters {
        let px = (i % 4200) as f64;
        acc_s += g.surface_ahead(px, 5).1.len() as i64;
        acc_w += g.wall_ahead(px, 6, 2) as i64;
        acc_p += g.pipe_ahead(px, 5, 2).map(|v| v as i64).unwrap_or(-1);
        acc_n += g.is_narrow_corridor(px) as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    let q = 4.0 * iters as f64;
    println!(
        "bench {} iters ({} queries) in {:.3}s -> {:.0} queries/s | sink {}",
        iters,
        q as i64,
        dt,
        q / dt,
        acc_s + acc_w + acc_p + acc_n
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nsmb_planner <geometry.bin> serve|batch <f>|bench <iters>");
        return ExitCode::from(2);
    }
    let g = match load(&args[1]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("geometry: {e}");
            return ExitCode::from(3);
        }
    };

    match args[2].as_str() {
        "bench" => {
            let n: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
            bench(&g, n);
            ExitCode::SUCCESS
        }
        "batch" => {
            let path = match args.get(3) {
                Some(p) => p.clone(),
                None => {
                    eprintln!("batch needs a query file");
                    return ExitCode::from(2);
                }
            };
            let f = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("open {path}: {e}");
                    return ExitCode::from(3);
                }
            };
            let out = io::stdout();
            let mut w = io::BufWriter::new(out.lock());
            for line in BufReader::new(f).lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                if line.trim().is_empty() {
                    continue;
                }
                if let Err(e) = writeln!(w, "{}", g.answer(&line)) {
                    eprintln!("write: {e}");
                    return ExitCode::from(4);
                }
            }
            ExitCode::SUCCESS
        }
        _ => {
            // serve: one answer per stdin line, flushed so the client never stalls.
            let stdin = io::stdin();
            let out = io::stdout();
            let mut w = io::BufWriter::new(out.lock());
            let mut r = BufReader::new(stdin.lock());
            let mut line = String::new();
            loop {
                line.clear();
                match r.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
                let trimmed = line.trim_end_matches(['\n', '\r']);
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "quit" {
                    let _ = w.flush();
                    break;
                }
                if writeln!(w, "{}", g.answer(trimmed)).is_err() {
                    break;
                }
                if w.flush().is_err() {
                    break;
                }
            }
            ExitCode::SUCCESS
        }
    }
}
