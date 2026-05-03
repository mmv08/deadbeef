// Must match `@workgroup_size(WORKGROUP_SIZE)` in `gpu.wgsl`.
//
// A workgroup is the unit of compute work scheduled by the GPU. Each workgroup
// runs 256 shader invocations, where one invocation is one independent run of
// the shader on a single candidate salt nonce. 256 is the portable WebGPU
// default limit for compute invocations per workgroup, so this shader does not
// need to request any higher device limits.
pub(super) const WORKGROUP_SIZE: u32 = 256;

// WebGPU caps each compute dispatch dimension at 65,535 workgroups (the
// `maxComputeWorkgroupsPerDimension` minimum required limit; see
// https://www.w3.org/TR/webgpu/). We describe the dispatch as rows and columns;
// at the WebGPU boundary, columns are x and rows are y.
pub(super) const MAX_WORKGROUP_COLUMNS: u32 = 65_535;
pub(super) const CANDIDATES_PER_ROW: u32 = WORKGROUP_SIZE * MAX_WORKGROUP_COLUMNS;

// More rows let one GPU submission scan more candidates before the CPU reads
// the tiny result buffer. The maximum row count is bounded by the requirement
// that the shader's per-candidate offset (`column + row * candidates_per_row`)
// fits in a u32.
// Worst case at this limit:
//
//   MAX_WORKGROUP_COLUMNS * WORKGROUP_SIZE * MAX_DISPATCH_ROWS
//     = 65,535 * 256 * 256
//     = 4,294,901,760
//
// which is just under `u32::MAX` (4,294,967,295). The default row count stays
// conservative for slower GPUs.
pub(super) const MAX_DISPATCH_ROWS: u32 = 256;
pub(super) const DEFAULT_DISPATCH_ROWS: u32 = 8;
pub(super) const MAX_BATCH_SIZE: u32 = CANDIDATES_PER_ROW * MAX_DISPATCH_ROWS;
pub(super) const DEFAULT_BATCH_SIZE: u32 = CANDIDATES_PER_ROW * DEFAULT_DISPATCH_ROWS;

#[derive(Copy, Clone, Debug)]
pub(super) struct DispatchLayout {
    // One GPU dispatch is a 2D grid.
    //
    //     R = candidates_per_row
    //
    //              columns: 0 ---------------------- R - 1
    //            +------------------------------------------+
    // row 0      | candidates 0 -------------------- R - 1  |
    // row 1      | candidates R ------------------ 2R - 1  |
    //   ...      | ...                                      |
    //            +------------------------------------------+
    //
    //     candidate_offset = column + row * R
    //
    // Why a rectangle? WebGPU limits one dispatch dimension to 65,535
    // workgroups. If a request is just one workgroup over that limit, filling
    // the first row and adding a full second row would almost double the work.
    // Instead we make the smallest rectangle that holds the request. The
    // shader still gets dense offsets and does not need an
    // `offset >= batch_size` branch.
    //
    // Actual candidates scanned by this dispatch after rounding.
    pub(super) candidates: u32,
    // Workgroup rectangle passed to `dispatch_workgroups(columns, rows, 1)`.
    pub(super) workgroup_columns: u32,
    pub(super) workgroup_rows: u32,
    // Candidate count in one dispatch row.
    pub(super) candidates_per_row: u32,
}

impl DispatchLayout {
    pub(super) fn for_requested_candidates(requested_candidates: u32) -> Self {
        let requested_candidates = requested_candidates.clamp(WORKGROUP_SIZE, MAX_BATCH_SIZE);
        let requested_workgroups = requested_candidates.div_ceil(WORKGROUP_SIZE);

        // WebGPU caps a dispatch dimension at 65,535 workgroups. When the
        // request needs more than one row, use the fewest rows needed, then
        // spread the columns evenly across those rows. For example, one row
        // plus one workgroup becomes 32,768 columns x 2 rows, not 65,535 x 2.
        // That keeps the shader branch-free without doing a lot of unrequested
        // work.
        let workgroup_rows = requested_workgroups.div_ceil(MAX_WORKGROUP_COLUMNS);
        let workgroup_columns = requested_workgroups.div_ceil(workgroup_rows);
        let dispatched_workgroups = workgroup_columns * workgroup_rows;

        debug_assert!(workgroup_columns <= MAX_WORKGROUP_COLUMNS);
        debug_assert!(workgroup_rows <= MAX_DISPATCH_ROWS);
        Self {
            candidates: dispatched_workgroups * WORKGROUP_SIZE,
            workgroup_columns,
            workgroup_rows,
            candidates_per_row: workgroup_columns * WORKGROUP_SIZE,
        }
    }
}
