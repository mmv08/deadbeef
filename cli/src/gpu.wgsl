const WORKGROUP_SIZE: u32 = 256u;

// Keep these structs in the same field order as `GpuInput` and `GpuResult` in
// `gpu.rs`. The Rust tests compare both sides field-by-field.
struct Input {
  // Fixed for a Safe configuration:
  // - initializer_hash = keccak256(initializer)
  // - factory = SafeProxyFactory address, 20 bytes packed into 5 words
  // - init_code_hash = keccak256(proxy init code || singleton ABI word)
  initializer_hash: array<u32, 8>,
  factory: array<u32, 5>,
  init_code_hash: array<u32, 8>,
  // Prefix bytes packed into 5 words. `prefix_len` tells us how many bytes are
  // significant, so the unused high bytes stay zero.
  prefix: array<u32, 5>,
  // First 24 bytes of the candidate salt nonce. The final 8 bytes are the
  // counter in big-endian byte order: high bytes first, then low bytes. This
  // struct stores the numeric halves as plain u32 values: `counter_low` holds
  // the lower 32 bits, and `counter_high` holds the upper 32 bits.
  nonce_prefix: array<u32, 6>,
  prefix_len: u32,
  // Candidate count in one dispatch row. The host uses this to support
  // rectangular dispatches without adding a per-invocation bounds check.
  candidates_per_row: u32,
  counter_low: u32,
  counter_high: u32,
}

struct ResultBuffer {
  // Claimed with an atomic exchange so only one invocation writes a result.
  // `atomic<u32>` has the same 4-byte size and alignment as a plain `u32`;
  // after the GPU finishes, Rust reads this slot as `GpuResult.found`.
  // The CPU re-derives the address from `nonce` before accepting it.
  found: atomic<u32>,
  nonce: array<u32, 8>,
}

// Keccak-f[1600] stores its 1600-bit state as 25 64-bit words, traditionally
// called lanes. These are Keccak state lanes, not GPU execution lanes. WGSL
// does not expose portable u64 arithmetic across all target backends we care
// about, so each Keccak lane is represented as two u32 halves.
struct U64 {
  lo: u32,
  hi: u32,
}

@group(0) @binding(0) var<storage, read> input: Input;
@group(0) @binding(1) var<storage, read_write> result: ResultBuffer;

// Basic bitwise operations on emulated 64-bit Keccak lanes.
fn xor64(a: U64, b: U64) -> U64 {
  return U64(a.lo ^ b.lo, a.hi ^ b.hi);
}

fn and64(a: U64, b: U64) -> U64 {
  return U64(a.lo & b.lo, a.hi & b.hi);
}

fn not64(a: U64) -> U64 {
  return U64(~a.lo, ~a.hi);
}

// Rotate an emulated 64-bit lane left by Keccak's fixed rotation offsets.
fn rotl64(a: U64, n: u32) -> U64 {
  if (n == 0u) {
    return a;
  }
  if (n < 32u) {
    return U64((a.lo << n) | (a.hi >> (32u - n)), (a.hi << n) | (a.lo >> (32u - n)));
  }
  if (n == 32u) {
    return U64(a.hi, a.lo);
  }

  let m = n - 32u;
  return U64((a.hi << m) | (a.lo >> (32u - m)), (a.lo << m) | (a.hi >> (32u - m)));
}

// Keccak round constants RC[0..24] for the iota step (FIPS 202 sec. 3.2.5,
// table at the end of the section). These values are identical between
// original Keccak and the SHA-3 family; only the padding rule differs.
//
// Spec PDF: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
//
// Each constant is written as two u32 halves in little-endian byte order to
// match the `U64` lane representation above.
const ROUND_CONSTANTS: array<U64, 24> = array<U64, 24>(
  U64(0x00000001u, 0x00000000u),
  U64(0x00008082u, 0x00000000u),
  U64(0x0000808au, 0x80000000u),
  U64(0x80008000u, 0x80000000u),
  U64(0x0000808bu, 0x00000000u),
  U64(0x80000001u, 0x00000000u),
  U64(0x80008081u, 0x80000000u),
  U64(0x00008009u, 0x80000000u),
  U64(0x0000008au, 0x00000000u),
  U64(0x00000088u, 0x00000000u),
  U64(0x80008009u, 0x00000000u),
  U64(0x8000000au, 0x00000000u),
  U64(0x8000808bu, 0x00000000u),
  U64(0x0000008bu, 0x80000000u),
  U64(0x00008089u, 0x80000000u),
  U64(0x00008003u, 0x80000000u),
  U64(0x00008002u, 0x80000000u),
  U64(0x00000080u, 0x80000000u),
  U64(0x0000800au, 0x00000000u),
  U64(0x8000000au, 0x80000000u),
  U64(0x80008081u, 0x80000000u),
  U64(0x00008080u, 0x80000000u),
  U64(0x80000001u, 0x00000000u),
  U64(0x80008008u, 0x80000000u),
);

// Reverse byte order for one 32-bit half of the counter. The input buffer keeps
// the counter as numeric low/high words; this conversion happens only at the
// boundary where that number becomes bytes in the final saltNonce.
fn bswap32(x: u32) -> u32 {
  return ((x & 0x000000ffu) << 24u)
    | ((x & 0x0000ff00u) << 8u)
    | ((x & 0x00ff0000u) >> 8u)
    | ((x & 0xff000000u) >> 24u);
}

fn shift_join(a: u32, b: u32) -> u32 {
  // CREATE2's 85-byte preimage starts with one leading 0xff byte, so every
  // following field is shifted by one byte. This joins the last byte of `a`
  // with the first three bytes of `b`.
  //
  // Example with little-endian words:
  //   a = 0xDDCCBBAA, bytes AA BB CC DD
  //   b = 0x44332211, bytes 11 22 33 44
  //   shift_join(a, b) returns 0x332211DD, bytes DD 11 22 33
  return (a >> 24u) | ((b & 0x00ffffffu) << 8u);
}

// Low 32 bits of this invocation's 64-bit counter value.
fn counter_low(offset: u32) -> u32 {
  return input.counter_low + offset;
}

// High 32 bits of this invocation's 64-bit counter value, including carry from
// adding the invocation offset to the low half.
fn counter_high(offset: u32) -> u32 {
  let low = counter_low(offset);
  let carry = select(0u, 1u, low < input.counter_low);
  return input.counter_high + carry;
}

fn keccakf(state: ptr<function, array<U64, 25>>) {
  // Keccak-f[1600] permutation, 24 rounds. Each round applies five named steps
  // in fixed order: theta -> rho -> pi -> chi -> iota. This implementation
  // unrolls every step inside the round body because that was materially
  // faster than dynamic lane-index loops on Metal; the outer round loop is
  // kept to keep shader code size and register pressure bounded.
  //
  // Spec: FIPS 202 sec. 3.2 (https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf).
  //
  // WGSL note: `state` is a pointer to a function-local array, so each lane
  // access uses `(*state)[i]` rather than `state[i]`.
  for (var round = 0u; round < 24u; round = round + 1u) {
    var b: array<U64, 25>;

    // theta: mix each column's parity into every lane in that column.
    let c0 = xor64(xor64(xor64(xor64((*state)[0], (*state)[5]), (*state)[10]), (*state)[15]), (*state)[20]);
    let c1 = xor64(xor64(xor64(xor64((*state)[1], (*state)[6]), (*state)[11]), (*state)[16]), (*state)[21]);
    let c2 = xor64(xor64(xor64(xor64((*state)[2], (*state)[7]), (*state)[12]), (*state)[17]), (*state)[22]);
    let c3 = xor64(xor64(xor64(xor64((*state)[3], (*state)[8]), (*state)[13]), (*state)[18]), (*state)[23]);
    let c4 = xor64(xor64(xor64(xor64((*state)[4], (*state)[9]), (*state)[14]), (*state)[19]), (*state)[24]);
    let d0 = xor64(c4, rotl64(c1, 1u));
    let d1 = xor64(c0, rotl64(c2, 1u));
    let d2 = xor64(c1, rotl64(c3, 1u));
    let d3 = xor64(c2, rotl64(c4, 1u));
    let d4 = xor64(c3, rotl64(c0, 1u));

    (*state)[0] = xor64((*state)[0], d0);
    (*state)[5] = xor64((*state)[5], d0);
    (*state)[10] = xor64((*state)[10], d0);
    (*state)[15] = xor64((*state)[15], d0);
    (*state)[20] = xor64((*state)[20], d0);
    (*state)[1] = xor64((*state)[1], d1);
    (*state)[6] = xor64((*state)[6], d1);
    (*state)[11] = xor64((*state)[11], d1);
    (*state)[16] = xor64((*state)[16], d1);
    (*state)[21] = xor64((*state)[21], d1);
    (*state)[2] = xor64((*state)[2], d2);
    (*state)[7] = xor64((*state)[7], d2);
    (*state)[12] = xor64((*state)[12], d2);
    (*state)[17] = xor64((*state)[17], d2);
    (*state)[22] = xor64((*state)[22], d2);
    (*state)[3] = xor64((*state)[3], d3);
    (*state)[8] = xor64((*state)[8], d3);
    (*state)[13] = xor64((*state)[13], d3);
    (*state)[18] = xor64((*state)[18], d3);
    (*state)[23] = xor64((*state)[23], d3);
    (*state)[4] = xor64((*state)[4], d4);
    (*state)[9] = xor64((*state)[9], d4);
    (*state)[14] = xor64((*state)[14], d4);
    (*state)[19] = xor64((*state)[19], d4);
    (*state)[24] = xor64((*state)[24], d4);

    // rho + pi: rotate each lane left by a fixed offset (rho) and move it to
    // a new position in the state (pi). Both tables are from FIPS 202 sec.
    // 3.2.2/3.2.3; rotation offsets like 1, 62, 28, 27, ... are the rho
    // offset table, and the b[10]/b[20]/... destinations are the pi
    // permutation. The b array is the new state being assembled.
    b[0] = (*state)[0];
    b[10] = rotl64((*state)[1], 1u);
    b[20] = rotl64((*state)[2], 62u);
    b[5] = rotl64((*state)[3], 28u);
    b[15] = rotl64((*state)[4], 27u);
    b[16] = rotl64((*state)[5], 36u);
    b[1] = rotl64((*state)[6], 44u);
    b[11] = rotl64((*state)[7], 6u);
    b[21] = rotl64((*state)[8], 55u);
    b[6] = rotl64((*state)[9], 20u);
    b[7] = rotl64((*state)[10], 3u);
    b[17] = rotl64((*state)[11], 10u);
    b[2] = rotl64((*state)[12], 43u);
    b[12] = rotl64((*state)[13], 25u);
    b[22] = rotl64((*state)[14], 39u);
    b[23] = rotl64((*state)[15], 41u);
    b[8] = rotl64((*state)[16], 45u);
    b[18] = rotl64((*state)[17], 15u);
    b[3] = rotl64((*state)[18], 21u);
    b[13] = rotl64((*state)[19], 8u);
    b[14] = rotl64((*state)[20], 18u);
    b[24] = rotl64((*state)[21], 2u);
    b[9] = rotl64((*state)[22], 61u);
    b[19] = rotl64((*state)[23], 56u);
    b[4] = rotl64((*state)[24], 14u);

    // chi: nonlinear row mix. For each row of five lanes, replace each lane
    // with `lane XOR (NOT next_lane AND lane_after_that)`. This is the only
    // nonlinear step in the round (FIPS 202 sec. 3.2.4).
    (*state)[0] = xor64(b[0], and64(not64(b[1]), b[2]));
    (*state)[1] = xor64(b[1], and64(not64(b[2]), b[3]));
    (*state)[2] = xor64(b[2], and64(not64(b[3]), b[4]));
    (*state)[3] = xor64(b[3], and64(not64(b[4]), b[0]));
    (*state)[4] = xor64(b[4], and64(not64(b[0]), b[1]));
    (*state)[5] = xor64(b[5], and64(not64(b[6]), b[7]));
    (*state)[6] = xor64(b[6], and64(not64(b[7]), b[8]));
    (*state)[7] = xor64(b[7], and64(not64(b[8]), b[9]));
    (*state)[8] = xor64(b[8], and64(not64(b[9]), b[5]));
    (*state)[9] = xor64(b[9], and64(not64(b[5]), b[6]));
    (*state)[10] = xor64(b[10], and64(not64(b[11]), b[12]));
    (*state)[11] = xor64(b[11], and64(not64(b[12]), b[13]));
    (*state)[12] = xor64(b[12], and64(not64(b[13]), b[14]));
    (*state)[13] = xor64(b[13], and64(not64(b[14]), b[10]));
    (*state)[14] = xor64(b[14], and64(not64(b[10]), b[11]));
    (*state)[15] = xor64(b[15], and64(not64(b[16]), b[17]));
    (*state)[16] = xor64(b[16], and64(not64(b[17]), b[18]));
    (*state)[17] = xor64(b[17], and64(not64(b[18]), b[19]));
    (*state)[18] = xor64(b[18], and64(not64(b[19]), b[15]));
    (*state)[19] = xor64(b[19], and64(not64(b[15]), b[16]));
    (*state)[20] = xor64(b[20], and64(not64(b[21]), b[22]));
    (*state)[21] = xor64(b[21], and64(not64(b[22]), b[23]));
    (*state)[22] = xor64(b[22], and64(not64(b[23]), b[24]));
    (*state)[23] = xor64(b[23], and64(not64(b[24]), b[20]));
    (*state)[24] = xor64(b[24], and64(not64(b[20]), b[21]));

    // iota: XOR a per-round constant into lane 0 to break round symmetry.
    (*state)[0] = xor64((*state)[0], ROUND_CONSTANTS[round]);
  }
}

fn keccak_safe_salt(offset: u32) -> array<u32, 8> {
  var state: array<U64, 25>;

  // SafeProxyFactory does not use the user-provided salt nonce directly. It
  // first hashes `keccak(initializer) || saltNonce`, then uses that digest as
  // the CREATE2 salt. This is the Safe-specific step that a generic CREATE2
  // vanity miner would miss.
  //
  // The input is exactly 64 bytes, which fits in one Keccak rate block of
  // 136 bytes (rate r = 1088 bits for keccak256). Original Keccak (Ethereum's
  // `keccak256`) uses pad10*1 padding written as bytes: a `0x01` right after
  // the message and a `0x80` at the last byte of the rate block (here, bytes
  // 64 and 135). This is *not* SHA-3, which would use `0x06` instead of
  // `0x01` for domain separation.
  //
  // Padding reference: https://keccak.team/keccak_specs_summary.html.
  //
  // WGSL gives `state` its zero value before these assignments run. That means
  // every Keccak lane starts as zero, and the code below writes only the lanes
  // that hold message bytes or padding bytes.
  state[0] = U64(input.initializer_hash[0], input.initializer_hash[1]);
  state[1] = U64(input.initializer_hash[2], input.initializer_hash[3]);
  state[2] = U64(input.initializer_hash[4], input.initializer_hash[5]);
  state[3] = U64(input.initializer_hash[6], input.initializer_hash[7]);
  state[4] = U64(input.nonce_prefix[0], input.nonce_prefix[1]);
  state[5] = U64(input.nonce_prefix[2], input.nonce_prefix[3]);
  state[6] = U64(input.nonce_prefix[4], input.nonce_prefix[5]);
  // state[7] holds preimage bytes 56..63 - the 8-byte counter tail of the
  // saltNonce. Safe expects that tail in big-endian byte order:
  //
  //   counter 0x0102030405060708 -> nonce bytes 01 02 03 04 05 06 07 08
  //
  // Keccak lanes read each u32 as little-endian bytes. Writing counter_high
  // directly into lane.lo would produce 04 03 02 01, so each 32-bit half is
  // byte-swapped before it enters the lane. The high half still lands in lo
  // because lane.lo covers bytes 56..59; the low half lands in hi because
  // lane.hi covers bytes 60..63.
  state[7] = U64(bswap32(counter_high(offset)), bswap32(counter_low(offset)));
  state[8] = U64(0x00000001u, 0x00000000u);
  state[16] = U64(0x00000000u, 0x80000000u);
  keccakf(&state);

  return array<u32, 8>(
    state[0].lo,
    state[0].hi,
    state[1].lo,
    state[1].hi,
    state[2].lo,
    state[2].hi,
    state[3].lo,
    state[3].hi,
  );
}

fn keccak_create2(safe_salt: array<u32, 8>) -> array<u32, 8> {
  var state: array<U64, 25>;

  // CREATE2 preimage:
  //   0xff || factory(20) || safe_salt(32) || init_code_hash(32)
  //
  // That is 85 bytes, still one Keccak rate block (136 bytes). Because the
  // first byte is 0xff, every following component starts at byte offset 1;
  // `shift_join` assembles those unaligned bytes directly into Keccak lanes.
  //
  // Padding follows the same original-Keccak rule used in `keccak_safe_salt`:
  // a `0x01` byte at offset 85 (right after the message) and a `0x80` byte at
  // offset 135 (last byte of the rate block). See
  // https://keccak.team/keccak_specs_summary.html.
  //
  // WGSL gives `state` its zero value before these assignments run. That means
  // every Keccak lane starts as zero, and the code below writes only the lanes
  // that hold message bytes or padding bytes.
  // state[0]: byte 0 = 0xff (the CREATE2 leading byte), then the low 3 bytes
  // of factory[0] at offsets 1..3. shift_join handles every later lane, but
  // this first lane is asymmetric because of that 0xff.
  state[0] = U64(0x000000ffu | ((input.factory[0] & 0x00ffffffu) << 8u), shift_join(input.factory[0], input.factory[1]));
  state[1] = U64(shift_join(input.factory[1], input.factory[2]), shift_join(input.factory[2], input.factory[3]));
  state[2] = U64(shift_join(input.factory[3], input.factory[4]), (input.factory[4] >> 24u) | ((safe_salt[0] & 0x00ffffffu) << 8u));
  state[3] = U64(shift_join(safe_salt[0], safe_salt[1]), shift_join(safe_salt[1], safe_salt[2]));
  state[4] = U64(shift_join(safe_salt[2], safe_salt[3]), shift_join(safe_salt[3], safe_salt[4]));
  state[5] = U64(shift_join(safe_salt[4], safe_salt[5]), shift_join(safe_salt[5], safe_salt[6]));
  state[6] = U64(shift_join(safe_salt[6], safe_salt[7]), (safe_salt[7] >> 24u) | ((input.init_code_hash[0] & 0x00ffffffu) << 8u));
  state[7] = U64(shift_join(input.init_code_hash[0], input.init_code_hash[1]), shift_join(input.init_code_hash[1], input.init_code_hash[2]));
  state[8] = U64(shift_join(input.init_code_hash[2], input.init_code_hash[3]), shift_join(input.init_code_hash[3], input.init_code_hash[4]));
  state[9] = U64(shift_join(input.init_code_hash[4], input.init_code_hash[5]), shift_join(input.init_code_hash[5], input.init_code_hash[6]));
  // state[10]: bytes 80..83 are the next 4 bytes of init_code_hash, byte 84 is
  // its final byte (last byte of the 85-byte preimage), and the | 0x00000100u
  // places the 0x01 Keccak padding byte at preimage offset 85.
  state[10] = U64(shift_join(input.init_code_hash[6], input.init_code_hash[7]), (input.init_code_hash[7] >> 24u) | 0x00000100u);
  state[16] = U64(0x00000000u, 0x80000000u);
  keccakf(&state);

  return array<u32, 8>(
    state[0].lo,
    state[0].hi,
    state[1].lo,
    state[1].hi,
    state[2].lo,
    state[2].hi,
    state[3].lo,
    state[3].hi,
  );
}

// Returns a u32 with the low `bytes` bytes set to 0xff and the rest zero. Used
// to compare a partial trailing word of the prefix against the digest.
fn prefix_mask(bytes: u32) -> u32 {
  switch bytes {
    case 1u: { return 0x000000ffu; }
    case 2u: { return 0x0000ffffu; }
    case 3u: { return 0x00ffffffu; }
    default: { return 0x00000000u; }
  }
}

fn matches_prefix(digest: array<u32, 8>) -> bool {
  // Ethereum addresses are the low 20 bytes of the Keccak digest, i.e.
  // digest[12..32]. Since `digest` is packed little-endian, that begins at word
  // index 3. Compare whole words first, then mask the final partial word.
  let words = input.prefix_len / 4u;
  for (var i = 0u; i < words; i = i + 1u) {
    if (digest[i + 3u] != input.prefix[i]) {
      return false;
    }
  }

  let remainder = input.prefix_len & 3u;
  if (remainder == 0u) {
    return true;
  }

  let mask = prefix_mask(remainder);
  return (digest[words + 3u] & mask) == (input.prefix[words] & mask);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // WebGPU calls columns x and rows y. The host writes the candidate count per
  // row here so rectangular dispatches still produce dense candidate offsets
  // without an `offset >= batch_size` branch.
  let offset = id.x + id.y * input.candidates_per_row;
  let safe_salt = keccak_safe_salt(offset);
  let digest = keccak_create2(safe_salt);
  if (!matches_prefix(digest)) {
    return;
  }

  // Many invocations may match at once. atomicExchange writes 1u into
  // result.found and returns whatever was there before. Whichever invocation
  // sees a return value of 0 is the first to claim the slot, so it writes the
  // nonce. Later claimants see 1 and skip the write.
  if (atomicExchange(&result.found, 1u) == 0u) {
    result.nonce[0] = input.nonce_prefix[0];
    result.nonce[1] = input.nonce_prefix[1];
    result.nonce[2] = input.nonce_prefix[2];
    result.nonce[3] = input.nonce_prefix[3];
    result.nonce[4] = input.nonce_prefix[4];
    result.nonce[5] = input.nonce_prefix[5];
    result.nonce[6] = bswap32(counter_high(offset));
    result.nonce[7] = bswap32(counter_low(offset));
  }
}
