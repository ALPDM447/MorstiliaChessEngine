//! Reading the `.nnue` container format.
//!
//! The file layout (all integers little-endian) is
//!
//! ```text
//! header      u32  version (0x6A448AFA)
//!            u32  network hash  (must equal Network::HASH)
//!            u32  description length
//!            ...  description bytes (UTF-8, informational only)
//! section     u32  FeatureTransformer hash  (0xCB685313)
//!            LEB128 biases        i16[1024]
//!            raw    threatWeights  i8 [59808 * 1024]
//!            LEB128 threatPsqt    i32[59808 * 8]
//!            raw    ppWeights      i8 [4560 * 1024]
//!            LEB128 ppPsqt        i32[4560 * 8]
//!            LEB128 weights       i16[1024 * 22528]
//!            LEB128 psqtWeights   i32[8 * 22528]
//! section ×8 u32  NetworkArchitecture hash (0x63337116)
//!                 LEB128 fc_0 biases i32[32] / weights i8[1024*32]
//!                 LEB128 fc_1 biases i32[32] / weights i8[64*32]
//!                 LEB128 fc_2 biases i32[1]  / weights i8[128*1]
//! eof
//! ```
//!
//! Every section starts with a hash of the *code* that interprets it, so a net
//! built for a different feature set or a different network shape is rejected
//! before a single weight is read. Every `LEB128` block additionally carries a
//! 19-byte magic string and the byte count of the compressed payload, which must
//! be consumed exactly.

use std::fmt;
use std::io::{self, Read};

/// Container version written by the NNUE trainer.
pub const VERSION: u32 = 0x6A44_8AFA;

/// Magic string that prefixes every LEB128-compressed block (without the NUL).
pub const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

/// Upper bound on a net description, so a corrupt length cannot make us try to
/// allocate gigabytes.
const MAX_DESCRIPTION: usize = 1 << 16;

/// Internal read buffer. 8 KiB is what Stockfish uses; large enough to keep the
/// per-value LEB128 loop cheap, small enough to be irrelevant.
const BUF_SIZE: usize = 8192;

/// A load failure. Kept deliberately small: every variant is a "this net cannot
/// be used by this engine" condition, and all of them are reported the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError {
    pub context: &'static str,
    pub reason: &'static str,
}

impl FormatError {
    /// A load failure. `context` names the section being read, `reason` why it
    /// was rejected.
    pub const fn new(context: &'static str, reason: &'static str) -> Self {
        FormatError { context, reason }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.context, self.reason)
    }
}

impl std::error::Error for FormatError {}

pub type Result<T> = std::result::Result<T, FormatError>;

/// Integer types that can be read as raw little-endian arrays.
pub trait LeBytes: Copy {
    const SIZE: usize;
    fn from_le_bytes(b: &[u8]) -> Self;
}

impl LeBytes for i8 {
    const SIZE: usize = 1;
    #[inline]
    fn from_le_bytes(b: &[u8]) -> Self {
        b[0] as i8
    }
}

impl LeBytes for i16 {
    const SIZE: usize = 2;
    #[inline]
    fn from_le_bytes(b: &[u8]) -> Self {
        i16::from_le_bytes([b[0], b[1]])
    }
}

impl LeBytes for i32 {
    const SIZE: usize = 4;
    #[inline]
    fn from_le_bytes(b: &[u8]) -> Self {
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

/// A buffered, bounds-checked reader over the net file.
///
/// Errors are always `InvalidData`: a net that cannot be parsed is not an I/O
/// failure, it is an incompatible net.
pub struct Reader<R: Read> {
    inner: R,
    buf: Box<[u8; BUF_SIZE]>,
    pos: usize,
    end: usize,
    /// Set once the underlying stream is exhausted, so `eof()` can tell a clean
    /// end-of-file apart from a truncated read.
    at_eof: bool,
}

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self {
        Reader {
            inner,
            buf: Box::new([0u8; BUF_SIZE]),
            pos: 0,
            end: 0,
            at_eof: false,
        }
    }

    /// Reads one section hash and checks it against the expected value.
    pub fn read_section_hash(&mut self, expected: u32, context: &'static str) -> Result<u32> {
        let h = self.read_u32()?;
        if h != expected {
            return Err(FormatError::new(
                context,
                "net is incompatible with this engine (section hash mismatch)",
            ));
        }
        Ok(h)
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads `n` raw little-endian bytes, bypassing the buffer for bulk data.
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        self.read_exact(&mut out, "truncated net")?;
        Ok(out)
    }

    /// Fills `out` completely or fails.
    pub fn read_exact(&mut self, out: &mut [u8], what: &'static str) -> Result<()> {
        // Drain whatever is already buffered.
        let avail = self.end - self.pos;
        let take = avail.min(out.len());
        out[..take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
        self.pos += take;
        if take == out.len() {
            return Ok(());
        }
        self.inner
            .read_exact(&mut out[take..])
            .map_err(|_| FormatError::new("truncated net", what))?;
        Ok(())
    }

    /// Reads a raw little-endian array, element by element so the file's byte
    /// order does not depend on the host's.
    pub fn read_le_array<T: LeBytes>(&mut self, out: &mut [T], what: &'static str) -> Result<()> {
        let mut raw = [0u8; 4096];
        let mut done = 0usize;
        while done < out.len() {
            let want = (out.len() - done).min(raw.len() / T::SIZE);
            let n = want * T::SIZE;
            self.read_exact(&mut raw[..n], what)?;
            for i in 0..want {
                let o = i * T::SIZE;
                out[done + i] = T::from_le_bytes(&raw[o..o + T::SIZE]);
            }
            done += want;
        }
        Ok(())
    }

    /// Reads a LEB128-compressed block of `out.len()` signed values.
    ///
    /// The decoding is Stockfish's, verbatim: the payload is read in 7-bit
    /// groups, the shift wraps at 32 bits, and a value whose last byte has bit
    /// `0x40` set (and which did not overflow 32 bits) is sign-extended. The
    /// declared byte count must be exhausted exactly.
    pub fn read_leb128<T: LeBytes>(&mut self, out: &mut [T], what: &'static str) -> Result<()> {
        let magic = self.read_bytes(LEB128_MAGIC.len()).map_err(|_| {
            FormatError::new(
                what,
                "missing COMPRESSED_LEB128 marker (section is not compressed)",
            )
        })?;
        if magic.as_slice() != LEB128_MAGIC {
            return Err(FormatError::new(
                what,
                "missing COMPRESSED_LEB128 marker (section is not compressed)",
            ));
        }
        let mut bytes_left = self.read_u32()?;

        let mut i = 0usize;
        let mut result: u32 = 0;
        let mut shift: u32 = 0;
        while i < out.len() {
            // The declared byte count is checked *before* the refill: a net
            // that declares fewer bytes than the block needs must be reported as
            // an over-long payload even when the reader is sitting exactly on a
            // buffer boundary (where the refill would otherwise report a
            // truncation first).
            if bytes_left == 0 {
                return Err(FormatError::new(
                    what,
                    "LEB128 payload longer than declared",
                ));
            }
            if self.pos == self.end && !self.refill(bytes_left)? {
                return Err(FormatError::new(what, "LEB128 payload ended too early"));
            }
            let byte = self.buf[self.pos];
            self.pos += 1;
            bytes_left -= 1;

            result |= u32::from(byte & 0x7f) << (shift % 32);
            shift += 7;

            if byte & 0x80 == 0 {
                // Sign-extend unless the value overflowed 32 bits.
                let value = if shift >= 32 || byte & 0x40 == 0 {
                    result
                } else {
                    result | !((1u32 << shift) - 1)
                };
                out[i] = T::from_le_bytes(&value.to_le_bytes());
                i += 1;
                result = 0;
                shift = 0;
            }
        }
        if bytes_left != 0 {
            return Err(FormatError::new(
                what,
                "LEB128 payload shorter than declared",
            ));
        }
        Ok(())
    }

    /// Refills the internal buffer with at most `limit` bytes.
    ///
    /// Returns `false` once the stream is dry, which is how a truncated net is
    /// told apart from a clean end-of-file.
    fn refill(&mut self, limit: u32) -> Result<bool> {
        if self.at_eof {
            return Ok(false);
        }
        debug_assert_eq!(self.pos, self.end, "refill only when the buffer is drained");
        let want = (limit as usize).min(BUF_SIZE);
        self.pos = 0;
        self.end = 0;
        if want == 0 {
            self.at_eof = true;
            return Ok(false);
        }
        let mut n = 0usize;
        while n < want {
            match self.inner.read(&mut self.buf[n..want]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    self.at_eof = true;
                    break;
                }
            }
        }
        self.end = n;
        if n == 0 {
            self.at_eof = true;
            return Ok(false);
        }
        Ok(true)
    }

    /// Peeks whether the stream is positioned exactly at the end of the file.
    ///
    /// Stockfish requires this after the last section, so trailing garbage is
    /// rejected just as firmly as a truncated net.
    pub fn expect_eof(&mut self, what: &'static str) -> Result<()> {
        if self.pos < self.end {
            return Err(FormatError::new(what, "unexpected trailing data"));
        }
        self.pos = 0;
        self.end = 0;
        if !self.at_eof {
            self.at_eof = true;
            let mut probe = [0u8; 1];
            loop {
                match self.inner.read(&mut probe) {
                    Ok(0) => break,
                    Ok(_) => return Err(FormatError::new(what, "unexpected trailing data")),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => return Err(FormatError::new(what, "unexpected trailing data")),
                }
            }
        }
        Ok(())
    }
}

/// The network header: version, structural hash and the trainer's description.
#[derive(Debug, Clone)]
pub struct Header {
    pub network_hash: u32,
    pub description: String,
}

impl<R: Read> Reader<R> {
    /// Reads and validates the file header.
    pub fn read_header(&mut self, expected_network_hash: u32) -> Result<Header> {
        let version = self.read_u32().map_err(|_| {
            FormatError::new("net header", "file is too short to contain a version")
        })?;
        if version != VERSION {
            return Err(FormatError::new(
                "net header",
                "unsupported net version (expected 0x6A448AFA)",
            ));
        }
        let network_hash = self.read_u32().map_err(|_| {
            FormatError::new("net header", "file is too short to contain a network hash")
        })?;
        if network_hash != expected_network_hash {
            return Err(FormatError::new(
                "net header",
                "network architecture hash mismatch (net built for a different engine)",
            ));
        }
        let desc_len = self
            .read_u32()
            .map_err(|_| FormatError::new("net header", "file is truncated in the description"))?
            as usize;
        if desc_len > MAX_DESCRIPTION {
            return Err(FormatError::new(
                "net header",
                "implausible description length (corrupt file)",
            ));
        }
        let raw = self
            .read_bytes(desc_len)
            .map_err(|_| FormatError::new("net header", "file is truncated in the description"))?;
        let description = String::from_utf8_lossy(&raw).into_owned();
        Ok(Header {
            network_hash,
            description,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(bytes: &[u8]) -> Reader<&[u8]> {
        Reader::new(bytes)
    }

    /// Builds a valid LEB128 block the way the trainer's writer does.
    fn leb_block<T: LeBytes, I: IntoIterator<Item = i32>>(values: I) -> Vec<u8> {
        let mut payload = Vec::new();
        for v in values {
            let mut value = v as i32;
            loop {
                let byte = (value & 0x7f) as u8;
                value >>= 7;
                if (byte & 0x40) == 0 {
                    if value == 0 {
                        payload.push(byte);
                        break;
                    }
                } else if value == -1 {
                    payload.push(byte);
                    break;
                }
                payload.push(byte | 0x80);
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(LEB128_MAGIC);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn leb128_round_trips_positive_negative_and_extreme_values() {
        // i16 range edges plus values needing 2 and 3 bytes.
        let values: [i32; 10] = [0, 1, -1, 63, 64, -64, -65, 8191, -8192, i16::MIN as i32];
        let block = leb_block::<i16, _>(values);
        let mut r = reader(&block);
        let mut out = vec![0i16; values.len()];
        r.read_leb128(&mut out, "test").unwrap();
        assert_eq!(out, values.map(|v| v as i16));
    }

    #[test]
    fn leb128_i32_uses_more_than_two_bytes() {
        let values: [i32; 6] = [0, 1, -1, 1 << 20, -(1 << 20), 0x3fff_ffff];
        let block = leb_block::<i32, _>(values);
        let mut r = reader(&block);
        let mut out = vec![0i32; values.len()];
        r.read_leb128(&mut out, "test").unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn leb128_rejects_a_missing_magic_string() {
        let mut r = reader(b"NOT_COMPRESSED_LEB128\0\0\0\0");
        let mut out = [0i16; 1];
        let err = r.read_leb128(&mut out, "test").unwrap_err();
        assert!(err.to_string().contains("COMPRESSED_LEB128"), "{err}");
    }

    #[test]
    fn leb128_rejects_a_wrong_byte_count() {
        // Claim one byte more than the payload holds.
        let mut block = leb_block::<i16, _>([1i32, 2]);
        let n = u32::from_le_bytes([
            block[LEB128_MAGIC.len()],
            block[LEB128_MAGIC.len() + 1],
            block[LEB128_MAGIC.len() + 2],
            block[LEB128_MAGIC.len() + 3],
        ]);
        block[LEB128_MAGIC.len()] = (n + 1) as u8;
        let mut r = reader(&block);
        let mut out = [0i16; 2];
        let err = r.read_leb128(&mut out, "test").unwrap_err();
        assert!(err.to_string().contains("shorter than declared"), "{err}");

        // ... and one byte less.
        let mut block = leb_block::<i16, _>([1i32, 2]);
        block[LEB128_MAGIC.len()] = (n - 1) as u8;
        let mut r = reader(&block);
        let err = r.read_leb128::<i16>(&mut out, "test").unwrap_err();
        assert!(err.to_string().contains("longer than declared"), "{err}");
    }

    #[test]
    fn leb128_rejects_a_truncated_payload() {
        let block = leb_block::<i16, _>([1i32, 2, 3]);
        let mut r = reader(&block[..block.len() - 1]);
        let mut out = [0i16; 3];
        let err = r.read_leb128(&mut out, "test").unwrap_err();
        assert!(err.to_string().contains("ended too early"), "{err}");
    }

    #[test]
    fn header_rejects_a_bad_version_or_hash() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x1234_5678u32.to_le_bytes());
        bytes.extend_from_slice(&0xA85B_2205u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut r = reader(&bytes);
        let err = r.read_header(0xA85B_2205).unwrap_err();
        assert!(err.to_string().contains("version"), "{err}");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut r = reader(&bytes);
        let err = r.read_header(0xA85B_2205).unwrap_err();
        assert!(err.to_string().contains("architecture hash"), "{err}");
    }

    #[test]
    fn header_round_trips_the_description() {
        let desc = "nn-1a298aa575a0.nnue";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0xA85B_2205u32.to_le_bytes());
        bytes.extend_from_slice(&(desc.len() as u32).to_le_bytes());
        bytes.extend_from_slice(desc.as_bytes());
        let mut r = reader(&bytes);
        let h = r.read_header(0xA85B_2205).unwrap();
        assert_eq!(h.network_hash, 0xA85B_2205);
        assert_eq!(h.description, desc);
        r.expect_eof("test").unwrap();
    }

    #[test]
    fn trailing_data_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0xA85B_2205u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.push(0xff);
        let mut r = reader(&bytes);
        r.read_header(0xA85B_2205).unwrap();
        let err = r.expect_eof("test").unwrap_err();
        assert!(err.to_string().contains("trailing data"), "{err}");
    }

    #[test]
    fn raw_little_endian_arrays_are_decoded() {
        let bytes: Vec<u8> = (0..16u8).collect();
        let mut r = reader(&bytes);
        let mut out = [0i16; 8];
        r.read_le_array(&mut out, "test").unwrap();
        for (k, v) in out.iter().enumerate() {
            assert_eq!(*v, i16::from_le_bytes([bytes[2 * k], bytes[2 * k + 1]]));
        }
    }

    #[test]
    fn truncated_raw_array_is_an_error() {
        let r = reader(&[1u8, 2, 3]);
        let mut r = r;
        let mut out = [0i32; 4];
        let err = r.read_le_array(&mut out, "test").unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
    }
}
