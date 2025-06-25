use crate::{num::traits::BinaryInteger, traits::RandomNumberGenerator};

/// A system-based random number generator using platform-specific APIs.
///
/// `SystemRandomNumberGenerator` provides a cryptographically secure source
/// of randomness using the operating system's facilities.
///
/// - On **macOS**, it uses `arc4random_buf` to securely fill a buffer.
/// - On **Windows**, it uses `BCryptGenRandom` with the system-preferred RNG.
/// - On unsupported platforms, this generator currently does nothing (no-op).
///
/// This struct is designed to be simple and safe, abstracting away
/// the complexity of platform-specific random number generation.
///
/// # Examples
///
/// ```
/// use foundation::random::SystemRandomNumberGenerator;
///
/// let rng = SystemRandomNumberGenerator::new();
/// let mut buffer = [0u8; 32];
/// rng.fill_bytes(&mut buffer);
/// println!("Random bytes: {:?}", buffer);
/// ```
///
/// # Platform Support
///
/// | Platform | Source             | Status      |
/// |----------|--------------------|-------------|
/// | macOS    | `arc4random_buf`   | ✅ Supported |
/// | Windows  | `BCryptGenRandom`  | ✅ Supported |
/// | Others   | *(none)*           | ⚠️ No-op     |
///
/// # Panics
///
/// - On **Windows**, this method will panic if `BCryptGenRandom` fails.
#[derive(Debug, Clone, Copy)]
pub struct SystemRandomNumberGenerator;

impl SystemRandomNumberGenerator {
    /// Creates a new `SystemRandomNumberGenerator`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Fills the given buffer with random bytes using the system's RNG.
    ///
    /// On macOS, this uses `arc4random_buf` to securely fill the buffer.
    /// On other platforms, this function currently does nothing.
    ///
    /// # Parameters
    ///
    /// - `buf`: A mutable byte slice to fill with random data.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::random::SystemRandomNumberGenerator;
    ///
    /// let rng = SystemRandomNumberGenerator::new();
    /// let mut data = [0u8; 16];
    /// rng.fill_bytes(&mut data);
    /// ```
    pub fn fill_bytes(&self, buf: &mut [u8]) {
        #[cfg(target_os = "windows")]
        unsafe {
            use core::ptr::null_mut;
            use windows_sys::Win32::Security::Cryptography::{
                BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom,
            };

            let status = BCryptGenRandom(
                null_mut(),
                buf.as_mut_ptr(),
                buf.len() as u32,
                BCRYPT_USE_SYSTEM_PREFERRED_RNG,
            );

            assert!(
                (status == 0),
                "BCryptGenRandom failed with status: 0x{status:X}"
            );
        }

        #[cfg(target_os = "macos")]
        {
            unsafe {
                use libc::arc4random_buf;

                arc4random_buf(buf.as_mut_ptr().cast(), buf.len());
            }
        }
    }
}

impl Default for SystemRandomNumberGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomNumberGenerator for SystemRandomNumberGenerator {
    fn next<T: Sized + Copy + Default>(&mut self) -> T {
        let mut val: T = Default::default();
        let ptr = (&raw mut val).cast::<u8>();
        let buf = unsafe { alloc::slice::from_raw_parts_mut(ptr, size_of::<T>()) };
        self.fill_bytes(buf);
        val
    }

    fn next_below<T>(&mut self, upper_bound: T) -> T
    where
        T: BinaryInteger + Default,
    {
        let value: T = self.next();
        value % upper_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_next_u32_returns_different_values() {
        let mut rng = SystemRandomNumberGenerator::new();
        let a: u32 = rng.next();
        let b: u32 = rng.next();
        // It's possible they match, but very unlikely!
        assert_ne!(
            a, b,
            "Two successive values should not be equal most of the time"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_next_below_respects_upper_bound() {
        let mut rng = SystemRandomNumberGenerator::new();
        for _ in 0..100 {
            let val: u32 = rng.next_below(100);
            assert!(val < 100, "Value {val} is not less than 100");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_fill_bytes_fills_buffer() {
        let rng = SystemRandomNumberGenerator::new();
        let mut buffer = [0u8; 32];
        rng.fill_bytes(&mut buffer);
        assert!(
            buffer.iter().any(|&b| b != 0),
            "Buffer contains only zeroes"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_randomness_stays_within_bounds_for_various_types() {
        let mut rng = SystemRandomNumberGenerator::new();
        let max = 250u8;
        for _ in 0..100 {
            let x: u8 = rng.next_below(max);
            assert!(x < max);
        }

        let max = 10_000u16;
        for _ in 0..100 {
            let x: u16 = rng.next_below(max);
            assert!(x < max);
        }
    }
}
